# -*- coding: utf-8 -*-
"""
Multi-modal Video Summarization (TH output, transcript-first)
Workflow:
  1) (optional) Download audio + extract frames by scene-cut
  2) ASR (Whisper) -> transcription.txt
  3) Image captioning (Florence-2 by default) + optional OCR -> captions.json
  4) Merge by timestamp -> scene-level "facts" (for visual notes)
  5) Global summaries:
     - final_summary_transcript.txt              (TRANSCRIPT only)
     - final_summary_transcript_plus_visual.txt  (TRANSCRIPT primary + VISUAL evidence)
---------------------------------------------------------------
Reqs:
  - ffmpeg, ffprobe
  - pip install yt_dlp openai-whisper torch torchvision torchaudio
  - pip install transformers accelerate pillow
"""

import os, sys, functools
os.environ["HF_ATTENTION_BACKEND"] = "PYTORCH_EAGER"
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTORCH_ENABLE_FLASH_SDP"] = "0"

class _StderrLogger:
    def debug(self, msg):
        # เงียบ debug (หรือจะ log ไป stderr ก็ได้)
        pass
    def warning(self, msg):
        print(f"⚠️ {msg}", file=sys.stderr, flush=True)
    def error(self, msg):
        print(f"❌ {msg}", file=sys.stderr, flush=True)

import io, json, time, base64, shutil, subprocess, re, math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pythainlp.tokenize import word_tokenize

import requests
import torch
from PIL import Image
import whisper  # openai-whisper
import textwrap

from datetime import datetime
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook

EXCEL_LOG = "stat.xlsx"

_progress_fp = None
try:
    _progress_fp = os.fdopen(3, "w", buffering=1, encoding="utf-8")  # line-buffered
except Exception:
    _progress_fp = None

def send_progress(step: str, percent: int):
    if _progress_fp:
        _progress_fp.write(json.dumps({"type":"progress","step":step,"percent":percent}) + "\n")
        _progress_fp.flush()
    # ไม่แตะ stdout เด็ดขาด
    
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# ====== CONFIG ======
YOUTUBE_URL = "https://youtu.be/Rq7plosixd0?si=-Xw05o5mTZd-eTBt"
AUDIO_OUT = "audio.wav"
FRAMES_DIR = "frames"
SCENES_JSON = "scenes.json"
CAPTIONS_JSON = "captions.json"
SCENE_FACTS_JSON = "scene_facts.json"
TRANSCRIPT_TXT = "transcription.txt"
METRICS_JSON = globals().get("METRICS_JSON", None)
log = functools.partial(print, file=sys.stderr, flush=True)

LANGUAGE = "th"
WHISPER_MODEL = "large-v3-turbo"
ASR_DEVICE = "cpu"      
VL_DEVICE  = "cuda"     # ใช้กับ Florence เท่านั้น

VL_MODEL_NAME = "microsoft/Florence-2-base"
SCENE_THRESH = 0.6
ENABLE_OCR = False

# ใช้ 127.0.0.1 กันปัญหา IPv6/localhost บางเครื่อง
OLLAMA_API = os.environ.get("OLLAMA_API", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")  

# ===== NEW OUTPUT NAMES =====
DROPDOWN_JSON = "dropdown_items.json"           # สำหรับ UI dropdown
FINAL_TXT = "dropdown_list.txt"  # bullet รวม
FINAL_ARTICLE_TXT = "final_article_th.txt"      # บทความสั้นภาษาไทย

# ===== STRONG THAI-ONLY SYSTEM (อัปเดตให้เข้มแบบร้อยแก้วไทย) =====
SYSTEM_PROMPT_TH = (
    "คุณคือผู้สรุปข่าวภาษาไทย ใช้สำนวนข่าวเล่าเรื่อง กระชับ ชัดเจน ไม่ใช้หัวข้อย่อยหรือบูลเล็ต "
    "หลีกเลี่ยงการพูดซ้ำ และหากอ้างสิ่งที่ยืนยันด้วยภาพให้แทรก (จากภาพ) ภายในประโยค "
    "คงชื่อเฉพาะ/ตัวย่อ/ตัวเลขตามต้นฉบับ"
)

# ===== NEW: Generation presets (minimal-safe) =====
GEN_OPTS_QUALITY = {
    "temperature": 0.7,        # ใกล้ default ของ Ollama
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.15,    # ลดจาก 1.6 -> 1.15 เพื่อลดประหลาด/ซ้ำ
    "repeat_last_n": 256,      # สำคัญมากสำหรับกันวน
    "num_ctx": 8192,           # llama3 8B รองรับ ~8k
    "num_predict": 1024,       # เพดานกว้างพอสำหรับบทความสั้น
    "stop": ["<|eot_id|>", "</s>"],  # stop ของ Llama 3
}
GEN_OPTS_FAST = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.12,
    "repeat_last_n": 256,
    "num_ctx": 8192,
    "num_predict": 512,
    "stop": ["<|eot_id|>", "</s>"],
}

# ===== NEW: Word count helpers (TH/EN mix safe) =====
WS_SPLIT_RE = re.compile(r"[ \t\r\n]+")
def word_count_th(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    tokens = [tok for tok in WS_SPLIT_RE.split(t) if tok]
    if len(tokens) >= 50:
        return len(tokens)
    approx = max(len(t) // 5, len(tokens))
    return approx

def clamp_article_to_words(text: str, min_words: int, max_words: int) -> str:
    t = (text or "").strip()
    for _ in range(2):
        wc = word_count_th(t)
        if min_words <= wc <= max_words:
            return t
        if wc < min_words:
            prompt = (
                f"ขยายความเรียงภาษาไทยด้านล่างให้ยาวประมาณ {min_words}-{max_words} คำ "
                f"คงสำนวนและสาระเดิม หลีกเลี่ยงการใส่หัวข้อย่อยหรือเลขลิสต์:\n{t}"
            )
            t = ensure_thai(ollama_summarize(prompt, options=GEN_OPTS_QUALITY))
        else:
            prompt = (
                f"ย่อความเรียงภาษาไทยด้านล่างให้ยาวประมาณ {min_words}-{max_words} คำ "
                f"คงสำนวนและสาระเดิม หลีกเลี่ยงหัวข้อย่อย/เลขลิสต์:\n{t}"
            )
            t = ensure_thai(ollama_summarize(prompt, options=GEN_OPTS_QUALITY))
    return t

# ====== UTIL ======
def wrap_text(text: str, width: int = 100) -> str:
    """
    จัดข้อความให้มีความยาวไม่เกิน width ตัวอักษรต่อบรรทัด
    """
    # แปลงข้อความให้เป็น list ของบรรทัดที่จัดแล้ว
    wrapped_lines = textwrap.wrap(text, width=width, replace_whitespace=True, drop_whitespace=True)
    # รวมกลับเป็นข้อความใหม่ โดยคั่นด้วย '\n'
    return "\n".join(wrapped_lines)

def check_cmd(cmd: str):
    if shutil.which(cmd) is None:
        raise RuntimeError(f"❌ '{cmd}' not found in PATH.")

def run(cmd: List[str]):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
    return p.stdout

def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

THAI_RE = re.compile(r"[ก-๙]")
def looks_thai(s: str) -> bool:
    return bool(THAI_RE.search(s or ""))

# ===== NEW: sanitize options + healthcheck + fallback /api/chat =====
ALLOWED_OLLAMA_KEYS = {
    "temperature", "top_p", "top_k",
    "repeat_penalty", "repeat_last_n",
    "num_ctx", "num_predict",
    "stop", "seed",
}

def sanitize_ollama_options(opts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not opts:
        return {}
    return {k: v for k, v in opts.items() if k in ALLOWED_OLLAMA_KEYS}

def ollama_healthcheck(base: str) -> bool:
    url = base.rsplit("/", 1)[0] + "/tags"
    try:
        r = requests.get(url, timeout=5)
        return r.ok
    except Exception:
        return False

def ollama_ensure_model(model: str, base: str) -> None:
    url = base.rsplit("/", 1)[0] + "/pull"
    try:
        requests.post(url, json={"name": model}, timeout=60)
    except Exception:
        pass

def _post_json(url: str, payload: dict, stream: bool, timeout: int):
    last_exc = None
    for _ in range(2):
        try:
            resp = requests.post(url, json=payload, stream=stream, timeout=timeout)
            if resp.status_code == 405 and url.endswith("/generate"):
                return resp
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            stream = False
    if last_exc:
        raise last_exc

def ollama_summarize(
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    system: Optional[str] = None,
    stream: bool = False,           # ปิดสตรีมเพื่อความเสถียรและจับข้อความครบ
    timeout: int = 600,
) -> str:
    base = OLLAMA_API
    if system is None:
        system = SYSTEM_PROMPT_TH
    if not ollama_healthcheck(base):
        raise RuntimeError("❌ ติดต่อ Ollama ไม่ได้: ตรวจสอบว่า `ollama serve` รันอยู่ และพอร์ต 11434 เปิดอยู่")

    ollama_ensure_model(OLLAMA_MODEL, base)

    chat_base = base.rsplit("/", 1)[0] + "/chat"
    opts = sanitize_ollama_options(options)
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if opts:
        payload["options"] = opts

    resp = requests.post(chat_base, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    msg = (data.get("message") or {}).get("content", "")
    return (msg or "").strip()

def ensure_thai(text: str, max_chars: int = None) -> str:
    t = (text or "").strip()
    if looks_thai(t):
        if max_chars and len(t) > max_chars:
            t = t[:max_chars]
        return t
    p1 = (
        "แปล/เขียนข้อความต่อไปนี้เป็น 'ภาษาไทยล้วนเท่านั้น' "
        "ห้ามขึ้นต้นด้วยภาษาอังกฤษ ไม่ต้องคำนำ ไม่ต้องคำอธิบายเพิ่ม:\n"
        f"{t}"
    )
    out = (ollama_summarize(p1, options=GEN_OPTS_FAST) or "").strip()
    if not looks_thai(out):
        p2 = (
            "จงตอบเป็น 'ภาษาไทยล้วนเท่านั้น' ห้ามมีภาษาอังกฤษแม้แต่คำเดียว "
            "ห้ามใส่คำว่า Here is, Translation, หรือคำนำอื่น ๆ "
            "ให้ส่งเฉพาะเนื้อความที่แปลแล้วเท่านั้น:\n"
            f"{t}"
        )
        out = (ollama_summarize(p2, options=GEN_OPTS_FAST) or "").strip()
    if max_chars and len(out) > max_chars:
        out = out[:max_chars]
    return out

# ===== NEW: Thai sentence tools & de-dup =====
SENT_SPLIT_RE = re.compile(r"(?:\s*(?<=[\.!?…])\s+|\n+)")
ELLIPSIS_RE = re.compile(r"(\.{2,}|…{2,})")
MULTI_SPACE_RE = re.compile(r"\s+")
COMMON_FIXES = {
    "บริษาส": "บริษัท",
    "ดัชนิ": "ดัชนี",
    "ทรุ": "ทะลุ",
    "พิสุด": "พิสูจน์",
    "ตบแทน": "ตอบแทน",
    "กะจุกตัว": "กระจุกตัว",
    "ลองทุน": "ลงทุน",
}

def split_sentences_th(text: str) -> List[str]:
    t = text.strip()
    t = ELLIPSIS_RE.sub("…", t)
    parts = [p.strip() for p in SENT_SPLIT_RE.split(t) if p.strip()]
    return parts

def simple_tokenize_th(s: str) -> List[str]:
    s = re.sub(r"[^\wก-๙%\.:\-/]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.split()

def jaccard_sim(a: str, b: str) -> float:
    A, B = set(simple_tokenize_th(a)), set(simple_tokenize_th(b))
    if not A or not B:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()
    return len(A & B) / max(1, len(A | B))

def similarity(a: str, b: str) -> float:
    return jaccard_sim(a, b)

def dedup_sentences(text: str, thr: float = 0.88) -> str:
    sents = split_sentences_th(text)
    keep = []
    for s in sents:
        s_norm = MULTI_SPACE_RE.sub(" ", s)
        if not s_norm:
            continue
        if any(similarity(s_norm, k) >= thr or s_norm == k for k in keep):
            continue
        keep.append(s_norm)
    out = " ".join(keep)
    return MULTI_SPACE_RE.sub(" ", out).strip()

def basic_thai_fixes(text: str) -> str:
    out = text
    for wrong, right in COMMON_FIXES.items():
        out = out.replace(wrong, right)
    out = re.sub(r"(สัญญาณเตือนสีแดง)+", "สัญญาณเตือนสีแดง", out)
    out = re.sub(r"(บริษัทเทคโนโลยีใหญ่\s*10\s*บริษัท)+", "บริษัทเทคโนโลยีใหญ่ 10 บริษัท", out)
    return out

def enforce_paragraphs(text: str, min_para=2, max_para=4) -> str:
    t = MULTI_SPACE_RE.sub(" ", text).strip()
    words = t.split(" ")
    if len(words) < 120:
        return t
    chunks = []
    target = max(min_para, min(max_para, 3))
    step = max(1, len(words)//target)
    for i in range(target):
        start = i*step
        end = None if i == target-1 else (i+1)*step
        para = " ".join(words[start:end]).strip()
        if para:
            chunks.append(para)
    return "\n\n".join(chunks)

def polish_thai_article(text: str, min_words=300, max_words=400) -> str:
    t = basic_thai_fixes(text)
    t = dedup_sentences(t, thr=0.88)
    prompt = ("ปรับสำนวนไทยของย่อหน้าด้านล่างให้อ่านลื่นไหล ชัดเจน ไม่ซ้ำประโยค "
              "ห้ามเปลี่ยนสาระสำคัญ ห้ามทำเป็นลิสต์ ห้ามเพิ่มหัวข้อ:\n" + t)
    t2 = ensure_thai(ollama_summarize(prompt, options=GEN_OPTS_FAST))
    t2 = dedup_sentences(t2, thr=0.9)
    t2 = clamp_article_to_words(t2, min_words, max_words)
    t2 = enforce_paragraphs(t2, 2, 4)
    return t2

def needs_retry(text: str, min_words=300, max_words=400) -> bool:
    wc = word_count_th(text)
    if wc < min_words*0.9 or wc > max_words*1.1:
        return True
    sents = split_sentences_th(text)
    dup_count = 0
    for i in range(1, len(sents)):
        if similarity(sents[i], sents[i-1]) >= 0.92:
            dup_count += 1
    return dup_count >= 1

def normalize_transcript_for_summary(t: str) -> str:
    p = f"ปรับปรุงการสะกด เว้นวรรค และภาษาไทยให้ถูกต้อง โดยคงใจความเดิม:\n{t}"
    cleaned = ollama_summarize(p, options=GEN_OPTS_FAST)
    return ensure_thai(cleaned)

def delete_all_files_in_directory(directory_path):
    """
    Deletes all files within a specified directory.

    Args:
        directory_path (str): The path to the directory.
    """
    if not os.path.isdir(directory_path):
        log(f"Error: Directory '{directory_path}' does not exist.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):  # Ensure it's a file, not a subdirectory
            try:
                os.remove(file_path)
                log(f"Deleted: {file_path}")
            except OSError as e:
                log(f"Error deleting {file_path}: {e}")

def get_video_duration(video_path: str) -> float:
    """คืนค่าความยาววิดีโอเป็นวินาที (float)"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        return duration
    except Exception as e:
        log(f"⚠️ ไม่สามารถอ่านความยาววิดีโอได้: {e}")
        return 0.0

# ====== STEP 1: Download audio + scene-cut frames ======
COOKIES_FILE = os.environ.get("YDL_COOKIES")
BROWSER_FOR_COOKIES = os.environ.get("YDL_BROWSER", "chrome")
YDL_RETRIES = 3

def ydl_opts_common(outtmpl: str = "%(title).200B.%(ext)s"):
    opts = {
        "quiet": True,
        "no_warnings": True,
        "logger": _StderrLogger(),
        "noprogress": True,
        "noplaylist": True,
        "retries": YDL_RETRIES,
        "fragment_retries": YDL_RETRIES,
        "concurrent_fragment_downloads": 4,
        "throttledratelimit": 0,
        "geo_bypass": True,
        "nocheckcertificate": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "th,en-US;q=0.9,en;q=0.8",
        },
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
        "outtmpl": outtmpl,
    }
    if COOKIES_FILE and os.path.exists(COOKIES_FILE):
        opts["cookiefile"] = COOKIES_FILE
    return opts

def download_audio_wav_16k(url: str, out_path: str):
    import yt_dlp
    tmp_in = None
    opts = ydl_opts_common(outtmpl="tmp_audio_raw.%(ext)s")
    opts.update({
        "format": "bestaudio/best",
        "keepvideo": False,
        "merge_output_format": "m4a",
    })
    last_err = None
    for attempt in range(1, YDL_RETRIES + 1):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                tmp_in = ydl.prepare_filename(info)
            break
        except Exception as e:
            last_err = e
            log(f"⚠️ audio download attempt {attempt}/{YDL_RETRIES} failed: {e}")
            opts["extractor_args"]["youtube"]["player_client"] = ["web", "android"]
    if not tmp_in or not os.path.exists(tmp_in):
        raise RuntimeError(f"ดาวน์โหลดเสียงไม่สำเร็จ (403/บล็อค): {last_err}")
    run(["ffmpeg", "-y", "-i", tmp_in, "-ar", "16000", "-ac", "1", out_path])
    try: os.remove(tmp_in)
    except: pass
    log(f"✅ Audio saved -> {out_path}")

def download_video_file(url: str) -> str:
    import yt_dlp
    outtmpl = "tmp_video.%(ext)s"
    opts = ydl_opts_common(outtmpl=outtmpl)
    opts.update({
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
    })
    last_err = None
    for attempt in range(1, YDL_RETRIES + 1):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            break
        except Exception as e:
            last_err = e
            log(f"⚠️ video download attempt {attempt}/{YDL_RETRIES} failed: {e}")
            opts["extractor_args"]["youtube"]["player_client"] = ["web", "android"]
    candidates = [f for f in os.listdir(".") if f.startswith("tmp_video.") and
                  f.lower().endswith((".mp4", ".mkv", ".webm"))]
    if not candidates:
        raise RuntimeError(f"ดาวน์โหลดวิดีโอไม่สำเร็จ (403/บล็อค): {last_err}")
    return sorted(candidates, key=os.path.getsize, reverse=True)[0]

# ====== STEP 2: ASR (Whisper) ======
def transcribe_whisper(wav_path: str, model_name: str, language: str, device: str) -> str:
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(wav_path, language=language, fp16=(device=="cuda"), temperature=0.0, verbose=False,)
    text = (result["text"] or "").strip()
    text = ensure_thai(text)
    with open(TRANSCRIPT_TXT, "w", encoding="utf-8") as f: f.write(text)
    log("✅ Transcription done.")
    return text

def iapp_asr_api(wav_path: str, wav_name: str) -> str:
    url = "https://api.iapp.co.th/asr/v3"
    payload = {'use_asr_pro': '1', 'chunk_size': '7'} #Set to '1' for iApp ASR PRO
    files=[('file',(wav_name,open(wav_path,'rb'),'application/octet-stream'))]
    headers = {'apikey': 'demo'}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    data = json.loads(response.text)
    textlist = [item["text"] for item in data["output"]]
    text = " ".join(textlist)
    with open(TRANSCRIPT_TXT, "w", encoding="utf-8") as f: f.write(text)
    log("✅ Transcription done.")
    return text

# ====== STEP 3: Image Captioning (+ optional OCR) ======
def clean_tags(tags):
    out = []
    for t in (tags or []):
        t = str(t).strip()
        if not t or t.startswith("TAGS><loc_"): continue
        if looks_thai(t) and len(t) <= 20: out.append(t)
    return sorted(list(set(out)))[:8]

def translate_to_th(text: str, max_chars: int = 200) -> str:
    if not text or looks_thai(text): return text or ""
    prompt = f"แปลข้อความต่อไปนี้เป็นภาษาไทยแบบสั้น กระชับ ไม่เกิน {max_chars} อักขระ:\n{text}"
    out = ollama_summarize(prompt, options=GEN_OPTS_FAST)
    return (out or "").strip()

class VisionCaptioner:
    def __init__(self, model_name: str, device: str):
        import os, torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        # กัน TF โผล่มากวน
        os.environ["TRANSFORMERS_NO_TF"] = "1"
        os.environ["TRANSFORMERS_NO_FLAX"] = "1"
        # ปิดเส้นทาง SDPA/Flash ทั้งระบบ (สำคัญบนบางเวอร์ชัน)
        os.environ["PYTORCH_ENABLE_FLASH_SDP"] = "0"
        os.environ["PYTORCH_ENABLE_MEM_EFFICIENT_SDP"] = "0"
        os.environ["PYTORCH_FORCE_DISABLE_FUSED_ADAM"] = "1"

        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.backend = "florence"
        self.img_size = 448
        self.batch_size = 2
        self.max_new_tokens = 64

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation="eager",   # ⬅️ บังคับ eager
        )
        if self.device != "cuda":
            self.model.to("cpu")

        # --- กันพลาดเรื่อง SDPA/_supports_sdpa/use_cache ทุกชั้น ---
        try: self.model.eval()
        except: pass

        for obj in [self.model,
                    getattr(self.model, "language_model", None),
                    getattr(self.model, "model", None),
                    getattr(self.model, "vision_tower", None)]:
            if obj is None: 
                continue
            # ปิด cache
            try: obj.config.use_cache = False
            except: pass
            try: obj.generation_config.use_cache = False
            except: pass
            # บังคับ eager
            try: obj.config._attn_implementation = "eager"
            except: pass
            # กันโค้ดฝั่ง transformers ที่เช็ค field นี้
            if not hasattr(obj, "_supports_sdpa"):
                try: setattr(obj, "_supports_sdpa", False)
                except: pass

        log(f"✅ Florence-2 ready on {self.device} (eager attention, no cache)")

    @torch.inference_mode()
    def _florence_generate(self, imgs, task: str):
        if not isinstance(imgs, list):
            imgs = [imgs]
        with torch.autocast("cuda", enabled=self.device == "cuda"):
            inputs = self.processor(text=[task]*len(imgs), images=imgs,
                                    return_tensors="pt", padding=True).to(self.device)
            ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                early_stopping=True,
                use_cache=False, 
            )
        outs = []
        for i in range(len(imgs)):
            try:
                txt = self.processor.batch_decode(ids[i:i+1], skip_special_tokens=False)[0]
                post = self.processor.post_process_generation(txt, task=task,
                                                              image_size=(imgs[i].height, imgs[i].width))
                val = post.get(task) or post.get("description") or txt
                outs.append(str(val).strip())
            except Exception:
                outs.append(self.processor.batch_decode(ids[i:i+1], skip_special_tokens=True)[0].strip())
        return outs

    def caption_image(self, img: Image.Image) -> Dict[str, Any]:
        if self.backend != "florence":
            txt = "อธิบายภาพนี้เป็นภาษาไทยแบบสั้น กระชับ"
            inputs = self.processor(images=img.convert("RGB"), text=txt, return_tensors="pt").to(self.device)
            out_ids = self.model.generate(**inputs, max_new_tokens=90, do_sample=False, num_beams=1)
            text = self.processor.decode(out_ids[0], skip_special_tokens=True)
            if not looks_thai(text):
                text = translate_to_th(text, max_chars=180)
            return {"caption_short": text[:80], "caption_detailed": text, "tags": []}

        # Florence: caption เดียว (<CAPTION>) เพื่อความเร็ว
        imgs = [img.convert("RGB").resize((448, 448))]
        caps = self._florence_generate(imgs, "<CAPTION>")
        cap = caps[0] if caps else ""
        cap = cap.replace("Caption the image", "").replace("Describe", "").strip()
        if not looks_thai(cap):
            cap = translate_to_th(cap, max_chars=90)
        return {"caption_short": cap, "caption_detailed": cap, "tags": []}

    def run_ocr(self, img: Image.Image) -> str:
        return ""

def stream_scene_frames_and_caption(url: str,
                                    frames_dir: str,
                                    thresh: float,
                                    out_json: str,
                                    captioner: VisionCaptioner):
    os.makedirs(frames_dir, exist_ok=True)
    delete_all_files_in_directory(frames_dir)

    # ดาวน์โหลดวิดีโอเก็บเป็นไฟล์ชั่วคราว
    video_path = download_video_file(url)

    def _safe_unlink(p: str):
        try:
            if p and os.path.exists(p):
                os.remove(p)
                log(f"🧹 removed temp file: {p}")
        except Exception as e:
            log(f"⚠️ failed to remove temp file {p}: {e}")

    results = []
    next_id = 1
    proc = None
    try:
        # รัน ffmpeg แตกเฟรมฉาก + showinfo (stderr)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "info",
            "-i", video_path,
            "-vf", f"select='gt(scene,{thresh})',showinfo",
            "-vsync", "vfr",
            os.path.join(frames_dir, "scene_%06d.jpg")
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        while True:
            line = proc.stderr.readline()
            if not line and proc.poll() is not None:
                break
            if not line:
                continue

            if "showinfo" in line and "pts_time:" in line:
                # ดึง timestamp
                try:
                    ts = float(line.split("pts_time:")[1].split(" ")[0])
                except Exception:
                    ts = None

                # รอให้ไฟล์ภาพเขียนเสร็จ
                img_path = os.path.join(frames_dir, f"scene_{next_id:06d}.jpg")
                for _ in range(50):
                    if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                        break
                    time.sleep(0.02)

                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                    except Exception:
                        time.sleep(0.05)
                        img = Image.open(img_path).convert("RGB")

                    cap = captioner.caption_image(img)
                    results.append({
                        "ts": round(ts, 2) if ts is not None else None,
                        "frame": os.path.basename(img_path),
                        "caption_short": cap.get("caption_short", ""),
                        "caption_detailed": cap.get("caption_detailed", ""),
                        "tags": cap.get("tags", []),
                        "ocr_text": ""
                    })
                    log(f"🖼️ {os.path.basename(img_path)} @{ts:.2f}s -> captioned")

                    # ลบภาพเฟรมทันที ลด IO/พื้นที่
                    try: os.remove(img_path)
                    except: pass

                    next_id += 1

        # เขียนผล caption ทั้งหมด
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log(f"✅ Captions saved -> {out_json}")

    finally:
        # รอ ffmpeg ปิดให้เรียบร้อยก่อนค่อยลบวิดีโอชั่วคราว (กันไฟล์ล็อก)
        try:
            if proc is not None:
                proc.wait(timeout=5)
        except Exception:
            pass
        _safe_unlink(video_path)

# ====== STEP 4: Merge into scene-level facts ======
@dataclass
class SceneFacts:
    start: float
    end: float
    speech: str
    visual_caption: str
    ocr_text: str
    tags: List[str]

def split_text_to_scenes(text: str, scene_ts: List[float]) -> List[SceneFacts]:
    if not scene_ts: return [SceneFacts(0.0, 0.0, text, "", "", [])]
    scene_ts = sorted(scene_ts)
    ts_bounds = scene_ts + [scene_ts[-1] + 99999]
    chunks = []
    n = len(scene_ts)
    avg_len = max(1, len(text)//max(1,n))
    for i in range(n):
        seg = text[i*avg_len:(i+1)*avg_len] if i < n-1 else text[i*avg_len:]
        chunks.append(seg.strip())
    facts: List[SceneFacts] = []
    for i in range(n):
        s = scene_ts[i]; e = ts_bounds[i+1]
        facts.append(SceneFacts(start=s, end=e, speech=chunks[i], visual_caption="", ocr_text="", tags=[]))
    return facts

def enrich_scenes_with_captions(facts: List[SceneFacts], captions: List[Dict[str,Any]]) -> List[SceneFacts]:
    for sc in facts:
        candidates = [c for c in captions if sc.start - 0.5 <= c["ts"] <= sc.end + 0.5] or \
                     sorted(captions, key=lambda c: abs(c["ts"]-sc.start))[:1]
        vc, ocrs, tags = [], [], []
        for c in candidates:
            if c.get("caption_detailed"): vc.append(c["caption_detailed"])
            elif c.get("caption_short"): vc.append(c["caption_short"])
            if c.get("ocr_text"): ocrs.append(c["ocr_text"])
            tags.extend(c.get("tags", []))
        sc.visual_caption = ensure_thai(" ".join(vc).strip()) if vc else ""
        sc.ocr_text = ensure_thai(" ".join(ocrs).strip()) if ocrs else ""
        sc.tags = sorted(list(set(tags)))
    return facts

# ====== STEP 5: Visual Evidence (domain-agnostic) ======
EVIDENCE_POLICY = "strict"    # "strict" | "balanced"
VISUAL_MAX_EVIDENCE = None    # None -> dynamic by #scenes

WEIGHT_MAP = {
    "number": 3, "percent": 3, "timecode": 3, "date": 3,
    "unit": 2, "currency": 2, "keyword_on_screen": 2,
    "short_len": 1, "has_ocr": 2, "url_or_id": 1, "all_caps_token": 1,
}
PROMO_VAGUE_TERMS = re.compile(
    r"(โปรโมท|ยอดนิยม|สุดยอด|อลังการ|ตื่นเต้น|มันส์|เจ๋งมาก|ห้ามพลาด|"
    r"amazing|awesome|must[- ]see|incredible|epic|promo|trailer|official)",
    flags=re.I
)
RE_NUMBER   = re.compile(r"\d")
RE_PERCENT  = re.compile(r"\d+\s*%")
RE_TIMECODE = re.compile(r"\b(?:\d{1,2}:){1,2}\d{2}\b")
RE_DATE     = re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:\d{1,2}[-/]){1,2}\d{2,4})\b")
RE_UNIT     = re.compile(r"\b(?:kg|g|km|m|cm|mm|mb|gb|tb|fps|hz|px|ms|s|min|hr|ชั่วโมง|นาที|วินาที|บาท|กก|กม)\b", re.I)
RE_CURRENCY = re.compile(r"[€£$¥฿]|(?:USD|THB|JPY|EUR)\b", re.I)
RE_URL_ID   = re.compile(r"(https?://\S+)|\b[A-Z0-9]{6,}\b")
RE_ALLCAPS  = re.compile(r"\b[A-Z]{2,}\b")
RE_HEADERY  = re.compile(r"\b(introduction|overview|summary|สรุป|บทนำ|หัวข้อ|สารบัญ|สไลด์|ผลลัพธ์|วิธีใช้|ข้อควรทราบ)\b", re.I)

def _detect_signals(txt: str, has_ocr: bool) -> Dict[str, bool]:
    return {
        "number": bool(RE_NUMBER.search(txt)),
        "percent": bool(RE_PERCENT.search(txt)),
        "timecode": bool(RE_TIMECODE.search(txt)),
        "date": bool(RE_DATE.search(txt)),
        "unit": bool(RE_UNIT.search(txt)),
        "currency": bool(RE_CURRENCY.search(txt)),
        "url_or_id": bool(RE_URL_ID.search(txt)),
        "all_caps_token": bool(RE_ALLCAPS.search(txt)),
        "keyword_on_screen": bool(RE_HEADERY.search(txt)),
        "short_len": len(txt) <= 160,
        "has_ocr": has_ocr,
    }

def _score_visual_note_generic(txt: str, has_ocr: bool) -> int:
    penalty = -4 if PROMO_VAGUE_TERMS.search(txt) and EVIDENCE_POLICY == "strict" else \
              (-2 if PROMO_VAGUE_TERMS.search(txt) else 0)
    sigs = _detect_signals(txt, has_ocr)
    score = sum(WEIGHT_MAP[k] for k, v in sigs.items() if v)
    return score + penalty

def build_visual_corpus(facts: List[SceneFacts],
                        max_per_scene_chars: int = 220) -> List[str]:
    pool = []
    for sc in sorted(facts, key=lambda x: x.start):
        raw = " ".join(filter(None, [sc.visual_caption, sc.ocr_text])).strip()
        if not raw:
            continue
        vt = re.sub(r"\s+", " ", raw)
        if len(vt) > max_per_scene_chars:
            vt = vt[:max_per_scene_chars]
        has_ocr = bool(sc.ocr_text.strip())
        s = _score_visual_note_generic(vt, has_ocr)
        if EVIDENCE_POLICY == "strict" and s < 3:
            continue
        if EVIDENCE_POLICY == "balanced" and s < 2:
            continue
        pool.append((s, sc.start, sc.end, vt))
    if not pool:
        return []
    pool.sort(key=lambda x: (-x[0], x[1]))
    n_scenes = len(facts)
    dynamic_cap = max(3, min(8, max(1, n_scenes // 8)))
    limit = VISUAL_MAX_EVIDENCE if isinstance(VISUAL_MAX_EVIDENCE, int) else dynamic_cap
    selected = pool[:limit]
    return [f"[{st:.1f}–{en:.1f}s] {vt}" for _, st, en, vt in selected]

# ====== STEP 6: Global Summaries (Improved: Transcript + Visual) ======
PLACEHOLDER_PATTERNS = [
    r"^ประเด็น(แรก|ที่\s*หนึ่ง|ที่\s*สอง|ที่\s*สาม|สี่|ห้า|หก|เจ็ด|แปด)\s*$",
    r"^ข้อ(แรก|ที่\s*\d+)\s*$",
    r"^บูลเล็ต(เสริม)?\s*:?\s*$",
    r"^สรุป(ย่อ)?\s*:?\s*$",
    r"^สกัด.+บูลเล็ต.*$",
]
NUM_PREFIX_RE = re.compile(r"^\s*(?:[-•]+|\(?\d+\)?\.?|[๑-๙]\.|\d+\)|ข้อ\s*\d+\.?)\s*")
PLACEHOLDER_RES = [re.compile(p, flags=re.I) for p in PLACEHOLDER_PATTERNS]

def parse_bullets(text: str) -> List[str]:
    raw = (text or "").splitlines()
    return [l.strip().lstrip("•-–— ").rstrip() for l in raw if l.strip()]

def clean_bullet(s: str) -> str:
    s = NUM_PREFIX_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s).strip(" .;:•-—–")
    return s

def is_placeholder(s: str) -> bool:
    if not s: return True
    for rr in PLACEHOLDER_RES:
        if rr.match(s): return True
    if "บูลเล็ต" in s and "ห้าม" in s: return True
    return False

def filter_bullets(items: List[str]) -> List[str]:
    out = []
    for it in items:
        x = clean_bullet(it)
        if not x: 
            continue
        if is_placeholder(x):
            continue
        out.append(x)
    return out

def is_duplicate(new_item: str, existing: List[str], thr: float = 0.6) -> bool:
    def _tok(s: str) -> List[str]:
        s = re.sub(r"[^\wก-๙%\.:\-/]", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s.split()
    A = set(_tok(new_item))
    for ex in existing:
        B = set(_tok(ex))
        if not A or not B:
            from difflib import SequenceMatcher
            if SequenceMatcher(None, new_item, ex).ratio() >= thr:
                return True
        else:
            inter = len(A & B) / max(1, len(A | B))
            if inter >= thr:
                return True
    return False

def score_informativeness(s: str) -> int:
    sigs = _detect_signals(s, has_ocr=False)
    return sum(WEIGHT_MAP[k] for k,v in sigs.items() if v)

def dedup_and_rerank(bullets: List[str], max_items: int = 8) -> List[str]:
    bullets = filter_bullets(bullets)
    uniq: List[str] = []
    for b in bullets:
        if not is_duplicate(b, uniq, thr=0.62):
            uniq.append(b)
    ranked = sorted(
        enumerate(uniq),
        key=lambda t: (-score_informativeness(t[1]), t[0])
    )
    return [uniq[i] for i,_ in ranked][:max_items]

def summarize_transcript_keypoints(transcript: str, min_items=4, max_items=8) -> List[str]:
    prompt = f"""
ตอบเป็นภาษาไทยล้วนเท่านั้น ห้ามคัดลอกหัวข้อหรือคำอธิบายจากคำสั่ง
ห้ามใส่คำว่า "บูลเล็ต", "หัวข้อ", "สรุป", "บูลเล็ตเสริม" ในผลลัพธ์
ให้ตอบเป็นรายการสั้น ๆ ทีละบรรทัดเท่านั้น (ไม่ต้องมีตัวเลขนำหน้า)

สกัด ประเด็นแก่นจาก TRANSCRIPT จำนวน {min_items}–{max_items} ข้อ
- เน้นข้อเท็จจริง/ใจความหลัก
- กระชับ ชัดเจน ไม่ฟุ่มเฟือย
- ห้ามคัดลอกคำสั่งหรือคำว่า TRANSCRIPT ใด ๆ ลงในคำตอบ

***อย่าเอาข้อความก่อนหน้านี้ไปใส่ในคำตอบ***

=== TRANSCRIPT ===
{transcript}
"""
    out = ollama_summarize(
        prompt,
        options={**GEN_OPTS_FAST, "stop": ["\n\n", "\n• ", "\n- "]}
    ).strip()
    bullets = parse_bullets(out)
    bullets = [ensure_thai(b, 220) for b in bullets]
    return dedup_and_rerank(bullets, max_items=max_items)

def summarize_visual_deltas(visual_notes: List[str],
                            transcript_keys: List[str],
                            max_new: int = 3) -> List[str]:
    if not visual_notes:
        return []
    prompt = f"""
ตอบเป็นภาษาไทยล้วนเท่านั้น
ห้ามใช้คำว่า "บูลเล็ต", "หัวข้อ", "สรุป", หรือขึ้นต้นด้วยหัวข้อประกาศ
ให้ตอบเป็นรายการสั้น ๆ ทีละบรรทัดเท่านั้น (ไม่ต้องมีตัวเลขนำหน้า)

จาก VISUAL NOTES ต่อไปนี้ สร้าง "บรรทัดเสริม" ที่เพิ่มสาระใหม่เท่านั้น
- ห้ามซ้ำ/พาราฟเรสกับ "รายการแก่น" ด้านล่าง
- ใช้เฉพาะสิ่งที่ภาพยืนยันได้จริง (ตัวเลข เปอร์เซ็นต์ ชื่อเฉพาะ หัวข้อบนจอ เวลา/ลิงก์)
- ถ้ามาจากภาพ ให้ลงท้ายวงเล็บ (จากภาพ: …) สั้น ๆ
- ไม่เกิน {max_new} บรรทัด

=== รายการแก่น (ห้ามซ้ำ) ===
{chr(10).join(f"- {k}" for k in transcript_keys)}

=== VISUAL NOTES ===
{chr(10).join(f"- {v}" for v in visual_notes)}
"""
    out = ollama_summarize(
        prompt,
        options={**GEN_OPTS_FAST, "stop": ["\n\n", "\n• ", "\n- "]}
    ).strip()
    candidates = parse_bullets(out)
    candidates = [ensure_thai(clean_bullet(b), 220) for b in candidates]
    filtered = [c for c in candidates if not is_duplicate(c, transcript_keys, thr=0.58)]
    filtered = filter_bullets(filtered)
    return dedup_and_rerank(filtered, max_items=max_new)

def summarize_transcript_global(transcript: str, max_items: int = 8, **kwargs) -> str:
    keys = summarize_transcript_keypoints(transcript, min_items=4, max_items=max_items)
    final = dedup_and_rerank(keys, max_items=max_items)
    return "\n".join(f"• {x}" for x in final)

def summarize_transcript_plus_visual_items(
    transcript: str,
    facts: List[SceneFacts],
    max_items: int = 8,
) -> List[Dict[str, Any]]:
    keys = summarize_transcript_keypoints(transcript, min_items=4, max_items=max_items)
    visual_notes = build_visual_corpus(facts, max_per_scene_chars=220)
    max_new = max(1, math.floor(max_items / 3))
    deltas = summarize_visual_deltas(visual_notes, keys, max_new=max_new)
    combined = [{"text": ensure_thai(k, 220), "source": "transcript"} for k in keys] + \
               [{"text": ensure_thai(d, 220), "source": "visual"} for d in deltas]
    ranked = sorted(
        enumerate(combined),
        key=lambda t: (-score_informativeness(clean_bullet(t[1]["text"])), t[0])
    )
    uniq: List[Dict[str, Any]] = []
    seen: List[str] = []
    for _, item in ranked:
        txt = clean_bullet(item["text"])
        if not txt:
            continue
        if is_duplicate(txt, seen, thr=0.62):
            continue
        seen.append(txt)
        uniq.append({"text": txt, "source": item["source"]})
        if len(uniq) >= max_items:
            break
    for it in uniq:
        if it["source"] == "visual" and not it["text"].rstrip().endswith("(จากภาพ)"):
            it["text"] = it["text"].rstrip() + " (จากภาพ)"
    return uniq

def render_bullets(items: List[Dict[str, Any]]) -> str:
    return "\n".join(f"• {clean_bullet(x['text'])}" for x in items)

# ===== Few-shot สั้น ๆ กันแตกฟอร์แมต =====
FEWSHOT_REF = """
[ตัวอย่างรูปแบบที่ต้องการ]
อินโทรเกริ่นภาพรวมสั้น ๆ ไม่ขายของ ไม่สรุปแบบหัวข้อ แต่วางบริบทว่าประเด็นหลักคืออะไรและทำไมสำคัญ
จากนั้นค่อยอธิบายเหตุผล/หลักฐานที่เชื่อมโยงกันเป็นย่อหน้าเดียวหรือสองย่อหน้า โดยตัดการซ้ำคำหรือ
ซ้ำตัวเลข หากมีข้อมูลจากภาพก็แทรกไว้ในประโยคเป็น (จากภาพ) เพื่อชี้ว่าเป็นหลักฐานเชิงสายตา
สุดท้ายปิดด้วยข้อคิด/ข้อควรระวังที่ปฏิบัติได้ โดยไม่แปลงเป็นรายการย่อย
""".strip()

def self_critique_and_rewrite(draft: str, target_min_words: int, target_max_words: int) -> str:
    check = (
        "ตรวจร่างข้อความต่อไปนี้แบบย่อ: "
        "- มีหัวข้อย่อย/บูลเล็ตไหม (ห้าม)? "
        "- มีประโยคหรือใจความซ้ำไหม? "
        "- จำนวนคำอยู่ช่วงที่กำหนดหรือยัง? "
        "หากพบปัญหา ให้ 'เขียนใหม่ทั้งย่อหน้า' ให้ลื่นไหลกว่าเดิม โดยไม่เพิ่มสาระใหม่ "
        "ถ้าไม่พบปัญหา ให้ส่งร่างเดิมกลับมาเฉย ๆ:\n" + draft
    )
    rewritten = ensure_thai(ollama_summarize(check, options=GEN_OPTS_QUALITY))
    return rewritten if rewritten else draft

# ===== NEW: สร้าง 'บทความสั้น' ภาษาไทย โดยใช้ทั้ง transcript+visual =====
def summarize_article_th(transcript: str,
                         items: List[Dict[str, Any]],
                         target_min_words: int = 300,
                         target_max_words: int = 400) -> str:
    """สร้างบทความภาษาไทยจาก transcript + visual bullets (รอบเดียว, ไม่ polish, ไม่ retry, ไม่ lock)"""
    transcript_src = ensure_thai(transcript) or ""
    if len(transcript_src) > 12000:
        head = transcript_src[:5000]
        mid  = transcript_src[len(transcript_src)//2-1500: len(transcript_src)//2+1500]
        tail = transcript_src[-5000:]
        transcript_src = f"{head}\n...\n{mid}\n...\n{tail}"

    # เลือกเฉพาะ visual items
    vis_points = []
    for x in items:
        if x.get("source") == "visual":
            txt = clean_bullet(x["text"])
            if txt and "(จากภาพ)" not in txt:
                txt += " (จากภาพ)"
            vis_points.append(txt)
    vis_points = dedup_and_rerank(vis_points, max_items=4)

    # ---- prompt หลัก (เน้นไม่ให้แก้ชื่อ แต่ไม่ใช้ lock) ----
    prompt = f"""
เขียน "บทความภาษาไทยแบบเล่าเรื่อง" ความยาวประมาณ {target_min_words}-{target_max_words} คำ 
โดยอ้างอิง TRANSCRIPT ด้านล่างเป็นหลัก และผสาน "หลักฐานจากภาพ" เท่าที่จำเป็น
**ข้อกำหนดการเขียน**
- อินโทรเกริ่นภาพรวมสั้น ๆ เพื่อให้เข้าใจว่าประเด็นหลักคืออะไรและทำไมสำคัญ  
- จากนั้นอธิบายเนื้อหาหลักแบบต่อเนื่อง 2–4 ย่อหน้า (ห้ามใช้หัวข้อย่อย/บูลเล็ต/เลขลิสต์)  
- ห้ามพูดซ้ำใจความเดิมหรือตัวเลขเดิมเกิน 1 ครั้ง  
- ใช้คำเชื่อมให้ลื่นไหล อ่านเข้าใจตั้งแต่ต้นจนจบ  
- ถ้ามีข้อมูลจากภาพ ให้แทรกไว้ในประโยคเป็น (จากภาพ) เพื่อชี้ว่าเป็นหลักฐานเชิงสายตา  
- ห้ามขึ้นต้นด้วยคำว่า "สวัสดี", "วันนี้", "บทความนี้", "เราจะมาดู"  
- หลีกเลี่ยงการเปลี่ยนชื่อเฉพาะ/ตัวย่อ/ชื่อบุคคล/ตัวเลขใน TRANSCRIPT  
- ปิดท้ายด้วยข้อคิดหรือข้อสังเกตเชิงเหตุผลสั้น ๆ ที่สอดคล้องกับเนื้อหา  

[TRANSCRIPT]
{transcript_src}

[หลักฐานจากภาพ]
{chr(10).join(f"- {p}" for p in vis_points)}
"""

    # ---- เรียก LLM "ครั้งเดียว" ----
    # ถ้า GEN_OPTS_QUALITY มีอยู่แล้วจะใช้เลย; เสริม temp ต่ำเพื่อลดการ "แก้ชื่อ" โดยพลการ
    GEN_OPTS = {
        **GEN_OPTS_QUALITY,
        "num_predict": 900,   # ~900 token พอสำหรับ 300-400 คำไทย
    }
    raw = ensure_thai(ollama_summarize(prompt, options=GEN_OPTS)) or ""
    
    return raw

def extract_single_keyword_th(text: str) -> str:
    """
    ใช้ LLM สกัด 'คำสำคัญหลัก' เพียงคำเดียว (ภาษาไทย) จากข้อความที่ให้มา
    """
    prompt = f"""
อ่านบทความต่อไปนี้ แล้วตอบเพียงคำเดียวที่เป็น "คำสำคัญหลัก" เท่านั้น
- ห้ามพูดเกินหนึ่งคำ
- ห้ามเติมคำอธิบาย
- ตัวอย่าง: ถ้าบทความพูดถึงการเมือง ให้ตอบว่า การเมือง
- ให้ตอบเฉพาะคำเดียวเช่นนั้น

[บทความ]
{text}
"""
    out = ollama_summarize(prompt, options={"temperature": 0.2, "num_ctx": 1024})
    # ตัดบรรทัด/เว้นวรรคให้เหลือแค่คำเดียว
    keyword = out.strip().split()[0]
    keyword = re.sub(r"[^\wก-๙]", "", keyword)
    return keyword or "ไม่พบคำสำคัญ"

def _safe_word_count(path: str):
    try:
        from pythainlp.tokenize import word_tokenize
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            tokens = word_tokenize(text, engine="newmm")
            return len(tokens)
    except Exception:
        return None

# ====== (อัปเดต) MAIN: ผลิตเฉพาะ transcript+visual ======
def main():
    t0 = time.time()
    for c in ["ffmpeg", "ffprobe"]:
        check_cmd(c)
        
    pbar = tqdm(total=10, desc="🚀 Overall Progress", unit="step")

    # 1) (เลือก) โหลดเสียง + ตัดฉาก
    pbar.set_description("🎧 Step 1: โหลดคลิปและดึง frames")
    download_audio_wav_16k(YOUTUBE_URL, AUDIO_OUT)
    duration = get_video_duration(AUDIO_OUT)
    download_t = time.time()
    download_time = download_t - t0
    pbar.update(1)
    send_progress("โหลดวิดีโอเสร็จสิ้น", 10)

    # 2) Transcript (บังคับไทย)
    pbar.set_description("🗣️ Step 2: Speech to text")
    transcript = transcribe_whisper(AUDIO_OUT, WHISPER_MODEL, LANGUAGE, ASR_DEVICE)
    # transcript = iapp_asr_api(AUDIO_OUT, "audio.wav")
    asr_t = time.time()
    asr_time = asr_t - download_t
    pbar.update(2)
    send_progress("ถอดเสียงเสร็จสิ้น", 60)
    # with open(TRANSCRIPT_TXT, "r", encoding="utf-8") as f:
    #     transcript = f.read()

    # 3) Caption + OCR
    pbar.set_description("🖼️ Step 3: Image captioning")
    captioner = VisionCaptioner(VL_MODEL_NAME, VL_DEVICE)
    stream_scene_frames_and_caption(YOUTUBE_URL, FRAMES_DIR, SCENE_THRESH, CAPTIONS_JSON, captioner)
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        caps = json.load(f)
    scene_ts = [c["ts"] for c in caps if "ts" in c]
    # with open(SCENES_JSON, "w", encoding="utf-8") as f:
    #     json.dump(scene_ts, f, ensure_ascii=False, indent=2)

    facts = split_text_to_scenes(transcript, scene_ts)
    facts = enrich_scenes_with_captions(facts, caps)
    frames_count = len(caps)
    with open(SCENE_FACTS_JSON, "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in facts], f, ensure_ascii=False, indent=2)
    log(f"✅ Scene facts saved -> {SCENE_FACTS_JSON}")
    cap_t = time.time()
    cap_time = cap_t - asr_t
    pbar.update(4)
    send_progress("สร้างคำบรรยายภาพเสร็จสิ้น", 80)

    # 4) สรุปเฉพาะ "Transcript + Visual"
    pbar.set_description("🧠 Step 4: : ทำสรุป")
    items = summarize_transcript_plus_visual_items(transcript, facts, max_items=8)
    pbar.update(2)
    send_progress("ทำสรุปเสร็จสิ้น", 90)

    # 5) Save dropdown + bullets + article
    pbar.set_description("💾 Step 5: สร้างสรุปแบบบทความ + บันทึกผลลัพธ์")
    with open(DROPDOWN_JSON, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    bullets_txt = render_bullets(items)
    with open(FINAL_TXT, "w", encoding="utf-8") as f:
        f.write(ensure_thai(bullets_txt, max_chars=900))

    article_th = summarize_article_th(transcript, items, target_min_words=300, target_max_words=400)
    with open(FINAL_ARTICLE_TXT, "w", encoding="utf-8") as f:
        f.write(wrap_text(article_th))
        
    main_keyword = extract_single_keyword_th(article_th)
    
    summarize_t = time.time()
    summarize_time = summarize_t - cap_t

    log("\n===== DROPDOWN ITEMS (preview) =====")
    for it in items:
        tag = "VIS" if it["source"] == "visual" else "TR"
        log(f"- [{tag}] {it['text']}")

    log("\n===== SUMMARY (TRANSCRIPT + VISUAL) =====")
    log(bullets_txt or "(empty)")

    log("\n===== SHORT ARTICLE (TH) =====")
    log(article_th or "(empty)")

    t1 = time.time()
    log("\n✅ Done.")
    log(f"\n⏱️ Elapsed: {t1 - t0:.2f} sec")
    
    log(f"""TIME BREAKDOWN:
          - Download & Audio Extract: {download_time:.2f} sec
          - ASR (Whisper): {asr_time:.2f} sec
          - Captioning + OCR: {cap_time:.2f} sec
          - Summarization: {summarize_time:.2f} sec""")
    
        # === LOG TO EXCEL ===
    try:
        scenes_count = len(scene_ts) if isinstance(scene_ts, list) else None
        captions_count = len(caps) if isinstance(caps, list) else None
        bullets_count = len(items) if isinstance(items, list) else None
        article_words = _safe_word_count(FINAL_ARTICLE_TXT)
        transcript_words = _safe_word_count(TRANSCRIPT_TXT)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            # ---- CONFIG / PARAMS ----
            "youtube_url": YOUTUBE_URL,
            "whisper_model": WHISPER_MODEL,
            "asr_device": ASR_DEVICE,
            "vl_device": VL_DEVICE,
            "vl_model": VL_MODEL_NAME,
            "scene_thresh": SCENE_THRESH,
            "enable_ocr": ENABLE_OCR,
            # ---- COUNTS / SIZES ----
            "frames": frames_count,
            "scenes": scenes_count,
            "captions": captions_count,
            "bullets": bullets_count,
            "transcript_words": transcript_words,
            "article_words": article_words,
            "keyword": main_keyword,
            # ---- OUTPUT FILES ----
            # "audio_out": AUDIO_OUT,
            # "scenes_json": SCENES_JSON,
            # "captions_json": CAPTIONS_JSON,
            # "scene_facts_json": SCENE_FACTS_JSON,
            # "dropdown_json": DROPDOWN_JSON,
            # "final_bullets_txt": FINAL_TXT,
            # "final_article_txt": FINAL_ARTICLE_TXT,
            # ---- TIMING (sec) ----
            "t_download": round(download_time, 2),
            "t_asr": round(asr_time, 2),
            "t_caption": round(cap_time, 2),
            "t_summarize": round(summarize_time, 2),
            "t_total": None,  # จะเติมด้านล่าง
            "duration_sec": round(duration, 2) if isinstance(duration, (int, float)) else None,
        }
        
        log(" keyword: " + main_keyword)

        # เติมเวลารวม (หากมีตัวแปร t0/t1 อยู่แล้ว)
        try:
            row["t_total"] = round(time.time() - t0, 2)
        except Exception:
            pass

        try:
            if METRICS_JSON:
                os.makedirs(os.path.dirname(METRICS_JSON), exist_ok=True)
                import json as _json
                with open(METRICS_JSON, "w", encoding="utf-8") as f:
                    _json.dump(row, f, ensure_ascii=False)
                log(f"📝 Metrics saved -> {METRICS_JSON}")
            else:
                # ถ้าไม่กำหนด METRICS_JSON ไว้ ก็ข้ามเฉยๆ
                pass
        except Exception as e:
            log(f"⚠️ Metrics JSON write failed: {e}")
    except Exception as e:
        log(f"⚠️ Statistic logging failed: {e}")
    pbar.update(1)
    pbar.close()
    send_progress("กำลังอัปเดต DB", 99)


if __name__ == "__main__":
    main()
