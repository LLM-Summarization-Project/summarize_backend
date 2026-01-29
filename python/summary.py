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
from pythainlp.tokenize import word_tokenize

import requests
import torch
from PIL import Image
import whisper  # openai-whisper
import textwrap

from datetime import datetime

_progress_fp = None
try:
    _progress_fp = os.fdopen(3, "w", buffering=1, encoding="utf-8")  # line-buffered
except Exception:
    _progress_fp = None

def send_progress(step: str, percent: int, subprogress: int):
    if _progress_fp:
        _progress_fp.write(json.dumps({"type":"progress","step":step,"percent":percent,"subprogress": subprogress}) + "\n")
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
TRANSCRIPT_SEGMENTS = "transcript_segments.json"
METRICS_JSON = globals().get("METRICS_JSON", None)
log = functools.partial(print, file=sys.stderr, flush=True)

LANGUAGE = "th"
WHISPER_MODEL = "large-v3-turbo"
WHISPER_TEMP = float(os.environ.get("WHISPER_TEMP", "0.0"))  # Whisper temperature (0.0 = deterministic)
ASR_DEVICE = "cpu"      
VL_DEVICE  = "cuda"     # ใช้กับ Florence เท่านั้น

VL_MODEL_NAME = "microsoft/Florence-2-base"
SCENE_THRESH = 0.6
ENABLE_OCR = False
USE_YOUTUBE_TRANSCRIPT = True  # ใช้ youtube_transcript_api เป็นทางเลือกแรก (เร็วกว่า Whisper มาก)

# ใช้ 127.0.0.1 กันปัญหา IPv6/localhost บางเครื่อง
OLLAMA_API_CHAT = os.environ.get("OLLAMA_API_CHAT", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")  

# ===== NEW OUTPUT NAMES =====
DROPDOWN_JSON = "dropdown_items.json"           # สำหรับ UI dropdown
FINAL_TXT = "dropdown_list.txt"  # bullet รวม
FINAL_ARTICLE_TXT = "final_article_th.txt"      # บทความสั้นภาษาไทย

# ===== STRONG THAI-ONLY SYSTEM (อัปเดตให้เข้มแบบร้อยแก้วไทย) =====
SYSTEM_PROMPT_TH = "ตอบเป็นภาษาไทยเท่านั้น"

# ===== NEW: Generation presets (minimal-safe) =====
GEN_OPTS_QUALITY = {
    "temperature": 0.3,        # ใกล้ default ของ Ollama
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

def restart_ollama():
    """Kill และ restart Ollama process (Windows)"""
    import platform
    log("🔄 กำลัง restart Ollama...")
    try:
        if platform.system() == "Windows":
            # Kill ollama.exe
            subprocess.run(["taskkill", "/IM", "ollama.exe", "/F"], 
                          capture_output=True, timeout=10)
            time.sleep(2)
            # Start ollama serve in background
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
        else:
            # Linux/Mac
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True, timeout=10)
            time.sleep(2)
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        
        # รอให้ Ollama พร้อม
        for _ in range(30):  # รอสูงสุด 30 วินาที
            time.sleep(1)
            if ollama_healthcheck(OLLAMA_API_CHAT):
                log("✅ Ollama restart สำเร็จ")
                return True
        log("⚠️ Ollama restart แต่ยังไม่พร้อม")
        return False
    except Exception as e:
        log(f"⚠️ ไม่สามารถ restart Ollama ได้: {e}")
        return False

def ollama_summarize(
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    system: Optional[str] = None,
    stream: bool = False,           # ปิดสตรีมเพื่อความเสถียรและจับข้อความครบ
    max_wait: int = 600,           # รอสูงสุด 10 นาที (สำหรับงานยาว)
    heartbeat_interval: int = 30,   # เช็ค heartbeat ทุก 30 วินาที
    max_retries: int = 3,           # จำนวน retry สูงสุด
) -> str:
    """
    เรียก Ollama พร้อม heartbeat monitoring
    - ถ้า Ollama ยังตอบ healthcheck = กำลังทำงาน รอต่อ
    - ถ้า healthcheck ไม่ตอบ = ค้าง ให้ restart
    """
    import threading
    import queue
    
    base = OLLAMA_API_CHAT
    if system is None:
        system = SYSTEM_PROMPT_TH

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }

    if options:
        ALLOWED_OPTS = {
            "temperature", "top_p", "top_k", "repeat_penalty", "repeat_last_n",
            "num_ctx", "num_predict", "stop", "seed"
        }
        filtered_opts = {k: v for k, v in options.items() if k in ALLOWED_OPTS}
        if filtered_opts:
            payload["options"] = filtered_opts

    def do_request(result_queue):
        """Thread function ที่ส่ง request จริง"""
        try:
            resp = requests.post(base, json=payload, timeout=max_wait)
            resp.raise_for_status()
            data = resp.json()
            msg = (data.get("message") or {}).get("content", "")
            result_queue.put(("success", (msg or "").strip()))
        except Exception as e:
            result_queue.put(("error", e))

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # เช็ค healthcheck ก่อนเริ่ม
            if not ollama_healthcheck(base):
                log(f"⚠️ Ollama ไม่ตอบ (attempt {attempt}/{max_retries}), กำลัง restart...")
                if not restart_ollama():
                    continue
            
            ollama_ensure_model(OLLAMA_MODEL, base)
            
            # ส่ง request ใน background thread
            result_queue = queue.Queue()
            request_thread = threading.Thread(target=do_request, args=(result_queue,))
            request_thread.daemon = True
            request_thread.start()
            
            # รอผลลัพธ์พร้อมเช็ค heartbeat
            elapsed = 0
            while elapsed < max_wait:
                try:
                    # รอผลลัพธ์ทุก heartbeat_interval วินาที
                    status, result = result_queue.get(timeout=heartbeat_interval)
                    if status == "success":
                        return result
                    else:
                        raise result  # Re-raise exception
                except queue.Empty:
                    # ยังไม่ได้ผล - เช็ค heartbeat
                    elapsed += heartbeat_interval
                    if ollama_healthcheck(base):
                        log(f"💓 Ollama ยังทำงานอยู่... (รอแล้ว {elapsed}s)")
                    else:
                        log(f"💔 Ollama ไม่ตอบ! กำลัง restart... (รอแล้ว {elapsed}s)")
                        restart_ollama()
                        break  # ออกจาก while loop เพื่อ retry
            else:
                # รอจน max_wait โดยไม่ได้ผล
                last_error = f"รอเกิน {max_wait}s"
                log(f"⚠️ Ollama ไม่ตอบภายใน {max_wait}s")
                restart_ollama()
                
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
            log(f"⚠️ Ollama connection error (attempt {attempt}/{max_retries})")
            restart_ollama()
            
        except Exception as e:
            last_error = str(e)
            log(f"⚠️ Ollama error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                restart_ollama()
    
    raise RuntimeError(f"❌ Ollama ล้มเหลวหลังจาก {max_retries} ครั้ง: {last_error}")

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

# get duration in sec for DB
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

# โหลดเสียงจาก youtube มาเป็น .m4a แล้วแปลงเป็น .wav เพื่อให้ whisper ใช้ได้(ตรงนี้โหลดแบบ bestaudio)
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

# โหลดวิดีโอจาก youtube มาเป็น .mp4 เพื่อใช้กับ scene-cut(ตรงนี้โหลดแบบ bv*+ba/b)
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
def transcribe_whisper(
    wav_path: str,
    model_name: str,
    language: str,
    device: str,
    step_start: int = 10,
    step_end: int = 45,
) -> tuple[str, List[Dict[str, Any]]]:
    """
    ถอดเสียงด้วย whisper และ return segments พร้อม timestamps
    Returns:
        tuple: (full_text, segments)
        - full_text: ข้อความทั้งหมด
        - segments: list of {start, end, text} สำหรับจับคู่กับ visual
    """

    log("🔄 Loading Whisper model...")
    model = whisper.load_model(model_name, device=device)
    
    if _progress_fp:
        _progress_fp.write(json.dumps({"type":"model_loaded"}) + "\n")
        _progress_fp.flush()
    
    send_progress("ถอดเสียง", step_start, 0)
    log("✅ Model loaded, starting transcription...")

    result = model.transcribe(
        wav_path,
        language=language,
        fp16=(device == "cuda"),
        temperature=WHISPER_TEMP,
        condition_on_previous_text=True,
        initial_prompt=None,
        compression_ratio_threshold=None,
        verbose=False,
    )

    # ดึง segments พร้อม timestamps
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": (seg.get("text") or "").strip()
        })

    text = (result["text"] or "").strip()
    text = ensure_thai(text)

    with open(TRANSCRIPT_TXT, "w", encoding="utf-8") as f:
        f.write(text)

    # บันทึก segments แยกด้วย
    with open(TRANSCRIPT_SEGMENTS, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    log(f"✅ Saved {len(segments)} transcript segments")

    send_progress("ถอดเสียง", step_end, 100)
    log("✅ Transcription done.")

    return text, segments


# ====== STEP 2B: YouTube Transcript API (ทางเลือกที่ 2 - เร็วกว่า Whisper มาก) ======
def extract_video_id(url: str) -> str:
    """ดึง video ID จาก YouTube URL"""
    import re
    patterns = [
        r'(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""

def transcribe_youtube_api(
    youtube_url: str,
    language: str = "th",
    step_start: int = 10,
    step_end: int = 45,
) -> tuple[str, List[Dict[str, Any]]]:
    """
    ดึง transcript จาก YouTube โดยใช้ youtube_transcript_api
    เร็วกว่า Whisper มากเพราะไม่ต้องโหลด audio และ process
    
    Returns:
        tuple: (full_text, segments)
        - full_text: ข้อความทั้งหมด
        - segments: list of {start, end, text} สำหรับจับคู่กับ visual
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        raise RuntimeError("❌ youtube_transcript_api ไม่ได้ติดตั้ง: pip install youtube-transcript-api")
    
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError(f"❌ ไม่สามารถดึง video ID จาก URL: {youtube_url}")
    
    log(f"🔄 Fetching YouTube transcript for video: {video_id}")
    send_progress("ถอดเสียง", step_start, 0)
    
    try:
        # ใช้ API format ใหม่ (version 1.x)
        ytt_api = YouTubeTranscriptApi()
        
        # ลองดึง transcript - จะพยายามหา transcript ที่เหมาะสมอัตโนมัติ
        # ลองภาษาไทยก่อน แล้ว fallback เป็นภาษาอื่น
        transcript_data = None
        transcript_type = ""
        
        try:
            # ลองดึงภาษาไทยก่อน
            transcript_data = ytt_api.fetch(video_id, languages=[language, 'th'])
            transcript_type = f"thai ({language})"
            log(f"✅ พบ transcript ภาษาไทย")
        except Exception as e:
            log(f"⚠️ ไม่พบ transcript ภาษาไทย: {e}")
            
            # ลองดึงภาษาใดก็ได้
            try:
                transcript_data = ytt_api.fetch(video_id)
                transcript_type = "auto"
                log(f"✅ พบ transcript (auto)")
            except Exception as e2:
                raise RuntimeError(f"ไม่พบ transcript: {e2}")
        
        send_progress("ถอดเสียง", (step_start + step_end) // 2, 50)
        
        # แปลงเป็น format เดียวกับ Whisper
        # เก็บข้อมูลดิบก่อน แล้วค่อยกำหนด end = start ของตัวถัดไป
        raw_items = []
        
        # transcript_data เป็น FetchedTranscript object, iterate ได้เลย
        for item in transcript_data:
            start = item.start
            text = (item.text or '').strip()
            if text:
                raw_items.append({"start": start, "text": text})
        
        # สร้าง segments โดยใช้ start ของตัวถัดไปเป็น end
        segments = []
        full_text_parts = []
        for i, item in enumerate(raw_items):
            if i < len(raw_items) - 1:
                end_time = raw_items[i + 1]["start"]
            else:
                # segment สุดท้าย - ใช้ start + duration โดยประมาณ (5 วินาที)
                end_time = item["start"] + 5.0
            
            segments.append({
                "start": item["start"],
                "end": end_time,
                "text": item["text"]
            })
            full_text_parts.append(item["text"])
        
        full_text = " ".join(full_text_parts)
        
        # ถ้า transcript ไม่ใช่ภาษาไทย ให้แปล
        if not looks_thai(full_text) and len(full_text) > 50:
            log("🔄 แปล transcript เป็นภาษาไทย...")
            full_text = ensure_thai(full_text)
        
        # บันทึกไฟล์
        with open(TRANSCRIPT_TXT, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        with open(TRANSCRIPT_SEGMENTS, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        
        send_progress("ถอดเสียง", step_end, 100)
        log(f"✅ YouTube Transcript done: {len(segments)} segments ({transcript_type})")
        
        return full_text, segments
        
    except Exception as e:
        raise RuntimeError(f"❌ YouTube Transcript API error: {e}")


# ====== STEP 3: Image Captioning (+ optional OCR) ======
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
# download scene frames and caption them
def stream_scene_frames_and_caption(url: str,
                                    frames_dir: str,
                                    thresh: float,
                                    out_json: str,
                                    captioner: VisionCaptioner,
                                    video_duration: float | None = None,):
    os.makedirs(frames_dir, exist_ok=True)
    delete_all_files_in_directory(frames_dir)

    # ดาวน์โหลดวิดีโอเก็บเป็นไฟล์ชั่วคราว
    video_path = download_video_file(url)
    duration = get_video_duration(video_path)

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
    processed = 0
    estimated = 50
    if not isinstance(video_duration, (int, float)) or video_duration <= 0:
        video_duration = None
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
                    processed += 1
                    if video_duration is not None and ts is not None:
                        ratio = max(0.0, min(ts / video_duration, 1.0))
                        subprogress = int(ratio * 100)
                    else:
                        subprogress = min(100, int(processed / estimated * 100))
                    percent = 45 + int((35 * subprogress) / 100)

                    send_progress("สร้างคำบรรยายภาพ", percent, subprogress)  # 45–80%

                    # ลบภาพเฟรมทันที ลด IO/พื้นที่
                    try: os.remove(img_path)
                    except: pass

                    next_id += 1

        # เขียนผล caption ทั้งหมด
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log(f"✅ Captions saved -> {out_json}")

    except Exception as e:
        log(f"❌ Error in scene captioning: {e}")
        # ถ้ามี error ก็ให้เขียน empty captions เพื่อไม่ให้ pipeline พัง
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log(f"⚠️ Saved partial/empty captions -> {out_json}")

    finally:
        # รอ ffmpeg ปิดให้เรียบร้อยก่อนค่อยลบวิดีโอชั่วคราว (กันไฟล์ล็อก)
        try:
            if proc is not None:
                proc.wait(timeout=5)
        except Exception:
            pass
        _safe_unlink(video_path)
    
    return duration

# ====== STEP 4: Merge into scene-level facts ======
@dataclass
class TranscriptSegment:
    """Segment จาก Whisper พร้อม timestamp"""
    start: float
    end: float
    text: str

@dataclass
class SceneFacts:
    start: float
    end: float
    speech: str
    visual_caption: str
    ocr_text: str
    tags: List[str]
# จับคู่ transcript segments กับ scene timestamps
def split_segments_to_scenes(
    segments: List[Dict[str, Any]],
    scene_ts: List[float]
) -> List[SceneFacts]:
    """
    จับคู่ transcript segments กับ scene timestamps โดยใช้ timestamp จริง
    แต่ละ scene จะมี speech ที่อยู่ในช่วงเวลานั้นจริงๆ
    
    ถ้าไม่มี scene_ts (ffmpeg ตรวจไม่เจอ scene cuts):
    - ใช้ transcript segments สร้าง scene boundaries แทน
    - แบ่งเป็นกลุ่มละประมาณ 15-30 วินาที
    """
    if not segments:
        return [SceneFacts(0.0, 99999.0, "", "", "", [])]
    
    # ถ้าไม่มี scene_ts -> ใช้ transcript segments สร้าง scene boundaries
    if not scene_ts:
        log("⚠️ No scene cuts detected, using transcript segments as boundaries")
        
        # กลุ่ม segments เป็นช่วงละ ~15-30 วินาที
        SCENE_DURATION = 20.0  # วินาที
        facts: List[SceneFacts] = []
        
        current_start = segments[0].get("start", 0)
        current_texts = []
        
        for seg in segments:
            seg_end = seg.get("end", 0)
            seg_text = seg.get("text", "").strip()
            
            # ถ้ายังอยู่ในช่วงเวลาปัจจุบัน ให้รวมข้อความ
            if seg_end - current_start < SCENE_DURATION:
                current_texts.append(seg_text)
            else:
                # จบช่วงปัจจุบัน สร้าง SceneFacts
                if current_texts:
                    facts.append(SceneFacts(
                        start=current_start,
                        end=seg.get("start", current_start + SCENE_DURATION),
                        speech=" ".join(current_texts),
                        visual_caption="",
                        ocr_text="",
                        tags=[]
                    ))
                # เริ่มช่วงใหม่
                current_start = seg.get("start", 0)
                current_texts = [seg_text]
        
        # เพิ่ม scene สุดท้าย
        if current_texts:
            last_end = segments[-1].get("end", current_start + SCENE_DURATION)
            facts.append(SceneFacts(
                start=current_start,
                end=last_end,
                speech=" ".join(current_texts),
                visual_caption="",
                ocr_text="",
                tags=[]
            ))
        
        log(f"✅ Created {len(facts)} scenes from transcript segments")
        return facts if facts else [SceneFacts(0.0, 99999.0, " ".join(s.get("text", "") for s in segments), "", "", [])]
    
    scene_ts = sorted(scene_ts)
    # สร้าง bounds: [(start1, end1), (start2, end2), ...]
    bounds = []
    for i, ts in enumerate(scene_ts):
        if i < len(scene_ts) - 1:
            bounds.append((ts, scene_ts[i+1]))
        else:
            bounds.append((ts, ts + 99999))
    
    facts: List[SceneFacts] = []
    for start, end in bounds:
        # หา segments ที่อยู่ในช่วงเวลานี้
        matching_segs = []
        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            seg_mid = (seg_start + seg_end) / 2
            # ใช้จุดกึ่งกลางของ segment ตัดสินว่าอยู่ใน scene ไหน
            if start <= seg_mid < end:
                matching_segs.append(seg.get("text", "").strip())
        
        speech = " ".join(matching_segs)
        facts.append(SceneFacts(
            start=start,
            end=end,
            speech=speech,
            visual_caption="",
            ocr_text="",
            tags=[]
        ))
    
    return facts
# ตรวจว่า visual caption เกี่ยวข้องกับ speech หรือไม่/มีข้อมูลที่มีประโยชน์
def check_visual_relevance(speech: str, visual_caption: str) -> bool:
    """
    ตรวจว่า visual caption เกี่ยวข้องกับ speech หรือไม่
    คืนค่า True ถ้าควร merge, False ถ้าไม่ควร
    
    เกณฑ์การตัดสิน:
    1. มีคำซ้อนทับกันอย่างน้อย 1 คำสำคัญ (ไม่นับ stopwords)
    2. หรือ visual มีข้อมูลที่มีประโยชน์ (ตัวเลข, %, เวลา, ชื่อเฉพาะ)
    """
    if not speech or not visual_caption:
        return False
    
    # Stopwords ภาษาไทย/อังกฤษ ที่ไม่ควรนับ
    STOPWORDS = {
        "ที่", "ใน", "ของ", "และ", "เป็น", "มี", "ได้", "ให้", "กับ", "จาก", "ไป", "มา", "อยู่", "แล้ว",
        "นี้", "นั้น", "ก็", "จะ", "ว่า", "ไม่", "เรา", "เขา", "คุณ", "ผม", "ฉัน", "ครับ", "ค่ะ",
        "the", "a", "an", "is", "are", "was", "were", "on", "in", "at", "to", "for", "of", "with"
    }
    
    # Tokenize ทั้งสอง (แบบง่าย)
    def tokenize(text: str) -> set:
        from pythainlp.tokenize import word_tokenize 
        tokens = word_tokenize(text, engine="newmm")
        return {t.lower() for t in tokens if t not in STOPWORDS and len(t) > 1}
    
    speech_tokens = tokenize(speech)
    visual_tokens = tokenize(visual_caption)
    
    # ตรวจคำซ้อนทับ
    overlap = speech_tokens & visual_tokens
    if len(overlap) >= 2:  # มีคำตรงกันอย่างน้อย 2 คำ
        return True
    
    # ตรวจว่า visual มีข้อมูลที่มีประโยชน์ (ตัวเลข, %, เวลา)
    has_useful_data = bool(
        RE_NUMBER.search(visual_caption) or
        RE_PERCENT.search(visual_caption) or
        RE_TIMECODE.search(visual_caption) or
        RE_DATE.search(visual_caption) or
        RE_CURRENCY.search(visual_caption)
    )
    
    if has_useful_data and len(overlap) >= 1:
        return True
    
    # ไม่เกี่ยวข้อง
    return False

def enrich_scenes_with_captions(facts: List[SceneFacts], captions: List[Dict[str,Any]]) -> List[SceneFacts]:
    """
    จับคู่และ MERGE visual captions เข้ากับ scenes ตาม timestamp
    - Visual caption จะถูกรวมเฉพาะเมื่ออยู่ในช่วงเวลาเดียวกัน
    - ตรวจ RELEVANCE ก่อน: caption ต้องเกี่ยวข้องกับ speech ด้วย
    - ช่วยเพิ่มรายละเอียดที่ transcript อาจไม่มี (ตัวเลข, ชื่อเฉพาะ)
    """
    for sc in facts:
        # หา captions ที่อยู่ในช่วงเวลาเดียวกัน (±2 วินาที tolerance)
        matched_caps = [
            c for c in captions 
            if c.get("ts") is not None and sc.start - 2.0 <= c["ts"] <= sc.end + 2.0
        ]
        
        # ถ้าไม่มี exact match ก็หา closest 1 อัน (ถ้าห่างไม่เกิน 10 วินาที)
        if not matched_caps and captions:
            sorted_caps = sorted(captions, key=lambda c: abs(c.get("ts", 0) - sc.start))
            closest = sorted_caps[0]
            if abs(closest.get("ts", 0) - sc.start) <= 10.0:
                matched_caps = [closest]
        
        vc, ocrs, tags = [], [], []
        for c in matched_caps:
            cap_text = c.get("caption_detailed") or c.get("caption_short") or ""
            
            # ✅ RELEVANCE CHECK: ตรวจว่าเกี่ยวข้องก่อน merge
            if check_visual_relevance(sc.speech, cap_text):
                vc.append(cap_text)
                if c.get("ocr_text"): 
                    ocrs.append(c["ocr_text"])
                tags.extend(c.get("tags", []))
            else:
                log(f"⚠️ Skipped irrelevant visual: '{cap_text[:50]}...' for speech: '{sc.speech[:50]}...'")
        
        sc.visual_caption = ensure_thai(" ".join(vc).strip()) if vc else ""
        sc.ocr_text = ensure_thai(" ".join(ocrs).strip()) if ocrs else ""
        sc.tags = sorted(list(set(tags)))
    
    return facts

# ====== STEP 5: Visual Evidence (domain-agnostic) ======
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

# ====== STEP 6: Global Summaries (Improved: Transcript + Visual) ======
def summarize_article_th(facts: List[SceneFacts],
                         target_min_words: int = None,
                         target_max_words: int = None) -> str:
    """
    สร้างบทความภาษาไทยจาก SceneFacts ที่รวม speech + visual_caption ไว้ด้วยกัน
    ทำให้ LLM เห็น context ว่าภาพอยู่ตรงไหนของเนื้อหา
    """
    
    # 1) สร้าง combined context จาก SceneFacts
    combined_segments = []
    total_speech = ""
    for sc in sorted(facts, key=lambda x: x.start):
        segment = sc.speech
        if sc.visual_caption:
            segment += f" (ภาพ: {sc.visual_caption})"
        if sc.ocr_text:
            segment += f" (ข้อความบนจอ: {sc.ocr_text})"
        combined_segments.append(segment)
        total_speech += sc.speech + " "
    
    combined_context = "\n".join(combined_segments)
    
    # 2) คำนวณความยาว transcript (ประมาณ)
    transcript_word_count = word_count_th(total_speech)
    
    # Dynamic target: ปรับความยาวตาม transcript (กำหนด target เสมอเพื่อป้องกันสรุปสั้นเกินไป)
    if target_min_words is None or target_max_words is None:
        if transcript_word_count < 300:
            target_min_words = 100
            target_max_words = 150
            log(f"📝 Transcript: ~{transcript_word_count} words → Target summary: 100-150 words")
        elif transcript_word_count < 800:
            target_min_words = 150
            target_max_words = 250
            log(f"📝 Transcript: ~{transcript_word_count} words → Target summary: 150-250 words")
        else:
            target_min_words = 300
            target_max_words = 400
            log(f"📝 Transcript: ~{transcript_word_count} words → Target summary: 300-400 words")
    
    # 3) ถ้าเนื้อหายาวเกินไป ใช้ระบบ "แบ่งสรุปทีละช่วง (Chunking)" เพื่อเก็บรายละเอียดให้ครบ
    # ช่วยแก้ปัญหา: 1. ข้อมูลตกหล่น (เพราะไม่ต้องตัดทิ้ง) 2. ไม่ค้าง (เพราะส่งทีละน้อย)
    if len(combined_context) > 8000:
        log(f"📝 เนื้อหายาว ({len(combined_context)} chars) -> ใช้ระบบแบ่งสรุป (Chunking) {len(combined_context)//6000 + 1} ส่วน")
        
        chunk_size = 6000
        chunks = [combined_context[i:i+chunk_size] for i in range(0, len(combined_context), chunk_size)]
        
        partial_summaries = []
        for i, chunk in enumerate(chunks):
            p = f"สรุปเนื้อหาส่วนที่ {i+1} จาก {len(chunks)} นี้ให้กระชับ เก็บประเด็นสำคัญให้ครบ เพื่อนำไปรวมกับส่วนอื่น:\n{chunk}"
            # ใช้ Fast mode เพื่อความรวดเร็ว
            sub_s = ensure_thai(ollama_summarize(p, options=GEN_OPTS_FAST))
            partial_summaries.append(sub_s)
        
        # รวมผลสรุปย่อยเป็น Context ใหม่สำหรับสรุปสุดท้าย
        combined_context = "\n---\n".join(partial_summaries)
        log(f"✅ รวมสรุปย่อยเสร็จสิ้น ({len(combined_context)} chars)")

    # 4) สร้าง prompt
    length_instruction = f"ความยาวประมาณ {target_min_words}-{target_max_words} คำ" if target_min_words else "สรุปให้กระชับตามความเหมาะสม"
    
    # 5) System prompt - ย้ายข้อห้าม/ข้อกำหนดทั้งหมดมาไว้ที่นี่
    ARTICLE_SYSTEM = f"""คุณเป็นนักเขียนบทความภาษาไทยมืออาชีพ ทำหน้าที่สรุปเนื้อหาจากคลิปวิดีโอ

ข้อห้ามที่ต้องปฏิบัติตามอย่างเคร่งครัด:
1. ห้ามใส่หัวข้อ/ชื่อบทความ (เช่น **ความสำคัญของ...** หรือ # หัวข้อ)
2. ห้ามใช้ตัวหนา (**) หรือ markdown ใดๆ
3. ห้ามใช้บูลเล็ต/เลขลิสต์
4. ห้ามแต่งเรื่องหรือข้อมูลที่ไม่มีในเนื้อหา
5. ห้ามขึ้นต้นด้วย "สวัสดี", "วันนี้", "บทความนี้", "คลิปนี้", "เราจะมาดู", "ในปัจจุบัน"
6. ห้ามพูดถึงคำว่า "transcript", "เนื้อหานี้", "ข้อความนี้", "ผู้พูด"
7. ห้ามแสดง instructions หรือข้อกำหนดใดๆ ในคำตอบ

ข้อกำหนดสำคัญ:
- เนื้อหาต้นฉบับมาจากคลิปวิดีโอ - ให้สรุปเป็นบทความเล่าเรื่อง
- รวมข้อมูลจากภาพเข้าเป็นส่วนหนึ่งของเนื้อหาอย่างเป็นธรรมชาติ
- ถ้าภาพมีตัวเลข/เปอร์เซ็นต์/ข้อมูลเฉพาะ ให้ใส่โดยไม่ต้องบอกว่า "จากภาพ"
- คงตัวเลข/เวลา/จำนวน/ชื่อเฉพาะตามต้นฉบับ
- เขียนเป็นย่อหน้าต่อเนื่อง เล่าเนื้อหาตรงๆ
- แก้ไขคำผิดสะกดที่เกิดจากการถอดเสียง

ตอบเฉพาะบทความที่สรุปเนื้อหาเท่านั้น"""

    # 6) User prompt - เหลือแค่คำสั่งสั้นๆ + เนื้อหา
    prompt = f"""สรุปเนื้อหาคลิปวิดีโอนี้เป็นบทความภาษาไทย {length_instruction}

{combined_context}"""

    # 7) เรียก LLM
    raw = ensure_thai(ollama_summarize(prompt, system=ARTICLE_SYSTEM)) or ""
    
    # 7) Post-processing: ลบหัวข้อ/markdown/instructions ที่ LLM อาจใส่มา
    raw = re.sub(r"^#+\s*.+$", "", raw, flags=re.MULTILINE)  # ลบ headings
    raw = re.sub(r"\*\*[^*]+\*\*", "", raw)  # ลบ bold
    raw = re.sub(r"^\*\*ข้อห้าม.*$", "", raw, flags=re.MULTILINE)  # ลบ instruction lines
    raw = re.sub(r"^\*\*ข้อกำหนด.*$", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^\d+\.\s*ห้าม.*$", "", raw, flags=re.MULTILINE)  # ลบ numbered prohibitions
    raw = re.sub(r"\[เนื้อหา.*?\]", "", raw)  # ลบ [เนื้อหา...] markers
    raw = re.sub(r"\[\d+[-–]\d+s?\]", "", raw)  # ลบ timestamp brackets ที่เหลือ
    raw = re.sub(r"\n{3,}", "\n\n", raw)  # ลด newlines ซ้ำ
    raw = raw.strip()
    
    return raw

def extract_single_keyword_th(text: str) -> str:
    """
    ใช้ LLM สกัด 'คำสำคัญหลัก' เพียงคำเดียว (ภาษาไทย) จากข้อความที่ให้มา
    """
    prompt = f"""
อ่านบทความต่อไปนี้ แล้วตอบเพียงคำเดียวที่เป็น "คำสำคัญหลัก" เท่านั้น
- ห้ามใช้คำประสมยาวๆ หรือวลี (ให้เลือกคำนามหลักคำเดียว)
- ห้ามเติมคำอธิบาย
- ให้ตอบเฉพาะคำเดียว
- ตอบเป็นภาษาไทย

[บทความ]
{text}
"""
    out = ollama_summarize(prompt, options={"temperature": 0.0, "num_ctx": 1024})
    # ตัดบรรทัด/เว้นวรรคให้เหลือแค่คำเดียว
    words = (out or "").strip().split()
    if not words:
        return "ไม่พบคำสำคัญ"
    keyword = words[0]
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

    transcript = None
    segments = None
    duration = None
    used_youtube_api = False
    
    # 2) Transcript - ลอง YouTube Transcript API ก่อน (ถ้าเปิด)
    if USE_YOUTUBE_TRANSCRIPT:
        try:
            log("📝 ลองใช้ YouTube Transcript API...")
            transcript, segments = transcribe_youtube_api(YOUTUBE_URL, LANGUAGE, step_start=10, step_end=45)
            used_youtube_api = True
            log("✅ ใช้ YouTube Transcript API สำเร็จ!")
        except Exception as e:
            log(f"⚠️ YouTube Transcript API ไม่สำเร็จ: {e}")
            log("🔄 Fallback ไปใช้ Whisper...")
    
    # ถ้า YouTube API ไม่สำเร็จ หรือไม่ได้เปิดใช้ -> ใช้ Whisper
    if transcript is None:
        # 1) โหลดเสียง + ตัดฉาก
        download_audio_wav_16k(YOUTUBE_URL, AUDIO_OUT)
        download_t = time.time()
        download_time = download_t - t0
        send_progress("โหลดวิดีโอ", 10, 100)

        # Transcript ด้วย Whisper
        transcript, segments = transcribe_whisper(AUDIO_OUT,WHISPER_MODEL,LANGUAGE,ASR_DEVICE,step_start=10,step_end=45)
    else:
        send_progress("โหลดวิดีโอ", 10, 100)
        send_progress("ถอดเสียง", 45, 100)
        download_time = 0 
    
    asr_t = time.time()
    asr_time = asr_t - t0 if not used_youtube_api else 0
    send_progress("ถอดเสียง", 45, 100)

    # 3) Caption + OCR
    captioner = VisionCaptioner(VL_MODEL_NAME, VL_DEVICE)
    duration = stream_scene_frames_and_caption(YOUTUBE_URL, FRAMES_DIR, SCENE_THRESH, CAPTIONS_JSON, captioner, video_duration=duration)
    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        caps = json.load(f)
    scene_ts = [c["ts"] for c in caps if "ts" in c]
    # with open(SCENES_JSON, "w", encoding="utf-8") as f:
    #     json.dump(scene_ts, f, ensure_ascii=False, indent=2)

    # ใช้ split_segments_to_scenes ที่จับคู่ตาม timestamp จริง
    facts = split_segments_to_scenes(segments, scene_ts)
    facts = enrich_scenes_with_captions(facts, caps)
    frames_count = len(caps)
    with open(SCENE_FACTS_JSON, "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in facts], f, ensure_ascii=False, indent=2)
    log(f"✅ Scene facts saved -> {SCENE_FACTS_JSON}")
    cap_t = time.time()
    cap_time = cap_t - asr_t
    send_progress("สร้างคำบรรยายภาพ", 80, 100)

    article_th = summarize_article_th(facts)
    send_progress("ทำสรุป", 90, 67)
    
    with open(FINAL_ARTICLE_TXT, "w", encoding="utf-8") as f:
        f.write(wrap_text(article_th))
        
    main_keyword = extract_single_keyword_th(article_th)
    send_progress("ทำสรุป", 95, 100)
    summarize_t = time.time()
    summarize_time = summarize_t - cap_t

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
        bullets_count = 0  # ไม่มี items แล้ว
        article_words = _safe_word_count(FINAL_ARTICLE_TXT)
        transcript_words = _safe_word_count(TRANSCRIPT_TXT)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            # ---- CONFIG / PARAMS ----
            "youtube_url": YOUTUBE_URL,
            "whisper_model": WHISPER_MODEL,
            "whisper_temp": WHISPER_TEMP,
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
    send_progress("บันทึกข้อมูล", 99, 80)


if __name__ == "__main__":
    main()
