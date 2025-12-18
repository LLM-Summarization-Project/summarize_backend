import os
import sys
import json
import requests
import re
import textwrap
from typing import Optional, Dict, Any, List

# CONFIG
OLLAMA_API = os.environ.get("OLLAMA_API", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b") 
SYSTEM_PROMPT_TH = "คุณคือผู้ช่วยสรุปความฉลาดสูง ตอบเป็นภาษาไทยเท่านั้น"

# GEN OPTS matches summary.py
GEN_OPTS_QUALITY = {
    "temperature": 0.1,        # Rigid to prevent hallucination
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.15,
    "repeat_last_n": 256,
    "num_ctx": 8192,
    "num_predict": 1024,
    "stop": ["<|eot_id|>", "</s>"],
}

GEN_OPTS_FAST = {
    "temperature": 0.3,
    "num_ctx": 4096,
    "num_predict": 256,
}

ALLOWED_OLLAMA_KEYS = {
    "temperature", "top_p", "top_k",
    "repeat_penalty", "repeat_last_n",
    "num_ctx", "num_predict",
    "stop", "seed",
}

# Regex for helper functions
NUM_PREFIX_RE = re.compile(r"^\s*(?:[-•]+|\(?\d+\)?\.?|[๑-๙]\.|\d+\)|ข้อ\s*\d+\.?)\s*")
PLACEHOLDER_PATTERNS = [
    r"^ประเด็น(แรก|ที่\s*หนึ่ง|ที่\s*สอง|ที่\s*สาม|สี่|ห้า|หก|เจ็ด|แปด)\s*$",
    r"^ข้อ(แรก|ที่\s*\d+)\s*$",
    r"^บูลเล็ต(เสริม)?\s*:?\s*$",
    r"^สรุป(ย่อ)?\s*:?\s*$",
    r"^สกัด.+บูลเล็ต.*$",
]
PLACEHOLDER_RES = [re.compile(p, flags=re.I) for p in PLACEHOLDER_PATTERNS]
THAI_RE = re.compile(r"[ก-๙]")

WEIGHT_MAP = {
    "number": 3, "percent": 3, "timecode": 3, "date": 3,
    "unit": 2, "currency": 2, "keyword_on_screen": 2,
    "short_len": 1, "has_ocr": 2, "url_or_id": 1, "all_caps_token": 1,
}

RE_NUMBER   = re.compile(r"\d")
RE_PERCENT  = re.compile(r"\d+\s*%")
RE_TIMECODE = re.compile(r"\b(?:\d{1,2}:){1,2}\d{2}\b")
RE_DATE     = re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:\d{1,2}[-/]){1,2}\d{2,4})\b")
RE_UNIT     = re.compile(r"\b(?:kg|g|km|m|cm|mm|mb|gb|tb|fps|hz|px|ms|s|min|hr|ชั่วโมง|นาที|วินาที|บาท|กก|กม)\b", re.I)
RE_CURRENCY = re.compile(r"[€£$¥฿]|(?:USD|THB|JPY|EUR)\b", re.I)
RE_URL_ID   = re.compile(r"(https?://\S+)|\b[A-Z0-9]{6,}\b")
RE_ALLCAPS  = re.compile(r"\b[A-Z]{2,}\b")
RE_HEADERY  = re.compile(r"\b(introduction|overview|summary|สรุป|บทนำ|หัวข้อ|สารบัญ|สไลด์|ผลลัพธ์|วิธีใช้|ข้อควรทราบ)\b", re.I)

# ===== HELPER FUNCTIONS =====

def sanitize_ollama_options(opts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not opts:
        return {}
    return {k: v for k, v in opts.items() if k in ALLOWED_OLLAMA_KEYS}

def ollama_summarize(
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    system: Optional[str] = None,
    stream: bool = False,           
    timeout: int = 600,
) -> str:
    base = OLLAMA_API
    if system is None:
        system = SYSTEM_PROMPT_TH
    
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

    try:
        resp = requests.post(chat_base, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg = (data.get("message") or {}).get("content", "")
        return (msg or "").strip()
    except Exception as e:
        print(f"Ollama Error: {e}")
        return ""

def looks_thai(s: str) -> bool:
    return bool(THAI_RE.search(s or ""))

def ensure_thai(text: str, max_chars: int = None) -> str:
    t = (text or "").strip()
    if looks_thai(t):
        if max_chars and len(t) > max_chars:
            t = t[:max_chars]
        return t
    p1 = f"แปลเป็นภาษาไทย: {t}"
    return (ollama_summarize(p1, options=GEN_OPTS_FAST) or "").strip()

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
        if not x: continue
        if is_placeholder(x): continue
        out.append(x)
    return out

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

def score_informativeness(s: str) -> int:
    sigs = _detect_signals(s, has_ocr=False)
    return sum(WEIGHT_MAP[k] for k,v in sigs.items() if v)

def is_duplicate(new_item: str, existing: List[str], thr: float = 0.6) -> bool:
    # Simplified dedup
    if new_item in existing: return True
    return False

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
            vis_points.append(txt)
    vis_points = dedup_and_rerank(vis_points, max_items=4)

    prompt = f"""
เขียนบทความ "สรุปสาระสำคัญ" (Factual Summary) ภาษาไทย ความยาวประมาณ {target_min_words}-{target_max_words} คำ 
โดยอ้างอิง TRANSCRIPT ด้านล่างเป็นหลัก 
**ข้อกำหนดการเขียน**
- เน้นสรุปข้อเท็จจริงตามเนื้อหาที่ให้มาเท่านั้น (Fact-based) ห้ามแต่งนิยายหรือเรื่องสมมติ
- อินโทรเกริ่นภาพรวมสั้น ๆ
- อธิบายเนื้อหาหลักแบบต่อเนื่อง 2–4 ย่อหน้า
- ห้ามใช้หัวข้อย่อย/บูลเล็ต/เลขลิสต์
- ปิดท้ายด้วยบทสรุปสั้นๆ
- ห้ามขึ้นต้นด้วย "สวัสดี" หรือ "ในบทความนี้"

[TRANSCRIPT]
{transcript_src}

[หลักฐานจากภาพ]
{chr(10).join(f"- {p}" for p in vis_points)}
"""

    # Using the strict GEN_OPTS_QUALITY
    raw = ensure_thai(ollama_summarize(prompt, options=GEN_OPTS_QUALITY)) or ""
    return raw

if __name__ == "__main__":
    # Test Transcript (Derived from the "Round Waist" hallucination case)
    TRANSCRIPT_TEST = """
    วัดความยาวรอบเอลกูด้วยไม่ใช้สายวัดทำยังไง1 เลยมึงเอาเชื่อกมาดูกันนะว่า รอบเอลกูยาวเท่าไหร่ต่าได้กูหน่อยได้ป่ะกูแอบชอบกูมั้ยเนี่ยเอา ไม่ต้องเขลล่ะโอเค ตอนนี้ได้เชื่อกมานะข้าททนที่ 2 นะ มึงเอาเชื่อกที่มึงได้มาฟูกอ่ะเยี่ยอะไรก็ได้เป็นลูกตุมเสร็จแล้วมึงจับเวลา Meng แก่งของทุ้มเฮี้ย นี่ 10 ขัง18.24เมื่อกี้กูวัดการแก่ง 10 ที ได้ 18.24 วินาทีเพราะฉะนั้น การแก่ง 1 ที ก็ต้องเป็น 1.824 วินาทีใส่สุดสรรมการT ถ้ากัด 2.5 สแคลรูจ L ส่วน GL คือความยาวของเชือกG คือ 9.8เพราะฉะนั้นย้ายข้างมา 9.8 คือกัน 1.824
    """
    
    # Mock Visual Items (Empty for this test as hallucination came from transcript)
    VISUAL_ITEMS = [
]

    print("--- INPUT TRANSCRIPT ---")
    print(TRANSCRIPT_TEST.strip()[:200] + "...")
    print("\n--- GENERATING ARTICLE (strict mode) ---")
    
    article = summarize_article_th(TRANSCRIPT_TEST, VISUAL_ITEMS)
    
    print("\n--- RESULT ---")
    print(article)
    print("\n--- CHECK ---")
    if "Newton" in article or "Round Earth" in article or "โลกกลม" in article:
        print("❌ FAIL: Hallucination detected (Newton/Round Earth)")
    else:
        print("✅ PASS: No obvious hallucinations detected")
