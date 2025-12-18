import os
import sys
import json
import requests
import re
from typing import Optional, Dict, Any

# CONFIG
OLLAMA_API = os.environ.get("OLLAMA_API", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b") 
SYSTEM_PROMPT_TH = "คุณคือผู้ช่วยสรุปความฉลาดสูง ตอบเป็นภาษาไทยเท่านั้น"

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
    keyword = out.strip().split()[0]
    keyword = re.sub(r"[^\wก-๙]", "", keyword)
    return keyword or "ไม่พบคำสำคัญ"

if __name__ == "__main__":
    # ใส่ข้อความที่ต้องการทดสอบตรงนี้
    TEST_TEXT = """
    **เรื่องความยาวรอบเอลกูด้วยไม่ใช้สายวัด**  ในโลกสมัยใหม่ เรามีเทคโนโลยีที่พัฒนาเพิ่มขึ้นทุกวัน
ทำให้เราสามารถทำสิ่งต่างๆ ได้อย่างรวดเร็ว เช่น การคำนวณเวลา ความยาว เป็นต้น
แต่ก่อนที่เราจะลงมือปฏิบัติ เราต้องทราบว่าอะไรคือความจริง
ความยาวรอบเอลกูด้วยไม่ใช้สายวัดเป็นหนึ่งในโจทย์ที่หลายๆ คนสนใจ วันนี้
เราจะไปสำรวจเรื่องนี้พร้อมกับหลักฐานจากภาพ  เชื่อกันมันก็จะยาวเท่าไหร่ต่อไปได้กูไหนได้ปะ ไม่ต้องเขิน
ไม่ต้องเขินโอเค ตอนนี้เราได้เห็นข้อสรุปแล้ว
มึงเอาเชื่อกที่มึงได้มาพูกอักแห้ย่าเลยก็ได้เป็นลูกตุมเสร็จแล้ว
มึงจับเวลาการกว่าเกล่าของรูปต้มเฮี่ยนี่ 10 ขัง18.24 เมื่อกี้กูวัดการกว่า 10 ทีได้ 18.24
วินาทีเพราะฉะนั้นการแกว่า 1 ทีก็ต้องเป็น 1.824 วินาทีใส่สูตรศรรมการ T ถ้ากัด 2 พายสะแรล ลูส L ส่วน
GL คือความยาวของเชือก G คือ 9.8 เพราะฉันย้ายข้างมา 3.8 ฟูกัน 1.8 ท้อง 4 ส่วนให้ 2 ภายทั้งหมดกำลัง 2
ก็จะเท่ากับความอย่างของเชือกเท่าไหร่วะ ประมาณ 32 นิ้ว 32 นิ้ว เพราะฉันเอวกูสายไป 2 นิ้ว
จากหลักฐานจากภาพ เราสามารถดูช่วงเวลาเหล่านี้ได้ เช่น ชายคนหนึ่งเขียนบนกระดานขาวตรงหน้ากระดานดำ
หลังจากนั้นก็ไปพบหญิงสาวคนหนึ่งที่ถนน ชายคนหนึ่งใช้คีมโกนคิ้วชายอื่น เป็นต้น
จากเหตุการณ์เหล่านี้เราเห็นว่ามึงเอาเชื่อกที่มึงได้มาพูกอักแห้ย่าเลยก็ได้เป็นลูกตุมเสร็จแล้ว
ในที่สุดเราก็มีคำตอบเกี่ยวกับเรื่องความยาวรอบเอลกูด้วยไม่ใช้สายวัด
เราต้องพิจารณาอ่อนและรู้คุณค่าของค่าที่เราได้หา
และจากการสำรวจเราควรจะตระหนักถึงความสำคัญของเทคโนโลยีที่ช่วยให้เราได้รู้คุณค่าเหล่านี้
    """
    
    print(f"Input: {TEST_TEXT[:100]}...")
    print("-" * 20)
    try:
        kw = extract_single_keyword_th(TEST_TEXT)
        print(f"Extracted Keyword: {kw}")
    except Exception as e:
        print(f"Error: {e}")
