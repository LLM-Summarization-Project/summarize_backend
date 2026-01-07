"""
Script สำหรับรัน YouTube URLs ทีละ row อัตโนมัติ
รัน 4 รอบต่อ URL โดยเปลี่ยน WHISPER_TEMP ใน .env แล้ว restart worker
ใช้: python batch_summarize.py urls.txt [--start N]
"""
import sys
import time
import subprocess
import requests
from pathlib import Path
import os
import argparse
from datetime import datetime

# ตั้งค่า API
API_BASE = "http://localhost:8081"  # Backend URL
AUTH_BASE = "http://localhost:4005"  # Auth service URL
TEST_USER_ID = 2
SYSTEM_PROMPT_TH = (
    "คุณคือผู้สรุปข่าวภาษาไทย ใช้สำนวนข่าวเล่าเรื่อง กระชับ ชัดเจน ไม่ใช้หัวข้อย่อยหรือบูลเล็ต "
    "หลีกเลี่ยงการพูดซ้ำ และหากอ้างสิ่งที่ยืนยันด้วยภาพให้แทรก (จากภาพ) ภายในประโยค "
    "คงชื่อเฉพาะ/ตัวย่อ/ตัวเลขตามต้นฉบับ"
)


# ⚠️ WHISPER_TEMP ที่จะทดสอบ
WHISPER_TEMPS = [0.0, 0.2, 0.4, 0.6]

# Path to .env file
ENV_FILE = Path(__file__).parent.parent / ".env"

# Path to logs directory
LOGS_DIR = Path(__file__).parent / "logs"

# Global log file handle
LOG_FILE = None

# ⚠️ ใส่ refresh token ที่นี่
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOjYsImlhdCI6MTc2NzczNTc3MCwiZXhwIjoxNzY4MzQwNTcwfQ.eeJjnK1B9Md4NXFyVcWQGzC6LkmTpObCaDMVpQ87ZQU")

# Access token (จะ refresh อัตโนมัติ)
ACCESS_TOKEN = ""

def log(message: str, end: str = "\n"):
    """Print and log a message"""
    print(message, end=end)
    if LOG_FILE:
        LOG_FILE.write(message + (end if end != "\r" else "\n"))
        LOG_FILE.flush()

def setup_logging():
    """Setup log file for this session"""
    global LOG_FILE
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"batch_{timestamp}.log"
    LOG_FILE = open(log_path, "w", encoding="utf-8")
    log(f"📝 Logging to: {log_path}")
    return log_path

def refresh_access_token():
    """ดึง access token ใหม่จาก refresh token"""
    global ACCESS_TOKEN
    
    if not REFRESH_TOKEN:
        log("⚠️ No REFRESH_TOKEN set, cannot refresh")
        return False
    
    try:
        resp = requests.post(
            f"{AUTH_BASE}/auth/refresh",
            cookies={"refresh_token": REFRESH_TOKEN}  # ส่งเป็น cookie แทน
        )
        
        if resp.status_code == 200 or resp.status_code == 201:
            # Access token มาในรูปแบบ cookie, ไม่ใช่ JSON body
            ACCESS_TOKEN = resp.cookies.get("access_token", "")
            if not ACCESS_TOKEN:
                # Fallback: ลองอ่านจาก JSON body
                data = resp.json()
                ACCESS_TOKEN = data.get("accessToken") or data.get("access_token", "")
            log(f"🔄 Token refreshed successfully")
            return True
        else:
            log(f"❌ Refresh failed: {resp.status_code} - {resp.text}")
            return False
            
    except Exception as e:
        log(f"❌ Refresh error: {e}")
        return False

def get_headers():
    """สร้าง headers พร้อม token"""
    if ACCESS_TOKEN:
        return {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    return {"x-user-id": str(TEST_USER_ID)}

def update_env_whisper_temp(temp: float):
    """อัพเดท WHISPER_TEMP ใน .env file"""
    if not ENV_FILE.exists():
        log(f"❌ .env file not found: {ENV_FILE}")
        return False
    
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    new_lines = []
    found = False
    
    for line in lines:
        if line.startswith("WHISPER_TEMP="):
            new_lines.append(f"WHISPER_TEMP={temp}")
            found = True
        else:
            new_lines.append(line)
    
    if not found:
        new_lines.append(f"WHISPER_TEMP={temp}")
    
    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    log(f"   📝 Updated .env: WHISPER_TEMP={temp}")
    return True

def wait_for_completion(job_id: str, timeout: int = 1200) -> dict:  # 20 minutes
    """รอจนกว่า job จะเสร็จ"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{API_BASE}/summary/{job_id}", headers=get_headers())
            
            if resp.status_code == 401:
                if refresh_access_token():
                    continue
                    
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "")
                
                if status == "DONE":
                    return {"status": "success", "data": data}
                elif status in ["FAILED", "ERROR", "CANCEL"]:
                    return {"status": "failed", "error": status}
                
                percent = data.get("percent", 0)
                log(f"   ⏳ {status} - {percent}%", end="\r")
                
        except Exception as e:
            log(f"   ⚠️ Error checking status: {e}")
        
        time.sleep(3)
    
    return {"status": "timeout"}

def submit_job(url: str, temp: float = None) -> dict:
    """ส่ง job ไป queue พร้อม whisperTemp"""
    try:
        payload = {"youtubeUrl": url}
        if temp is not None:
            payload["whisperTemp"] = temp
            
        resp = requests.post(
            f"{API_BASE}/summary",
            json=payload,
            headers=get_headers()
        )
        
        if resp.status_code == 401:
            if refresh_access_token():
                resp = requests.post(
                    f"{API_BASE}/summary",
                    json=payload,
                    headers=get_headers()
                )
        
        if resp.status_code in [200, 201]:
            return resp.json()
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
            
    except Exception as e:
        return {"error": str(e)}

def run_batch(url_file: str, start_job: int = 1):
    """รัน summarization สำหรับแต่ละ URL, 4 รอบต่อ URL
    
    Args:
        url_file: Path to file containing URLs
        start_job: Job number to start from (1-indexed)
    """
    
    # Setup logging
    log_path = setup_logging()
    
    # อ่าน URLs จากไฟล์
    url_path = Path(url_file)
    if not url_path.exists():
        log(f"❌ ไม่พบไฟล์: {url_file}")
        return
    
    urls = [line.strip() for line in url_path.read_text(encoding="utf-8").splitlines() 
            if line.strip() and not line.startswith("#")]
    
    if not urls:
        log("❌ ไม่มี URLs ในไฟล์")
        return
    
    total_jobs = len(urls) * len(WHISPER_TEMPS)
    log(f"📋 พบ {len(urls)} URLs × {len(WHISPER_TEMPS)} temps = {total_jobs} jobs")
    log(f"🌡️ WHISPER_TEMPS: {WHISPER_TEMPS}")
    log(f"🔗 API: {API_BASE}")
    
    if start_job > 1:
        log(f"⏩ Starting from job #{start_job}")
    
    log("=" * 60)
    
    # Refresh token ก่อน
    if REFRESH_TOKEN:
        log("🔄 Refreshing token...")
        refresh_access_token()
    
    results = []
    job_num = 0
    
    for temp in WHISPER_TEMPS:
        log(f"\n{'='*60}")
        log(f"🌡️ เริ่มรอบ WHISPER_TEMP = {temp}")
        log(f"{'='*60}")
        
        for url in urls:
            job_num += 1
            
            # Skip jobs before start_job
            if job_num < start_job:
                log(f"\n[{job_num}/{total_jobs}] ⏭️ Skipping (before start_job={start_job})")
                continue
            
            log(f"\n[{job_num}/{total_jobs}] 🎬 {url}")
            log(f"   🌡️ WHISPER_TEMP = {temp}")
            start_time = time.time()
            
            # 1) ส่ง job พร้อม whisperTemp ใน request body
            log("   📤 Submitting job...")
            submit_result = submit_job(url, temp)
            
            if "error" in submit_result:
                log(f"   ❌ Submit failed: {submit_result['error']}")
                results.append({
                    "url": url, 
                    "temp": temp,
                    "status": "submit_failed", 
                    "time": 0
                })
                continue
            
            job_id = submit_result.get("jobId")
            from_cache = submit_result.get("fromCache", False)
            
            if from_cache:
                log(f"   📦 From cache: {job_id}")
                results.append({
                    "url": url, 
                    "temp": temp,
                    "status": "cached", 
                    "time": 0, 
                    "job_id": job_id
                })
                continue
            
            log(f"   📥 Job ID: {job_id}")
            
            # 2) รอจนเสร็จ
            log("   ⏳ Waiting for completion...")
            wait_result = wait_for_completion(job_id)
            elapsed = time.time() - start_time
            
            if wait_result["status"] == "success":
                log(f"   ✅ Done! ({elapsed:.1f}s)                    ")
                results.append({
                    "url": url, 
                    "temp": temp,
                    "status": "success", 
                    "time": elapsed, 
                    "job_id": job_id
                })
            else:
                log(f"   ❌ {wait_result['status']} ({elapsed:.1f}s)")
                results.append({
                    "url": url, 
                    "temp": temp,
                    "status": wait_result["status"], 
                    "time": elapsed, 
                    "job_id": job_id
                })
    
    # สรุปผล
    log("\n" + "=" * 60)
    log("📊 สรุปผล:")
    success = sum(1 for r in results if r["status"] in ["success", "cached"])
    failed = len(results) - success
    total_time = sum(r["time"] for r in results)
    
    log(f"   ✅ สำเร็จ: {success}/{len(results)}")
    log(f"   ❌ ล้มเหลว: {failed}/{len(results)}")
    log(f"   ⏱️ เวลารวม: {total_time:.1f}s")
    
    if start_job > 1:
        log(f"   ⏩ (started from job #{start_job})")
    
    # สรุปตาม temp
    log("\n📊 สรุปตาม WHISPER_TEMP:")
    for temp in WHISPER_TEMPS:
        temp_results = [r for r in results if r["temp"] == temp]
        temp_success = sum(1 for r in temp_results if r["status"] in ["success", "cached"])
        log(f"   🌡️ {temp}: {temp_success}/{len(temp_results)} สำเร็จ")
    
    if failed > 0:
        log("\n❌ Jobs ที่ล้มเหลว:")
        for r in results:
            if r["status"] not in ["success", "cached"]:
                log(f"   - {r['url']} (temp={r['temp']}) - {r['status']}")
    
    # Close log file
    if LOG_FILE:
        log(f"\n📝 Log saved to: {log_path}")
        LOG_FILE.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch YouTube summarization script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
ตั้งค่า:
  API_BASE = {API_BASE}
  WHISPER_TEMPS = {WHISPER_TEMPS}
  ENV_FILE = {ENV_FILE}
  LOGS_DIR = {LOGS_DIR}
"""
    )
    parser.add_argument("urls_file", help="ไฟล์ที่มี YouTube URLs ทีละบรรทัด")
    parser.add_argument(
        "--start", "-s", 
        type=int, 
        default=1,
        metavar="N",
        help="Start from job number N (1-indexed, default: 1)"
    )
    
    args = parser.parse_args()
    
    if args.start < 1:
        print("❌ --start must be >= 1")
        sys.exit(1)
    
    run_batch(args.urls_file, args.start)
