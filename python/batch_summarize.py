"""
Script สำหรับรัน YouTube URLs ทีละ row อัตโนมัติ
รัน 4 รอบต่อ URL โดยเปลี่ยน WHISPER_TEMP ใน .env แล้ว restart worker
ใช้: python batch_summarize.py urls.txt
"""
import sys
import time
import subprocess
import requests
from pathlib import Path
import os

# ตั้งค่า API
API_BASE = "http://localhost:8081"  # Backend URL
AUTH_BASE = "http://localhost:4005"  # Auth service URL
TEST_USER_ID = 2

# ⚠️ WHISPER_TEMP ที่จะทดสอบ
WHISPER_TEMPS = [0.0, 0.2, 0.4, 0.6]

# Path to .env file
ENV_FILE = Path(__file__).parent.parent / ".env"

# ⚠️ ใส่ refresh token ที่นี่
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOjIsImlhdCI6MTc2NzU0MjE2MywiZXhwIjoxNzY4MTQ2OTYzfQ.LYp1UndWyQ0VdXSMCBlI9RCvRvEIyTuc7DCc6NiKHLs")

# Access token (จะ refresh อัตโนมัติ)
ACCESS_TOKEN = ""

def refresh_access_token():
    """ดึง access token ใหม่จาก refresh token"""
    global ACCESS_TOKEN
    
    if not REFRESH_TOKEN:
        print("⚠️ No REFRESH_TOKEN set, cannot refresh")
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
            print(f"🔄 Token refreshed successfully")
            return True
        else:
            print(f"❌ Refresh failed: {resp.status_code} - {resp.text}")
            return False
            
    except Exception as e:
        print(f"❌ Refresh error: {e}")
        return False

def get_headers():
    """สร้าง headers พร้อม token"""
    if ACCESS_TOKEN:
        return {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    return {"x-user-id": str(TEST_USER_ID)}

def update_env_whisper_temp(temp: float):
    """อัพเดท WHISPER_TEMP ใน .env file"""
    if not ENV_FILE.exists():
        print(f"❌ .env file not found: {ENV_FILE}")
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
    print(f"   📝 Updated .env: WHISPER_TEMP={temp}")
    return True

def wait_for_completion(job_id: str, timeout: int = 600) -> dict:
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
                print(f"   ⏳ {status} - {percent}%", end="\r")
                
        except Exception as e:
            print(f"   ⚠️ Error checking status: {e}")
        
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

def run_batch(url_file: str):
    """รัน summarization สำหรับแต่ละ URL, 4 รอบต่อ URL"""
    
    # อ่าน URLs จากไฟล์
    url_path = Path(url_file)
    if not url_path.exists():
        print(f"❌ ไม่พบไฟล์: {url_file}")
        return
    
    urls = [line.strip() for line in url_path.read_text(encoding="utf-8").splitlines() 
            if line.strip() and not line.startswith("#")]
    
    if not urls:
        print("❌ ไม่มี URLs ในไฟล์")
        return
    
    total_jobs = len(urls) * len(WHISPER_TEMPS)
    print(f"📋 พบ {len(urls)} URLs × {len(WHISPER_TEMPS)} temps = {total_jobs} jobs")
    print(f"🌡️ WHISPER_TEMPS: {WHISPER_TEMPS}")
    print(f"🔗 API: {API_BASE}")
    print("=" * 60)
    
    # Refresh token ก่อน
    if REFRESH_TOKEN:
        print("🔄 Refreshing token...")
        refresh_access_token()
    
    results = []
    job_num = 0
    
    for temp in WHISPER_TEMPS:
        print(f"\n{'='*60}")
        print(f"🌡️ เริ่มรอบ WHISPER_TEMP = {temp}")
        print(f"{'='*60}")
        
        for url in urls:
            job_num += 1
            print(f"\n[{job_num}/{total_jobs}] 🎬 {url}")
            print(f"   🌡️ WHISPER_TEMP = {temp}")
            start_time = time.time()
            
            # 1) ส่ง job พร้อม whisperTemp ใน request body
            print("   📤 Submitting job...")
            submit_result = submit_job(url, temp)
            
            if "error" in submit_result:
                print(f"   ❌ Submit failed: {submit_result['error']}")
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
                print(f"   📦 From cache: {job_id}")
                results.append({
                    "url": url, 
                    "temp": temp,
                    "status": "cached", 
                    "time": 0, 
                    "job_id": job_id
                })
                continue
            
            print(f"   📥 Job ID: {job_id}")
            
            # 2) รอจนเสร็จ
            print("   ⏳ Waiting for completion...")
            wait_result = wait_for_completion(job_id)
            elapsed = time.time() - start_time
            
            if wait_result["status"] == "success":
                print(f"   ✅ Done! ({elapsed:.1f}s)                    ")
                results.append({
                    "url": url, 
                    "temp": temp,
                    "status": "success", 
                    "time": elapsed, 
                    "job_id": job_id
                })
            else:
                print(f"   ❌ {wait_result['status']} ({elapsed:.1f}s)")
                results.append({
                    "url": url, 
                    "temp": temp,
                    "status": wait_result["status"], 
                    "time": elapsed, 
                    "job_id": job_id
                })
    
    # สรุปผล
    print("\n" + "=" * 60)
    print("📊 สรุปผล:")
    success = sum(1 for r in results if r["status"] in ["success", "cached"])
    failed = len(results) - success
    total_time = sum(r["time"] for r in results)
    
    print(f"   ✅ สำเร็จ: {success}/{len(results)}")
    print(f"   ❌ ล้มเหลว: {failed}/{len(results)}")
    print(f"   ⏱️ เวลารวม: {total_time:.1f}s")
    
    # สรุปตาม temp
    print("\n📊 สรุปตาม WHISPER_TEMP:")
    for temp in WHISPER_TEMPS:
        temp_results = [r for r in results if r["temp"] == temp]
        temp_success = sum(1 for r in temp_results if r["status"] in ["success", "cached"])
        print(f"   🌡️ {temp}: {temp_success}/{len(temp_results)} สำเร็จ")
    
    if failed > 0:
        print("\n❌ Jobs ที่ล้มเหลว:")
        for r in results:
            if r["status"] not in ["success", "cached"]:
                print(f"   - {r['url']} (temp={r['temp']}) - {r['status']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ใช้: python batch_summarize.py <urls.txt>")
        print("   urls.txt = ไฟล์ที่มี YouTube URLs ทีละบรรทัด")
        print("")
        print("ตั้งค่า:")
        print(f"   API_BASE = {API_BASE}")
        print(f"   WHISPER_TEMPS = {WHISPER_TEMPS}")
        print(f"   ENV_FILE = {ENV_FILE}")
        sys.exit(1)
    
    run_batch(sys.argv[1])
