from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import redis
import uuid
import json
import time
from email import policy
from email.parser import BytesParser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Redis Bağlantısını yapıyoruz 
r = redis.Redis(host='localhost', port=8443, db=0)

app = FastAPI(title="Siber Güvenlik Mail Motoru API")
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)

def check_whitelist_fast(sender):
    sender = (sender or "").lower()
    safe_domains = ["google.com", "microsoft.com", "apple.com", "amazon.com", ".gov.tr", ".edu.tr"] #bu tur domainleri safe domains olarak isaretle ve ham goster
    return any(domain in sender for domain in safe_domains)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Hata: index.html dosyası bulunamadı!</h1>"

@app.get("/result-view", response_class=HTMLResponse)
async def result_view():
    try:
        with open("templates/result.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Hata: result.html dosyası bulunamadı!</h1>"

@app.post("/analyze")
async def scan_email(file: UploadFile = File(...)):
    start_time = time.time()

    original_filename = (file.filename or "").strip()
    if not original_filename:
        original_filename = "mail.eml"

    content = await file.read()
    msg = BytesParser(policy=policy.default).parsebytes(content)
    sender = str(msg['from'] or "")
    subject = str(msg['subject'] or "")

    # 1) WHITELIST FAST-PATH
    if check_whitelist_fast(sender):
        job_id = str(uuid.uuid4())

        elapsed = time.time() - start_time
        result_data = {
            "status": "COMPLETED",
            "final_verdict": "HAM",
            "reason": "Whitelist (Güvenli Gönderici)",
            "file": original_filename,
            "process_time": elapsed,
            "time_ms": elapsed * 1000,

            # Frontend’in bar/etiket için beklediği alanlar:
            "svm_result": "HAM (%100)",
            "xgb_result": "HAM",

            "ocr_details": None,
        }

        r.setex(f"result:{job_id}", 600, json.dumps(result_data))
        return {"status": "INSTANT_HAM", "job_id": job_id}

    # 2) SLOW-PATH (QUEUE) #redis icin kuyruk yapisi 
    html_content = ""
    if msg.get_body(preferencelist=('html')):
        html_content = msg.get_body(preferencelist=('html')).get_content()
    else:
        try:
            html_content = msg.get_body(preferencelist=('plain')).get_content()
        except:
            html_content = ""

    job_id = str(uuid.uuid4())

    task_payload = {
        "job_id": job_id,
        "sender": sender,
        "subject": subject,
        "html_content": html_content,
        "file": original_filename,  # <-- worker’a geçsin
    }

    try:
        r.rpush('email_queue', json.dumps(task_payload))

        # Başlangıç durumunu kaydet (file'ı burada da tut)
        r.setex(
            f"result:{job_id}",
            600,
            json.dumps({"status": "PROCESSING", "file": original_filename})
        )

        return {"status": "QUEUED", "job_id": job_id}
    except redis.exceptions.ConnectionError: #redise baglanamama durumu 
        return JSONResponse(status_code=500, content={"detail": "Redis bağlantı hatası! Redis sunucusu açık mı?"})

@app.get("/analyze/{job_id}")
async def get_result(job_id: str):
    data = r.get(f"result:{job_id}")
    if data:
        return json.loads(data)
    return {"status": "NOT_FOUND"}

