import os
import warnings
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.exceptions import InconsistentVersionWarning
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from openai import OpenAI

# Forwards each scan to the live Offense AI dashboard. Keep dashboard_push.py next to this file.
from dashboard_push import push_to_dashboard

# 1. Suppress scikit-learn version mismatch warnings on model load
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Database Setup (SQLite for zero-config persistence)
DATABASE_URL = "sqlite:///./waf_logs.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AttackLogDB(Base):
    __tablename__ = "attack_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    ip = Column(String, index=True)
    timestamp = Column(String)
    path = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    payload_type = Column(String)
    confidence = Column(String)
    action = Column(String)

# Create the database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="OFFENSIVE AI SOC FOR EVERYONE")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model artifacts
model = joblib.load("off_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
labels_map = joblib.load("label_map.joblib")

# Initialize OpenAI client pointing to Groq's free endpoint
llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)


# 2. Add root route to handle Render health checks (fixes HEAD / 404 error)
@app.get("/")
@app.head("/")
def read_root():
    return {"status": "ok", "message": "OFFENSIVE AI SOC FOR EVERYONE"}


class Offensive(BaseModel):
    payload: str


class CrawlerRequest(BaseModel):
    target_url: str


class ReportRequest(BaseModel):
    scan_results: dict


@app.post("/v1/scan")
def scan(req: Offensive, request: Request, background_tasks: BackgroundTasks):
    # Extract client IP (handling proxies/load balancers like Render or Cloudflare)
    client_ip = request.client.host if request.client else "127.0.0.1"
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

    # Central Africa Time (Zambia is UTC+2)
    CAT = timezone(timedelta(hours=2))
    timestamp = datetime.now(CAT).isoformat()
    
    path = request.headers.get("x-original-uri", "/v1/scan")

    # Fetch geolocation data (latitude & longitude) based on IP
    latitude, longitude = 0.0, 0.0
    try:
        if client_ip not in ["127.0.0.1", "localhost", "::1"]:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = httpx.get(f"https://ipapi.co/{client_ip}/json/", headers=headers, timeout=4.0, verify=False)
            if response.status_code == 200:
                geo_data = response.json()
                latitude = geo_data.get("latitude", 0.0) or 0.0
                longitude = geo_data.get("longitude", 0.0) or 0.0
    except Exception as e:
        print(f"Geo lookup error: {e}")
        pass

    # Run ML inference
    vec = vectorizer.transform([req.payload])
    pre = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0].max()

    attack_type = labels_map[pre]
    action = "BLOCK" if attack_type != "Benign" else "ALLOW"
    confidence_str = f"{round(float(prob) * 100, 2)}%"

    # Construct the telemetry payload
    attack_log = {
        "IP": client_ip,
        "TIMESTAMP": timestamp,
        "PATH": path,
        "LATITUDE": latitude,
        "LONGITUDE": longitude,
        "PAYLOAD": attack_type,
        "CONFIDENCE": confidence_str,
        "ACTION": action
    }

    # Save telemetry log to SQLite database
    db = SessionLocal()
    try:
        db_record = AttackLogDB(
            ip=client_ip,
            timestamp=timestamp,
            path=path,
            latitude=latitude,
            longitude=longitude,
            payload_type=attack_type,
            confidence=confidence_str,
            action=action
        )
        db.add(db_record)
        db.commit()
    except Exception as db_err:
        print(f"Database save error: {db_err}")
    finally:
        db.close()

    # Push this event to the live dashboard after the response is sent (never blocks the scan).
    background_tasks.add_task(push_to_dashboard, attack_log, req.payload)

    return attack_log


# 3. Endpoint for dashboards to fetch recent attack logs for the live world map
@app.get("/v1/logs")
def get_logs(limit: int = 50):
    db = SessionLocal()
    try:
        logs = db.query(AttackLogDB).order_by(AttackLogDB.id.desc()).limit(limit).all()
        return [
            {
                "ID": log.id,
                "IP": log.ip,
                "TIMESTAMP": log.timestamp,
                "PATH": log.path,
                "LATITUDE": log.latitude,
                "LONGITUDE": log.longitude,
                "PAYLOAD": log.payload_type,
                "CONFIDENCE": log.confidence,
                "ACTION": log.action
            }
            for log in logs
        ]
    finally:
        db.close()


# 4. Active DAST Crawler & Vulnerability Assessment Endpoint
@app.post("/v1/crawler/scan")
def crawler_scan(req: CrawlerRequest):
    target = req.target_url
    vulnerabilities = []
    
    test_payloads = [
        {
            "type": "SQLi", 
            "payload": "' OR '1'='1", 
            "indicators": ["sql syntax", "syntax error", "unclosed quotation mark", "mysql_fetch", "pg_query"]
        },
        {
            "type": "Reflected XSS", 
            "payload": "<script>alert('XSS_TEST')</script>", 
            "indicators": ["<script>alert('XSS_TEST')</script>"]
        },
        {
            "type": "Command Injection", 
            "payload": "; cat /etc/passwd", 
            "indicators": ["root:x:", "bin:x:", "daemon:x:"]
        }
    ]

    try:
        headers = {"User-Agent": "OffensiveAI-ActiveScanner/1.0"}
        client = httpx.Client(timeout=30.0, follow_redirects=True, verify=False)
        
        response = client.get(target, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        forms = soup.find_all("form")
        
        discovered_forms = []
        for form in forms:
            action = form.get("action", "")
            method = form.get("method", "get").lower()
            form_url = urljoin(target, action)
            inputs = form.find_all(["input", "textarea"])
            input_names = [i.get("name") for i in inputs if i.get("name")]
            
            if input_names:
                discovered_forms.append({
                    "path": form_url,
                    "method": method,
                    "parameters": input_names
                })

        if not discovered_forms:
            discovered_forms.append({
                "path": target,
                "method": "get",
                "parameters": ["q", "search", "id"]
            })

        for form in discovered_forms:
            form_path = form["path"]
            method = form["method"]
            
            for param in form["parameters"]:
                for item in test_payloads:
                    attack_type = item["type"]
                    probe = item["payload"]
                    indicators = item["indicators"]
                    
                    try:
                        data = {param: probe}
                        if method == "post":
                            res = client.post(form_path, data=data, headers=headers)
                        else:
                            res = client.get(form_path, params=data, headers=headers)
                        
                        response_body = res.text.lower()
                        is_vulnerable = False
                        matched_indicator = ""
                        
                        for indicator in indicators:
                            if indicator.lower() in response_body:
                                is_vulnerable = True
                                matched_indicator = indicator
                                break
                        
                        vec = vectorizer.transform([probe])
                        prob = model.predict_proba(vec)[0].max()

                        if is_vulnerable:
                            vulnerabilities.append({
                                "PATH": form_path,
                                "METHOD": method.upper(),
                                "PARAMETER": param,
                                "VULNERABILITY_TYPE": attack_type,
                                "TEST_PAYLOAD": probe,
                                "EVIDENCE": f"Triggered by matching indicator: '{matched_indicator}'",
                                "CONFIDENCE": f"{round(float(prob) * 100, 2)}%"
                            })
                            
                    except Exception:
                        continue

        client.close()

        return {
            "TARGET": target,
            "STATUS": "SUCCESS",
            "TOTAL_FORMS_FOUND": len(forms),
            "VULNERABILITIES_DETECTED": len(vulnerabilities),
            "FINDINGS": vulnerabilities,
            "INSTRUCTION_FOR_LLM": "Analyze these confirmed live findings. Write an executive summary, explain the exact attack vector, and output secure remediation code blocks."
        }

    except Exception as e:
        return {
            "TARGET": target,
            "STATUS": "ERROR",
            "DETAILS": str(e)
        }


# 5. Free LLM-Powered Automated Security Report Generator Endpoint
@app.post("/v1/report/generate")
def generate_llm_report(req: ReportRequest):
    try:
        findings_json = str(req.scan_results)
        
        system_prompt = (
            "You are the official automated intelligence reporting engine for 'OFFENSIVE AI SOC FOR EVERYONE'. "
            "Every report you create must prominently display 'OFFENSIVE AI SOC FOR EVERYONE' as the primary header. "
            "Adopt the persona of an elite CISO and security auditor. "
            "Analyze the provided live DAST vulnerability findings and output: "
            "1. Executive Summary branded for 'OFFENSIVE AI SOC FOR EVERYONE' "
            "2. Technical Attack Vector Breakdown "
            "3. Production-ready secure remediation code patches."
        )
        
        user_prompt = f"Here are the live target scan results:\n{findings_json}\n\nGenerate the complete CISO security report now."

        response = llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        report_text = response.choices[0].message.content

        return {
            "STATUS": "SUCCESS",
            "AI_SECURITY_REPORT": report_text
        }

    except Exception as e:
        return {
            "STATUS": "ERROR",
            "DETAILS": str(e)
        }
