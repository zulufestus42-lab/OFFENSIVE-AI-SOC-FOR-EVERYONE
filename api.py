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
#
# Works on BOTH classic server-rendered sites (HTML <form> tags) AND modern
# single-page apps / JSON APIs (Angular, React, Vue) that have no forms and talk
# to /rest or /api endpoints. Endpoints that don't exist simply 404 and are
# skipped, so the same scan is safe to run against any authorized target.

# SQL-error fingerprints across common databases. If any of these leak into a
# response after we send a single quote, the parameter is injectable.
SQL_ERROR_INDICATORS = [
    "sql syntax", "syntax error", "unclosed quotation", "unterminated string",
    "unrecognized token", "sqlite_error", "sqlite3.", "mysql_fetch", "mysqli",
    "pg_query", "psql:", "ora-0", "odbc", "sqlstate", "sqlexception",
    "you have an error in your sql", "warning: mysql", "near \"'\"",
]
XSS_MARKER = "offai7xssprobe"  # unique token so a reflection can't be a coincidence


def _classify_confidence(probe: str) -> str:
    """Score the probe string with the existing ML model (best effort)."""
    try:
        vec = vectorizer.transform([probe])
        prob = model.predict_proba(vec)[0].max()
        return f"{round(float(prob) * 100, 2)}%"
    except Exception:
        return "n/a"


@app.post("/v1/crawler/scan")
def crawler_scan(req: CrawlerRequest):
    target = req.target_url.rstrip("/")
    vulnerabilities = []
    forms_found = 0
    probe_log = []  # diagnostics: what each probe actually saw
    # Browser-like UA so WAF/Cloudflare bot filters don't silently challenge us.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
    }

    def log(path, **kw):
        probe_log.append({"path": path, **kw})

    def record(path, method, param, vuln_type, severity, evidence, probe):
        vulnerabilities.append({
            "PATH": path,
            "METHOD": method.upper(),
            "PARAMETER": param,
            "VULNERABILITY_TYPE": vuln_type,
            "SEVERITY": severity,
            "TEST_PAYLOAD": probe,
            "EVIDENCE": evidence,
            "CONFIDENCE": _classify_confidence(probe),
        })

    try:
        client = httpx.Client(timeout=20.0, follow_redirects=True, verify=False, headers=headers)

        # ---- Phase 1: classic HTML <form> crawl -------------------------------
        form_payloads = [
            {"type": "SQL Injection", "severity": "high", "payload": "' OR '1'='1",
             "indicators": SQL_ERROR_INDICATORS},
            {"type": "Reflected XSS", "severity": "medium", "payload": f"<script>{XSS_MARKER}</script>",
             "indicators": [f"<script>{XSS_MARKER}</script>"]},
            {"type": "Command Injection", "severity": "critical", "payload": "; cat /etc/passwd",
             "indicators": ["root:x:", "bin:x:", "daemon:x:"]},
        ]
        try:
            response = client.get(target)
            log(target, phase="landing", status=response.status_code, bytes=len(response.text))
            soup = BeautifulSoup(response.text, "html.parser")
            forms = soup.find_all("form")
            forms_found = len(forms)
            for form in forms:
                action = form.get("action", "")
                method = form.get("method", "get").lower()
                form_url = urljoin(target + "/", action)
                input_names = [i.get("name") for i in form.find_all(["input", "textarea"]) if i.get("name")]
                for param in input_names:
                    for item in form_payloads:
                        try:
                            data = {param: item["payload"]}
                            res = client.post(form_url, data=data) if method == "post" else client.get(form_url, params=data)
                            body = res.text.lower()
                            hit = next((ind for ind in item["indicators"] if ind.lower() in body), None)
                            if hit:
                                record(form_url, method, param, item["type"], item["severity"],
                                       f"Response matched indicator '{hit}' after injection", item["payload"])
                        except Exception:
                            continue
        except Exception as e:
            log(target, phase="landing", error=str(e)[:200])

        # ---- Phase 2: JSON-API / SPA probes -----------------------------------
        # Login auth-bypass: a tautology in a login field should never succeed.
        login_paths = ["/rest/user/login", "/api/login", "/login", "/api/users/login", "/auth/login"]
        for lp in login_paths:
            url = target + lp
            try:
                baseline = client.post(url, json={"email": f"nouser-{XSS_MARKER}@example.com",
                                                  "password": f"wrong-{XSS_MARKER}"})
            except Exception as e:
                log(url, phase="login-baseline", error=str(e)[:200])
                continue
            if baseline.status_code == 404:
                log(url, phase="login", status=404, note="not found, skipped")
                continue
            try:
                injected = client.post(url, json={"email": "' OR 1=1--", "password": "x"})
            except Exception as e:
                log(url, phase="login-inject", error=str(e)[:200])
                continue
            body = injected.text.lower()
            log(url, phase="login", baseline_status=baseline.status_code,
                inject_status=injected.status_code, inject_bytes=len(injected.text),
                has_token=("token" in body or "authentication" in body or "bearer" in body))
            leaked = next((ind for ind in SQL_ERROR_INDICATORS if ind in body), None)
            if leaked:
                record(url, "POST", "email", "SQL Injection", "high",
                       f"Login endpoint leaked a SQL error ('{leaked}') on a single-quote payload", "' OR 1=1--")
            elif injected.status_code == 200 and baseline.status_code in (400, 401, 403) and \
                    ("token" in body or "authentication" in body or "bearer" in body):
                record(url, "POST", "email", "SQL Injection (authentication bypass)", "critical",
                       "Tautology payload bypassed login: rejected valid-shaped bad creds, "
                       "but the injection returned 200 with an auth token", "' OR 1=1--")

        # Search / listing injection + reflected XSS.
        search_paths = ["/rest/products/search", "/api/products", "/rest/track-order",
                        "/search", "/api/search", "/products"]
        for sp in search_paths:
            url = target + sp
            # SQL injection via single quote
            try:
                inj = client.get(url, params={"q": "'"})
            except Exception as e:
                log(url, phase="search", error=str(e)[:200])
                continue
            if inj.status_code == 404:
                log(url, phase="search", status=404, note="not found, skipped")
                continue
            body = inj.text.lower()
            leaked = next((ind for ind in SQL_ERROR_INDICATORS if ind in body), None)
            log(url, phase="search", status=inj.status_code, bytes=len(inj.text), sql_error=bool(leaked))
            if leaked:
                record(url, "GET", "q", "SQL Injection", "high",
                       f"Search endpoint leaked a SQL error ('{leaked}') on a single-quote payload", "'")
            # Reflected XSS
            try:
                xss = client.get(url, params={"q": f"<script>{XSS_MARKER}</script>"})
                if f"<script>{XSS_MARKER}</script>".lower() in xss.text.lower():
                    record(url, "GET", "q", "Reflected XSS", "medium",
                           "Search parameter reflected the script payload unescaped",
                           f"<script>{XSS_MARKER}</script>")
            except Exception:
                pass

        client.close()

        severities = [v["SEVERITY"] for v in vulnerabilities]
        return {
            "TARGET": target,
            "STATUS": "SUCCESS",
            "TOTAL_FORMS_FOUND": forms_found,
            "VULNERABILITIES_DETECTED": len(vulnerabilities),
            "FINDINGS": vulnerabilities,
            "HIGHEST_SEVERITY": ("critical" if "critical" in severities else
                                 "high" if "high" in severities else
                                 "medium" if "medium" in severities else "none"),
            "PROBE_LOG": probe_log,
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
