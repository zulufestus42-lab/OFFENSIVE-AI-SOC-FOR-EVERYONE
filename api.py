import os
import warnings
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import httpx
from sklearn.exceptions import InconsistentVersionWarning
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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


# 2. Add root route to handle Render health checks (fixes HEAD / 404 error)
@app.get("/")
@app.head("/")
def read_root():
    return {"status": "ok", "message": "OFFENSIVE AI SOC FOR EVERYONE"}


class Offensive(BaseModel):
    payload: str


@app.post("/v1/scan")
def scan(req: Offensive, request: Request):
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
        # Skip lookup for local/private IPs
        if client_ip not in ["127.0.0.1", "localhost", "::1"]:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = httpx.get(f"https://ipapi.co/{client_ip}/json/", headers=headers, timeout=4.0)
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
