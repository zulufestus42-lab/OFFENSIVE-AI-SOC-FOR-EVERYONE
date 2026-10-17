 import os
import warnings
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import httpx
from sklearn.exceptions import InconsistentVersionWarning

# 1. Suppress scikit-learn version mismatch warnings on model load
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

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

    timestamp = datetime.utcnow().isoformat()
    path = request.headers.get("x-original-uri", "/v1/scan")

    # Fetch geolocation data (latitude & longitude) based on IP
    latitude, longitude = 0.0, 0.0
    try:
        # Skip lookup for local/private IPs
        if client_ip not in ["127.0.0.1", "localhost", "::1"]:
            response = httpx.get(f"https://ipapi.co/{client_ip}/json/", timeout=2.0)
            if response.status_code == 200:
                geo_data = response.json()
                latitude = geo_data.get("latitude", 0.0)
                longitude = geo_data.get("longitude", 0.0)
    except Exception:
        pass

    # Run ML inference
    vec = vectorizer.transform([req.payload])
    pre = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0].max()

    attack_type = labels_map[pre]
    action = "BLOCK" if attack_type != "Benign" else "ALLOW"

    # Construct the telemetry payload for dashboards and live world maps
    attack_log = {
        "IP": client_ip,
        "TIMESTAMP": timestamp,
        "PATH": path,
        "LATITUDE": latitude,
        "LONGITUDE": longitude,
        "PAYLOAD": attack_type,
        "CONFIDENCE": f"{round(float(prob) * 100, 2)}%",
        "ACTION": action
    }

    # Optional: Push attack_log to customer's webhook or database in real-time
    # requests.post("https://customer-dashboard.com/api/telemetry", json=attack_log, timeout=1.0)

    return attack_log
