import os
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.exceptions import InconsistentVersionWarning

# 1. Suppress scikit-learn version mismatch warnings on model load
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = FastAPI(title="OFFENSIVE AI SOC FOR EVERYONE")

# Load model artifacts
model = joblib.load("off_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
labels_map = joblib.load("label_map.joblib")


# 2. Add root route to handle Render health checks (fixes HEAD / 404 error)
@app.get("/")
@app.head("/")
def read_root():
    return {"status": "ok", "message": "OFFENSIVE AI SOC API is running"}


class Offensive(BaseModel):
    payload: str


@app.post("/v1/scan")
def scan(req: Offensive):
    vec = vectorizer.transform([req.payload])
    pre = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0].max()

    return {
        "CEO": "FESTUS ZULU",
        "PAYLOAD": labels_map[pre],
        "CONFIDENCE": f"{round(float(prob) * 100, 2)}%",
        "ACTION": "BLOCK" if labels_map[pre] != "Benign" else "ALLOW"
    }
