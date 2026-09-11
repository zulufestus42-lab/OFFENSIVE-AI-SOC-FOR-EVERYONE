from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
 
app = FastAPI(title="OFFENSIVE AI SOC FOR EVERYONE")

model = joblib.load("off_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
labels_map = joblib.load("label_map.joblib")


class Offensive(BaseModel):
    payload: str
    
@app.post("/v1/scan")
def scan(req:Offensive ):
    vec = vectorizer.transform([req.payload])
    pre = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0].max()

    return {
            "CEO":"FESTUS ZULU",
            "PAYLOAD":labels_map[pre],
            "CONFIDENCE":f"{round(float(prob)*100, 2)}%",
            "ACTION":"BLOCK" if  labels_map[pre] != "Benign" else "ALLOW"
      }
