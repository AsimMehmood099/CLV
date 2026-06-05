import sys
import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

client        = MongoClient("mongodb://localhost:27017/")
db            = client["clv_project"]
customers_col = db["customers"]

app = FastAPI(title="CLV Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODELS    = {}
MODEL_DIR = os.path.join(BACKEND_DIR, "../models")
DATASETS  = ["AliExpress", "ebay", "lazada", "walmart"]

def load_all_models():
    for name in DATASETS:
        try:
            MODELS[name] = {
                "clv":     joblib.load(f"{MODEL_DIR}/{name}_clv.pkl"),
                "segment": joblib.load(f"{MODEL_DIR}/{name}_segment.pkl"),
                "scaler":  joblib.load(f"{MODEL_DIR}/{name}_scaler.pkl"),
            }
            print(f"Loaded: {name}")
        except Exception as e:
            print(f"Could not load {name}: {e}")

load_all_models()

class CustomerInput(BaseModel):
    customer_id:    str   = "NEW_001"
    frequency:      float
    recency:        float
    T:              float
    monetary_value: float
    dataset:        str   = "AliExpress"

@app.get("/")
def root():
    return {"status": "CLV API Running", "datasets": list(MODELS.keys())}

@app.get("/datasets")
def datasets():
    return {"datasets": DATASETS}

@app.post("/predict")
def predict(data: CustomerInput):
    m = MODELS.get(data.dataset)
    if not m:
        return {"error": f"No model for {data.dataset}"}
    X        = np.array([[data.frequency, data.recency, data.T, data.monetary_value]])
    X_scaled = m["scaler"].transform(X)
    clv_pred = m["clv"].predict(X_scaled)[0]
    X_seg    = np.append(X[0], clv_pred).reshape(1, -1)
    segment  = m["segment"].predict(X_seg)[0]
    result = {
        "customer_id":    data.customer_id,
        "dataset":        data.dataset,
        "predicted_clv":  round(float(clv_pred), 2),
        "segment":        segment,
        "frequency":      data.frequency,
        "recency":        data.recency,
        "T":              data.T,
        "monetary_value": data.monetary_value,
        "created_at":     datetime.now()
    }
    customers_col.insert_one({**result})
    return result

@app.get("/customers/{dataset}")
def customers(dataset: str, limit: int = 500):
    return {"customers": list(customers_col.find(
        {"dataset": dataset}, {"_id": 0}
    ).limit(limit))}

@app.get("/customers/{dataset}/top")
def top_customers(dataset: str):
    return {"customers": list(customers_col.find(
        {"dataset": dataset}, {"_id": 0}
    ).sort("predicted_clv", -1).limit(10))}

@app.get("/summary/{dataset}")
def summary(dataset: str):
    pipeline = [
        {"$match": {"dataset": dataset}},
        {"$group": {
            "_id": "$segment",
            "count": {"$sum": 1},
            "avg_clv": {"$avg": "$predicted_clv"},
            "total_clv": {"$sum": "$predicted_clv"}
        }}
    ]
    return {"summary": list(customers_col.aggregate(pipeline))}

@app.get("/stats/{dataset}")
def stats(dataset: str):
    data = list(customers_col.find({"dataset": dataset}))
    if not data:
        return {"total_customers": 0, "avg_clv": 0, "max_clv": 0, "total_revenue": 0}
    clvs = [c["predicted_clv"] for c in data]
    return {
        "total_customers": len(data),
        "avg_clv":         round(sum(clvs) / len(clvs), 2),
        "max_clv":         round(max(clvs), 2),
        "total_revenue":   round(sum(clvs), 2)
    }

@app.post("/upload/predict/{dataset_name}")
async def upload_predict(dataset_name: str, file: UploadFile = File(...)):
    sys.path.insert(0, os.path.join(BACKEND_DIR, "../"))
    from train import clean_data, build_clv
    import lifetimes

    m = MODELS.get(dataset_name)
    if not m:
        return {"error": f"No model for {dataset_name}"}

    df = pd.read_csv(file.file, encoding="ISO-8859-1")
    df = clean_data(df)
    if df is None or len(df) == 0:
        return {"error": "Could not process file - no valid rows after cleaning"}

    clv = build_clv(df)

    if clv is None or len(clv) == 0:
        clv = lifetimes.utils.summary_data_from_transaction_data(
            df,
            customer_id_col="CustomerID",
            datetime_col="OrderDate",
            monetary_value_col="TotalPrice"
        )
        clv = clv[clv["monetary_value"] > 0]
        clv["frequency"]      = clv["frequency"].clip(lower=1)
        clv["recency"]        = clv["recency"].clip(lower=1)
        clv["T"]              = clv["T"].clip(lower=1)
        clv["monetary_value"] = clv["monetary_value"].clip(lower=0.01)
        clv = clv.replace([float("inf"), float("-inf")], float("nan")).dropna()

    if clv is None or len(clv) == 0:
        return {"error": "No valid customers found. Check date, customer ID, and price columns."}

    results = []
    for idx, row in clv.iterrows():
        try:
            X        = np.array([[row["frequency"], row["recency"], row["T"], row["monetary_value"]]])
            X_scaled = m["scaler"].transform(X)
            clv_pred = m["clv"].predict(X_scaled)[0]
            X_seg    = np.append(X[0], clv_pred).reshape(1, -1)
            segment  = m["segment"].predict(X_seg)[0]
            results.append({
                "customer_id":    str(idx),
                "dataset":        dataset_name,
                "predicted_clv":  round(float(clv_pred), 2),
                "segment":        segment,
                "frequency":      float(row["frequency"]),
                "recency":        float(row["recency"]),
                "T":              float(row["T"]),
                "monetary_value": float(row["monetary_value"]),
                "created_at":     datetime.now()
            })
        except Exception as e:
            print(f"Skipping {idx}: {e}")
            continue

    if results:
        customers_col.insert_many(results)
        for r in results:
            r.pop("_id", None)
    return {
        "total":   len(results),
        "preview": results[:5],
        "message": "Predictions saved"
    }
