from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["clv_project"]

customers_col = db["customers"]
training_col  = db["training_log"]


def save_prediction(data: dict):
    data["created_at"] = datetime.now()
    customers_col.insert_one(data)
    return data


def get_all_customers(dataset: str):
    return list(customers_col.find(
        {"dataset": dataset}, {"_id": 0}
    ))


def get_segment_summary(dataset: str):
    pipeline = [
        {"$match": {"dataset": dataset}},
        {"$group": {
            "_id": "$segment",
            "count": {"$sum": 1},
            "avg_clv": {"$avg": "$predicted_clv"},
            "total_clv": {"$sum": "$predicted_clv"}
        }}
    ]
    return list(customers_col.aggregate(pipeline))


def get_dashboard_stats(dataset: str):
    customers = list(customers_col.find({"dataset": dataset}))
    if not customers:
        return {
            "total_customers": 0,
            "avg_clv": 0,
            "max_clv": 0,
            "total_revenue": 0
        }
    clvs = [c["predicted_clv"] for c in customers]
    return {
        "total_customers": len(customers),
        "avg_clv":         round(sum(clvs) / len(clvs), 2),
        "max_clv":         round(max(clvs), 2),
        "total_revenue":   round(sum(clvs), 2)
    }


def get_top_customers(dataset: str, limit: int = 10):
    return list(customers_col.find(
        {"dataset": dataset}, {"_id": 0}
    ).sort("predicted_clv", -1).limit(limit))