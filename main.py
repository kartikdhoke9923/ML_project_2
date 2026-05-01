import time
import joblib
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from feature_store import MODEL_DIR, FeatureStore

# ── Load model and feature store once at startup ──────────────────────────────
feature_store = FeatureStore()
model         = joblib.load(MODEL_DIR / "lgb_pipeline.pkl")

app = FastAPI(title="Fraud Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic input model ───────────────────────────────────────────────────────
class UserInput(BaseModel):
    TransactionAmt: float           = Field(..., gt=0)
    ProductCD:      str             = Field(...)
    TransactionDT:  Optional[int]   = Field(None)
    card1:          int             = Field(...)
    card2:          Optional[float] = None
    card3:          Optional[float] = None
    card4:          Optional[str]   = None
    card6:          Optional[str]   = None
    addr1:          Optional[float] = None
    addr2:          Optional[float] = None
    P_emaildomain:  Optional[str]   = None
    R_emaildomain:  Optional[str]   = None
    DeviceType:     Optional[str]   = None
    DeviceInfo:     Optional[str]   = None
    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None
    M4: Optional[str] = None
    M5: Optional[str] = None
    M6: Optional[str] = None
    M7: Optional[str] = None
    M8: Optional[str] = None
    M9: Optional[str] = None


# ── Shared scoring logic ───────────────────────────────────────────────────────
def score_transaction(txn: dict) -> dict:
    """
    Core scoring function used by both single and batch endpoints.
    Returns decision, probability, risk tier, card context.
    """
    start = time.time()

    if not txn.get("TransactionDT"):
        txn["TransactionDT"] = int(time.time())

    input_df = feature_store.build(txn)

    prob  = float(model.predict_proba(input_df)[:, 1][0])
    label = int(model.predict(input_df)[0])

    if prob < 0.3:
        risk = "LOW"
    elif prob < 0.6:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    card       = txn.get("card1")
    card_stats = feature_store.card_amt_stats.get(card)
    past_txns  = int(card_stats["count"]) if card_stats else 0

    return {
        "decision":          "BLOCK" if label == 1 else "APPROVE",
        "fraud_probability": round(prob, 4),
        "risk_tier":         risk,
        "card_history": {
            "is_new_card":       past_txns == 0,
            "past_transactions": past_txns,
        },
        "latency_ms": round((time.time() - start) * 1000, 2)
    }


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "history_rows": sum(
            v.get("count", 0)
            for v in feature_store.d_lookups.get("card_ts", {}).values()
        ),
        "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }


# ── /predict ──────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: UserInput):
    try:
        return score_transaction(data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /predict/batch ────────────────────────────────────────────────────────────
@app.post("/predict/batch")
def predict_batch(transactions: List[UserInput]):
    """Score up to 100 transactions. One bad row does not fail the whole batch."""
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Batch limit is 100 transactions.")

    results = []
    for txn in transactions:
        try:
            result = score_transaction(txn.model_dump())
        except Exception as e:
            result = {
                "decision":          "ERROR",
                "fraud_probability": -1,
                "risk_tier":         "UNKNOWN",
                "card_history":      {},
                "latency_ms":        0,
                "error":             str(e)
            }
        results.append(result)

    return {
        "count":       len(results),
        "approved":    sum(1 for r in results if r["decision"] == "APPROVE"),
        "blocked":     sum(1 for r in results if r["decision"] == "BLOCK"),
        "errors":      sum(1 for r in results if r["decision"] == "ERROR"),
        "predictions": results
    }


# ── /history/stats ────────────────────────────────────────────────────────────
@app.get("/history/stats")
def history_stats():
    """Stats about the precomputed transaction history used for C/D features."""
    card_ts = feature_store.d_lookups.get("card_ts", {})

    if not card_ts:
        return {"message": "No history loaded. Run precompute_history.py first."}

    all_first = [v["first_dt"] for v in card_ts.values() if "first_dt" in v]
    all_last  = [v["last_dt"]  for v in card_ts.values() if "last_dt"  in v]

    date_range_days = None
    if all_first and all_last:
        date_range_days = round((max(all_last) - min(all_first)) / 86400, 1)

    total_txns = sum(v.get("count", 0) for v in card_ts.values())

    return {
        "total_transactions": total_txns,
        "unique_cards":       len(card_ts),
        "unique_addresses":   len(feature_store.d_lookups.get("addr_ts", {})),
        "unique_emails":      len(feature_store.d_lookups.get("email_ts", {})),
        "unique_devices":     len(feature_store.d_lookups.get("device_ts", {})),
        "date_range_days":    date_range_days,
        "fraud_rate_pct":     None  # not stored in lookups, load history.pkl if needed
    }