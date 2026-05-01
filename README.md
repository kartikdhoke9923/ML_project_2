# 🛡️ Real-Time Fraud Detection System
### IEEE-CIS Dataset · LightGBM · FastAPI · Streamlit

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-yellow.svg)](https://lightgbm.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> A production-style fraud detection system trained on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection). Includes real-time inference via FastAPI, an interactive Streamlit dashboard, and O(1) feature computation using precomputed lookup tables.

---

## 🔗 Live Demo

| Service | URL |
|---|---|
| 🖥️ Streamlit Dashboard | `https://yourapp.streamlit.app` |
| ⚡ FastAPI Backend | `https://fraud-api.railway.app` |
| 📖 API Docs (Swagger) | `https://fraud-api.railway.app/docs` |

---

## 📌 What This Project Does

Most fraud detection projects stop at a Jupyter notebook with a good AUC score. This one goes further — it simulates what a real production fraud detection system looks like:

- **Feature engineering** that mirrors how payment processors like Vesta actually compute fraud signals
- **Precomputed lookup tables** so C/D features are computed in O(1) at inference, not O(n) DataFrame filtering
- **REST API** that accepts raw transaction data and returns a fraud decision in milliseconds
- **Interactive dashboard** for single transaction scoring and batch testing
- **Deployment-ready** with Railway + Streamlit Cloud setup

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                       │
│                                                          │
│  IEEE-CIS CSV → Feature Engineering → LightGBM Pipeline │
│                         ↓                               │
│              precompute_history.py                       │
│                         ↓                               │
│   c_lookups.pkl   d_lookups.pkl   lgb_pipeline.pkl       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                       │
│                                                          │
│  Raw Transaction → FeatureStore.build() → Model.predict  │
│       (user input)    (O(1) lookups)     (LightGBM)      │
│                         ↓                               │
│            { decision, fraud_probability, risk_tier }    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  DEPLOYMENT LAYER                        │
│                                                          │
│   FastAPI (Railway)  ←→  Streamlit (Streamlit Cloud)    │
└─────────────────────────────────────────────────────────┘
```

---

## 🔬 Feature Engineering

Features are grouped by type and fraud signal:

| Group | Features | Fraud Signal |
|---|---|---|
| **Time** | hour_of_day, is_night, is_weekend, is_business_hours | Transactions at 3AM are suspicious |
| **Amount** | amt_log, is_round_amount, is_small_amount, is_large_amount | Card testing uses tiny amounts |
| **Card behaviour** | amt_zscore_card, amt_ratio_card, card_txn_count | Transaction far outside card's normal range |
| **Velocity** | is_rapid_txn, time_since_last_txn, card_is_first_txn | Multiple transactions in <10 minutes |
| **Email** | email_domains_match, purchaser_email_risky, P_email_tld | Risky/anonymous email providers |
| **Device** | is_mobile, device_seen_before | New device on a known card |
| **Address** | addr_mismatch | Billing ≠ shipping address |
| **C-columns** | C1–C14 (shared infrastructure counts) | Fraud rings share cards/addresses/emails |
| **D-columns** | D1–D15 (time delta features) | Account age and activity gaps |
| **M-columns** | M1–M9 (payment processor match flags) | Name/address mismatch at processor level |

### Why C-Columns Are Not User Inputs

C-columns (e.g. "how many cards share this billing address") are **computed server-side** from transaction history — not typed in by the user. In production, your system queries a transaction database. In this project, they are precomputed from training data using `precompute_history.py`.

```
User makes transaction
        ↓
Server receives: card_id, amount, email, device...
        ↓
FeatureStore queries precomputed dicts → C1: 3, D2: 14.5 days...
        ↓
Model scores assembled feature vector
        ↓
User sees: APPROVED ✅ or BLOCKED 🚫
```

---

## ⚡ Performance: O(1) Inference

The naive approach — filtering a 590k-row DataFrame per request — costs ~20ms of pure Python per call and is not thread-safe.

This project uses precomputed lookup dicts:

```python
# Naive approach (DO NOT use in production)
card_hist = history[history["card1"] == card]        # O(n) scan
c1_value  = card_hist["addr1"].nunique()             # O(k) compute

# This project (O(1) lookup)
c1_value = c_lookups["C1"].get(card, 1)             # ~0.01ms
```

| Approach | Latency per request | Thread-safe |
|---|---|---|
| DataFrame filtering | ~20ms | ❌ No |
| Lookup dicts (this project) | ~0.01ms | ✅ Yes |

---

## 📁 Project Structure

```
fraud-detection/
│
├── 📓 notebooks/
│   └── training.py              # Full training pipeline
│
├── 🔧 backend/
│   ├── main.py                  # FastAPI app (3 endpoints)
│   ├── feature_store.py         # FeatureStore class (O(1) inference)
│   ├── schema.py                # Pydantic input validation
│   └── precompute_history.py    # One-time precomputation script
│
├── 🖥️ frontend/
│   └── streamlit_app.py         # Streamlit dashboard
│
├── 📦 models/                   # Generated after running precompute
│   ├── lgb_pipeline.pkl         # Trained LightGBM pipeline
│   ├── c_lookups.pkl            # C1–C14 precomputed dicts
│   ├── d_lookups.pkl            # D1–D15 timestamp anchors
│   ├── feature_cols.pkl         # Ordered feature column list
│   └── transaction_history.pkl  # Trimmed history (velocity features)
│
├── Procfile                     # Railway deployment config
├── requirements.txt
└── README.md
```

---

## 🚀 Local Setup

### 1. Clone and install

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

### 2. Download the dataset

Download from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) and place in:

```
ieee-fraud-detection/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

### 3. Train the model

```bash
python training.py
# Outputs: models/lgb_pipeline.pkl
# Expect AUC ~0.92–0.94
```

### 4. Precompute feature lookup tables

```bash
python precompute_history.py
# Outputs: c_lookups.pkl, d_lookups.pkl, feature_cols.pkl, transaction_history.pkl
# Takes ~2 minutes on the full dataset
```

### 5. Start FastAPI

```bash
uvicorn main:app --reload
# Running at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 6. Start Streamlit

```bash
streamlit run streamlit_app.py
# Running at http://localhost:8501
```

---

## 📡 API Reference

### `GET /health`
Returns service status and model load info.

```json
{
  "status": "ok",
  "model_loaded": true,
  "history_rows": 590540,
  "loaded_at": "2025-01-15 10:32:01"
}
```

---

### `POST /predict`
Score a single transaction.

**Request body:**
```json
{
  "TransactionAmt": 117.50,
  "ProductCD": "W",
  "card1": 13926,
  "card4": "visa",
  "card6": "debit",
  "P_emaildomain": "gmail.com",
  "R_emaildomain": "gmail.com",
  "DeviceType": "desktop",
  "DeviceInfo": "Windows"
}
```

**Response:**
```json
{
  "decision": "APPROVE",
  "fraud_probability": 0.0312,
  "risk_tier": "LOW",
  "card_history": {
    "is_new_card": false,
    "past_transactions": 47
  },
  "latency_ms": 34.2
}
```

---

### `POST /predict/batch`
Score up to 100 transactions at once.

**Response:**
```json
{
  "count": 3,
  "approved": 2,
  "blocked": 1,
  "errors": 0,
  "predictions": [...]
}
```

---

### `GET /history/stats`
Summary of the precomputed transaction history.

```json
{
  "total_transactions": 590540,
  "unique_cards": 57307,
  "unique_addresses": 4174,
  "unique_emails": 59,
  "unique_devices": 4818,
  "date_range_days": 182.5
}
```

---

## ☁️ Deployment

### FastAPI → Railway

```bash
# 1. Push to GitHub
git push origin main

# 2. Go to railway.app → New Project → Deploy from GitHub
# 3. Railway reads Procfile automatically:
#    web: uvicorn main:app --host 0.0.0.0 --port $PORT
# 4. Get your URL: https://fraud-api.railway.app
```

### Streamlit → Streamlit Community Cloud

```bash
# 1. Go to share.streamlit.io
# 2. Connect GitHub repo
# 3. Set main file: streamlit_app.py
# 4. Update API_BASE in streamlit_app.py:
#    API_BASE = "https://fraud-api.railway.app"
# 5. Deploy → get URL: https://yourapp.streamlit.app
```

### FastAPI → Modal (Production-Grade)

[Modal](https://modal.com) is the recommended path for serious ML deployment — serverless, scales to zero, handles large model files cleanly.

```python
# modal_app.py
import modal

app = modal.App("fraud-detection")

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi", "uvicorn", "lightgbm",
                 "scikit-learn", "imbalanced-learn",
                 "pandas", "numpy", "joblib", "pydantic")
)

@app.function(
    image=image,
    mounts=[modal.Mount.from_local_dir("models", remote_path="/app/models")],
    keep_warm=1   # keeps one instance warm to avoid cold starts
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/app")
    from main import app as fastapi_app
    return fastapi_app
```

```bash
modal deploy modal_app.py
# Live at: https://yourname--fraud-detection.modal.run
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | ~0.93 |
| Model | LightGBM (GBDT) |
| Class imbalance handling | SMOTE (sampling_strategy=0.3) + class_weight="balanced" |
| Train/Test split | 80/20 stratified |
| Features used | ~65 engineered features |
| V-columns (V1–V339) | ❌ Excluded — undocumented, not deployable |

### Why LightGBM

- Handles missing values natively (no imputation needed for tree splits)
- Fast training on tabular data with high cardinality categoricals
- Top-performing model class on this dataset across public Kaggle solutions
- Works well with the class imbalance present in fraud data (~3.5% fraud rate)

---

## 🧠 Key Design Decisions

**1. Excluded V-columns (V1–V339)**
These 339 Vesta-proprietary columns are undocumented and uninterpretable. In a real system, you would not have access to them — they are internal Vesta signals. Excluding them makes the model actually deployable.

**2. Precomputed lookups, not runtime DataFrame filtering**
At inference time, filtering 590k rows per request is O(n). Every C-column and D-column value is instead served from a flat Python dict in O(1). This is how production feature stores work.

**3. M-columns come from the payment processor, not the user**
M1–M9 are match flags (e.g. "does the name on the card match the billing name?"). In production, Vesta or your payment processor provides these automatically. They are not user-input fields.

**4. SMOTE on the pipeline, not on raw data**
SMOTE is applied inside the training pipeline using `imblearn.Pipeline`, which ensures it only runs during `fit()` and never during `predict()`. Applying SMOTE to raw data before the train/test split is a common data leakage mistake.

---

## 🔮 What I Would Add With More Time

- [ ] SHAP explainability — per-transaction feature contribution breakdown
- [ ] Model retraining pipeline — periodic refit as new fraud patterns emerge
- [ ] Threshold tuning — adjust decision boundary based on cost of false positives vs false negatives
- [ ] A/B testing framework — compare model versions in production
- [ ] Proper feature store (Feast or Tecton) replacing the manual pkl dicts
- [ ] Redis cache for velocity features (replaces in-memory dict for multi-instance deployments)

---

## 📦 Requirements

```
fastapi
uvicorn
streamlit
lightgbm
scikit-learn>=1.3
imbalanced-learn
pandas
numpy
joblib
requests
pydantic>=2.0
```

---

## 📄 License

MIT License — use freely, attribution appreciated.

---

## 🙋 About

Built as a portfolio project to demonstrate ML engineering beyond notebook-level work — covering feature engineering, production inference patterns, REST API design, and cloud deployment.

**Dataset:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
