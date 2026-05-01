# IEEE-CIS Fraud Detection

A real-time fraud detection system built on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection). Predicts whether a transaction is fraudulent using a LightGBM model deployed as a REST API with a Streamlit frontend.

**Live Demo:** [ieee-cis-fraud-detection.streamlit.app](https://ieee-cis-fraud-detection.streamlit.app)  
**API Docs:** [your-render-link.onrender.com/docs](https://your-render-link.onrender.com/docs)

---

## Project Structure

```
fraud-detection/
├── notebook.ipynb           # EDA, feature engineering, model selection
├── pipeline.py              # Training pipeline (final model)
├── main.py                  # FastAPI backend
├── streamlit_app.py         # Streamlit frontend
├── feature_store.py         # Feature engineering + O(1) inference lookups
├── precompute_history.py    # Precomputes C/D lookup tables from training data
├── models/
│   ├── lgb_pipeline.pkl     # Final trained LightGBM pipeline
│   ├── c_lookups.pkl        # Precomputed C-column lookup dicts
│   ├── d_lookups.pkl        # Precomputed D-column lookup dicts
│   └── feature_cols.pkl     # Feature column order (must match training)
├── render.yaml              # Render deployment config
└── requirements.txt
```

---

## What I Built

### 1. Exploratory Data Analysis (`notebook.ipynb`)
- Analyzed 590k transactions with 434 features
- Identified class imbalance: 3.5% fraud vs 96.5% legitimate
- Investigated V-columns (V1-V339), C-columns, D-columns, M-columns
- Found transaction amount, card velocity, and email domain as key signals

### 2. Feature Engineering
Built interpretable features grouped into 8 categories:

| Group | Features | Signal |
|---|---|---|
| Time | hour_of_day, is_night, is_weekend | Fraud happens at unusual hours |
| Amount | amt_log, is_round_amount, is_large_amount | Unusual amounts signal fraud |
| Card behaviour | amt_zscore_card, amt_ratio_card | Transaction deviates from card history |
| Velocity | is_rapid_txn, card_txn_rank | Multiple transactions in short time |
| Email | email_domains_match, purchaser_email_risky | Risky or mismatched email domains |
| Device | is_mobile, device_seen_before | New device for a known card |
| Address | addr_mismatch | Billing vs shipping mismatch |
| C/D columns | C1-C14, D1-D15 | Shared infrastructure counts, account age |

**V-columns (V1-V339) deliberately excluded** — proprietary Vesta features unavailable outside their system. See limitations section.

### 3. Model Selection (`notebook.ipynb`)
Compared four models on AUC-ROC:

| Model | AUC |
|---|---|
| Logistic Regression | ~0.78 |
| Random Forest | ~0.85 |
| XGBoost | ~0.88 |
| **LightGBM** | **~0.90** |

Applied SMOTE (sampling_strategy=0.3) to handle class imbalance. LightGBM selected as final model.

### 4. Inference Pipeline (`feature_store.py`)

**Problem:** Naive approach filters 590k-row DataFrame per request — O(n), ~20ms per call.

**Solution:** Precompute all C and D features into flat Python dicts at startup. Inference becomes O(1) dict lookups — ~0.01ms per call.

```
New transaction arrives
        ↓
feature_store.build(txn)        ← O(1) lookups, no DataFrame scanning
        ↓
lgb_pipeline.predict_proba()
        ↓
{ decision, fraud_probability, risk_tier }
```

### 5. Deployment

| Component | Platform |
|---|---|
| FastAPI backend | Render |
| Streamlit frontend | Streamlit Cloud |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service status |
| POST | `/predict` | Score a single transaction |
| POST | `/predict/batch` | Score up to 100 transactions |
| GET | `/history/stats` | Transaction history summary |

**Example request:**
```bash
curl -X POST "https://your-render-link.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 9200,
    "ProductCD": "H",
    "card1": 888888,
    "card4": "mastercard",
    "card6": "credit",
    "addr1": 10,
    "addr2": 999,
    "P_emaildomain": "anonymous.com",
    "R_emaildomain": "guerrillamail.com",
    "DeviceType": "mobile",
    "DeviceInfo": "Android 4.0"
  }'
```

**Example response:**
```json
{
  "decision": "BLOCK",
  "fraud_probability": 0.76,
  "risk_tier": "HIGH",
  "card_history": {
    "is_new_card": false,
    "past_transactions": 142
  },
  "latency_ms": 12.4
}
```

---

## How to Run Locally

```bash
# Clone repo
git clone https://github.com/kartikdhoke9923/ML_project_2
cd ML_project_2

# Install dependencies
pip install -r requirements.txt

# Precompute lookup tables (run once after training)
python precompute_history.py

# Start FastAPI backend
uvicorn main:app --host 0.0.0.0 --port 8000

# In a separate terminal, start Streamlit
streamlit run streamlit_app.py
```

---

## Known Limitations

**1. V-columns excluded**
V1-V339 are the strongest predictors in this dataset but are proprietary Vesta features unavailable at inference time outside their payment system. Excluding them reduces recall but makes the model actually deployable. A production integration would require a Vesta API to access these features.

**2. Cold start problem**
New cards with no transaction history score lower because C and D features default to base values. This is a known challenge in production fraud detection. A production fix would add a rule-based layer flagging all transactions above a threshold from new cards with risky email domains.

**3. Render free tier cold start**
The API spins down after 15 minutes of inactivity. First request after inactivity takes 30-60 seconds. Normal behavior on the free tier.

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — provided by Vesta Corporation via Kaggle.

- 590,540 training transactions
- 434 features
- 3.5% fraud rate
- Features intentionally anonymized by Vesta

---

## Tech Stack

`Python` `LightGBM` `scikit-learn` `imbalanced-learn` `FastAPI` `Streamlit` `Pydantic` `joblib` `pandas` `numpy`