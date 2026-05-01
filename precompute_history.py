"""
precompute_history.py
─────────────────────
Run ONCE after training to build fast lookup tables for C and D features.

Why this exists
───────────────
The naive approach (filter history DataFrame per request) is O(n) per call.
With 590k training rows, that's ~20ms of pure Python filtering at inference time.

Instead, we precompute groupby aggregations into flat Python dicts.
Inference becomes O(1) dict lookups — ~0.01ms.

Outputs (saved to models/)
──────────────────────────
  c_lookups.pkl       → dict of dicts for C1–C14
  d_lookups.pkl       → dict of dicts for D1–D15
  feature_cols.pkl    → ordered list of feature columns (for pipeline alignment)
  transaction_history.pkl → trimmed history DataFrame (for fallback / velocity features)

Usage
─────
  python precompute_history.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


DATA_DIR  = Path("ieee-fraud-detection")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ── Load & merge training data ────────────────────────────────────────────────
print("Loading training data...")
train_txn = pd.read_csv(DATA_DIR / "train_transaction.csv")
train_id  = pd.read_csv(DATA_DIR / "train_identity.csv")
df = train_txn.merge(train_id, on="TransactionID", how="left")

print(f"  {len(df):,} rows loaded")


# ── Fill key columns ──────────────────────────────────────────────────────────
df["P_emaildomain"] = df["P_emaildomain"].fillna("unknown")
df["R_emaildomain"] = df["R_emaildomain"].fillna("unknown")
df["DeviceInfo"]    = df["DeviceInfo"].fillna("unknown")    if "DeviceInfo"    in df.columns else "unknown"
df["addr1"]         = df["addr1"].fillna(-1)
df["addr2"]         = df["addr2"].fillna(-1)


# ═════════════════════════════════════════════════════════════════════════════
# C-COLUMN LOOKUP TABLES
# Each table maps a key (card1 / addr1 / email / ...) → precomputed count.
# At inference: c_lookup["C1"].get(card1, default)
# ═════════════════════════════════════════════════════════════════════════════
print("\nBuilding C-column lookups...")

c_lookups = {}

# C1: distinct billing addresses per card
c_lookups["C1"] = df.groupby("card1")["addr1"].nunique().to_dict()

# C2: distinct cards per billing address
c_lookups["C2"] = df.groupby("addr1")["card1"].nunique().to_dict()

# C3: total transaction count per card
c_lookups["C3"] = df.groupby("card1")["TransactionID"].count().to_dict()

# C4: distinct billing addresses per purchaser email
c_lookups["C4"] = df.groupby("P_emaildomain")["addr1"].nunique().to_dict()

# C5: distinct devices per card
if "DeviceInfo" in df.columns:
    c_lookups["C5"] = df.groupby("card1")["DeviceInfo"].nunique().to_dict()
else:
    c_lookups["C5"] = {}

# C6: distinct cards per purchaser email
c_lookups["C6"] = df.groupby("P_emaildomain")["card1"].nunique().to_dict()

# C7: distinct countries (addr2) per card
c_lookups["C7"] = df.groupby("card1")["addr2"].nunique().to_dict()

# C8: distinct emails per card
c_lookups["C8"] = df.groupby("card1")["P_emaildomain"].nunique().to_dict()

# C9: transaction count variant (same key as C3, kept separate for pipeline compat)
c_lookups["C9"] = c_lookups["C3"].copy()

# C10: distinct cards sharing same (addr1 + email) combo
#   Build composite key, then count distinct card1 per combo
df["_addr_email_key"] = df["addr1"].astype(str) + "|" + df["P_emaildomain"]
c_lookups["C10"] = df.groupby("_addr_email_key")["card1"].nunique().to_dict()
df.drop(columns=["_addr_email_key"], inplace=True)

# C11: cumulative transaction count (same as C3 — kept for pipeline compat)
c_lookups["C11"] = c_lookups["C3"].copy()

# C12: count of high-value transactions per card (amt > 100)
c_lookups["C12"] = (
    df[df["TransactionAmt"] > 100]
    .groupby("card1")["TransactionID"].count()
    .to_dict()
)

# C13: distinct countries per card (same as C7)
c_lookups["C13"] = c_lookups["C7"].copy()

# C14: distinct product codes per card
c_lookups["C14"] = df.groupby("card1")["ProductCD"].nunique().to_dict()

print(f"  ✓ C-lookups built for C1–C14")
for k, v in c_lookups.items():
    print(f"    {k}: {len(v):,} unique keys")


# ═════════════════════════════════════════════════════════════════════════════
# D-COLUMN LOOKUP TABLES
# D-features are time deltas (days). We precompute first/last timestamps
# per card / addr / email / device so inference can compute days on the fly.
# ═════════════════════════════════════════════════════════════════════════════
print("\nBuilding D-column lookups...")

d_lookups = {}

# Per-card: first and last TransactionDT, plus second-last for gap (D4)
card_ts = df.groupby("card1")["TransactionDT"].agg(
    first_dt="min",
    last_dt="max",
    count="count"
).to_dict(orient="index")

# For D4 (gap between last two transactions) we need the second-last timestamp
# Sort per card and grab second-to-last
df_sorted = df.sort_values("TransactionDT")
second_last = (
    df_sorted.groupby("card1")["TransactionDT"]
    .apply(lambda s: s.iloc[-2] if len(s) >= 2 else None)
    .dropna()
    .to_dict()
)

d_lookups["card_ts"]     = card_ts        # {card1: {first_dt, last_dt, count}}
d_lookups["card_2nd_last"] = second_last  # {card1: second_last_dt}

# Per-address: first and last TransactionDT
addr_ts = df.groupby("addr1")["TransactionDT"].agg(
    first_dt="min",
    last_dt="max"
).to_dict(orient="index")
d_lookups["addr_ts"] = addr_ts

# Per-email: first TransactionDT (D11)
email_ts = df.groupby("P_emaildomain")["TransactionDT"].agg(
    first_dt="min"
).to_dict(orient="index")
d_lookups["email_ts"] = email_ts

# Per-device: first TransactionDT (D10)
if "DeviceInfo" in df.columns:
    device_ts = df.groupby("DeviceInfo")["TransactionDT"].agg(
        first_dt="min"
    ).to_dict(orient="index")
    d_lookups["device_ts"] = device_ts
else:
    d_lookups["device_ts"] = {}

print(f"  ✓ D-lookups built (card/addr/email/device timestamp anchors)")
print(f"    card_ts:    {len(d_lookups['card_ts']):,} cards")
print(f"    addr_ts:    {len(d_lookups['addr_ts']):,} addresses")
print(f"    email_ts:   {len(d_lookups['email_ts']):,} email domains")
print(f"    device_ts:  {len(d_lookups['device_ts']):,} devices")


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMN LIST
# Must match the order used during model training exactly.
# ═════════════════════════════════════════════════════════════════════════════
m_cols = [f"M{i}" for i in range(1, 10)]
c_cols = [f"C{i}" for i in range(1, 15)]
d_cols = [f"D{i}" for i in range(1, 16)]

num_cols = [
    "hour_of_day", "day_of_week", "is_weekend", "is_night", "is_business_hours",
    "amt_log", "is_round_amount", "is_small_amount", "is_large_amount",
    "card_amt_mean", "card_amt_std", "card_txn_count",
    "amt_zscore_card", "amt_ratio_card",
    "card_txn_rank", "card_is_first_txn",
    "time_since_last_txn", "is_rapid_txn",
    "email_domains_match", "purchaser_email_risky", "recipient_email_risky",
    "is_mobile", "device_seen_before",
    "addr_mismatch",
    "card1", "card2", "card3",
    "addr1", "addr2",
] + c_cols + d_cols

cat_cols = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "P_email_tld", "R_email_tld",
    "DeviceType", "DeviceInfo",
] + m_cols

feature_cols = num_cols + cat_cols


# ═════════════════════════════════════════════════════════════════════════════
# SAVE PRECOMPUTED ARTIFACTS
# ═════════════════════════════════════════════════════════════════════════════
print("\nSaving precomputed artifacts...")

joblib.dump(c_lookups,    MODEL_DIR / "c_lookups.pkl")
joblib.dump(d_lookups,    MODEL_DIR / "d_lookups.pkl")
joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

# Trimmed history: only columns needed at inference (saves ~80% memory vs full df)
history_cols = [
    "TransactionID", "TransactionDT", "TransactionAmt",
    "card1", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "DeviceInfo", "ProductCD",
    "isFraud"
]
history_cols = [c for c in history_cols if c in df.columns]
joblib.dump(df[history_cols], MODEL_DIR / "transaction_history.pkl")

print(f"  ✓ c_lookups.pkl      ({MODEL_DIR / 'c_lookups.pkl'})")
print(f"  ✓ d_lookups.pkl      ({MODEL_DIR / 'd_lookups.pkl'})")
print(f"  ✓ feature_cols.pkl   ({MODEL_DIR / 'feature_cols.pkl'})")
print(f"  ✓ transaction_history.pkl  ({len(df[history_cols]):,} rows, {len(history_cols)} columns)")
print(f"\nPrecomputation complete. Run `uvicorn main:app` to start the API.")