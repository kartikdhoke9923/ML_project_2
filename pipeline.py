import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


from pathlib import Path
DATA_DIR   = Path("ieee-fraud-detection")
OUTPUT_DIR = Path("baseline_clean_outputs")
MODEL_DIR  = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# importing data 
test_identity=pd.read_csv(DATA_DIR / "test_identity.csv")
test_transaction=pd.read_csv(DATA_DIR / "test_transaction.csv")
train_identity=pd.read_csv(DATA_DIR / "train_identity.csv")
train_transaction=pd.read_csv(DATA_DIR / "train_transaction.csv")

# merging data
df=train_transaction.merge(train_identity,on="TransactionID",how="left")


# Time features    : hour_of_day, day_of_week, is_weekend, is_night, is_business_hours
df["hour_of_day"]  = (df["TransactionDT"] // 3600) % 24
df["day_of_week"]  = (df["TransactionDT"] // (3600 * 24)) % 7
df["is_weekend"]   = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
df["is_night"]     = df["hour_of_day"].apply(
                        lambda x: 1 if (x >= 22 or x <= 5) else 0)
df["is_business_hours"] = df["hour_of_day"].apply(
                        lambda x: 1 if 9 <= x <= 17 else 0)

# Amount features  : amt_log, is_round_amount, is_small_amount, is_large_amount
# Fraud often involves unusual amounts — very small (testing card) or very large
df["amt_log"]           = np.log1p(df["TransactionAmt"])
df["is_round_amount"]   = (df["TransactionAmt"] % 1 == 0).astype(int)
df["is_small_amount"]   = (df["TransactionAmt"] < 10).astype(int)    # card testing
df["is_large_amount"]   = (df["TransactionAmt"] > 1000).astype(int)  # large fraud

# How does this transaction compare to what's normal for this card?
card_stats = df.groupby("card1")["TransactionAmt"].agg(
    card_amt_mean="mean",
    card_amt_std="std",
    card_txn_count="count"
).reset_index()
 
df = df.merge(card_stats, on="card1", how="left")
df["card_amt_std"]   = df["card_amt_std"].fillna(0)


# Z-score: how far is this transaction from the card's normal amount?
df["amt_zscore_card"] = (
    (df["TransactionAmt"] - df["card_amt_mean"])
    / (df["card_amt_std"] + 1)
)
# Ratio: is this transaction much bigger/smaller than usual for this card?
df["amt_ratio_card"]  = df["TransactionAmt"] / (df["card_amt_mean"] + 1)
 
df = df.sort_values("TransactionDT")


# Rolling count of transactions per card in last ~1 hour window (3600s)
# Using expanding rank as a proxy (full rolling requires more memory)
df["card_txn_rank"]   = df.groupby("card1").cumcount() + 1
df["card_is_first_txn"] = (df["card_txn_rank"] == 1).astype(int)
 
# Time since last transaction for this card
df["time_since_last_txn"] = (
    df.groupby("card1")["TransactionDT"].diff().fillna(-1)
)
df["is_rapid_txn"] = (
    (df["time_since_last_txn"] > 0) &
    (df["time_since_last_txn"] < 600)   # < 10 minutes
).astype(int)
 
print("  ✓ Velocity features: card_txn_rank, card_is_first_txn, "
      "time_since_last_txn, is_rapid_txn")

# ── 2E: Email domain features ─────────────────────────────────────────────────
# Mismatched purchase/recipient emails, free vs corporate domains
df["P_emaildomain"] = df["P_emaildomain"].fillna("unknown")
df["R_emaildomain"] = df["R_emaildomain"].fillna("unknown")
 
df["email_domains_match"] = (
    df["P_emaildomain"] == df["R_emaildomain"]
).astype(int)
 
# High-risk email domains (free/anonymous providers)
risky_domains = {"anonymous.com", "guerrillamail.com", "mailnull.com",
                 "suremail.info", "spambog.com", "trashmail.com"}
 
df["purchaser_email_risky"] = df["P_emaildomain"].apply(
    lambda x: 1 if x in risky_domains else 0)
 
df["recipient_email_risky"] = df["R_emaildomain"].apply(
    lambda x: 1 if x in risky_domains else 0)
 
# Extract TLD (top-level domain) as a risk signal
df["P_email_tld"] = df["P_emaildomain"].apply(
    lambda x: x.split(".")[-1] if "." in x else x)
df["R_email_tld"] = df["R_emaildomain"].apply(
    lambda x: x.split(".")[-1] if "." in x else x)
 
print("  ✓ Email features   : email_domains_match, purchaser_email_risky, "
      "recipient_email_risky, P_email_tld, R_email_tld")


# ── 2F: Device features (from identity file) ──────────────────────────────────
if "DeviceType" in df.columns:
    df["DeviceType"] = df["DeviceType"].fillna("unknown")
    df["is_mobile"]  = (df["DeviceType"] == "mobile").astype(int)
else:
    df["is_mobile"] = -1   # no identity data for this transaction
 
if "DeviceInfo" in df.columns:
    df["DeviceInfo"] = df["DeviceInfo"].fillna("unknown")
    # New/unusual device for this card?
    device_card_counts = df.groupby(
        ["card1", "DeviceInfo"])["TransactionID"].transform("count")
    df["device_seen_before"] = (device_card_counts > 1).astype(int)
else:
    df["device_seen_before"] = -1
 
print("  ✓ Device features  : is_mobile, device_seen_before")


# ── 2G: Address mismatch features ─────────────────────────────────────────────
# Billing vs shipping address mismatch is a classic fraud signal
if "addr1" in df.columns and "addr2" in df.columns:
    df["addr1"] = df["addr1"].fillna(-1)
    df["addr2"] = df["addr2"].fillna(-1)
    df["addr_mismatch"] = (df["addr1"] != df["addr2"]).astype(int)
else:
    df["addr_mismatch"] = -1
 
print("  ✓ Address features : addr_mismatch")
 
# ── 2H: C-columns (Vesta count features — interpretable) ───────────────────────
# These ARE interpretable unlike V-columns.
# C1-C14 are counts: "how many cards share this address/email/etc."
# High counts = shared infrastructure = fraud ring signal
c_cols = [f"C{i}" for i in range(1, 15) if f"C{i}" in df.columns]
for col in c_cols:
    df[col] = df[col].fillna(0)
 
print(f"  ✓ C-features       : {c_cols} (shared infrastructure counts)")


# ── 2I: D-columns (timedelta features — interpretable) ────────────────────────
# D1-D15 are time deltas: "days since first transaction", "days since address update"
# These directly measure account age and activity patterns
d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in df.columns]
for col in d_cols:
    df[col] = df[col].fillna(-1)
 
print(f"  ✓ D-features       : {d_cols} (time-delta signals — account age, etc.)")
 
# ── 2J: M-columns (match flags — interpretable) ───────────────────────────────
# M1-M9 are binary match flags: "does name on card match billing name?", etc.
m_cols = [f"M{i}" for i in range(1, 10) if f"M{i}" in df.columns]
for col in m_cols:
    df[col] = df[col].fillna("unknown")
 
print(f"  ✓ M-features       : {m_cols} (match flags — name, address, etc.)")
 
print(f"\n  ❌ V-columns (V1–V339): EXCLUDED — black box, not deployable")
print(f"  ❌ TransactionID, TransactionDT: EXCLUDED — identifiers, not features")


# Categorical columns that need label encoding
cat_cols = (
    ["ProductCD", "card4", "card6",
     "P_emaildomain", "R_emaildomain",
     "P_email_tld", "R_email_tld",
     "DeviceType"] +
    (["DeviceInfo"] if "DeviceInfo" in df.columns else []) +
    m_cols
)
cat_cols = [c for c in cat_cols if c in df.columns]

# Numeric columns (no encoding needed)
num_cols = (
    # Time
    ["hour_of_day", "day_of_week", "is_weekend",
     "is_night", "is_business_hours",
    # Amount
     "amt_log", "is_round_amount", "is_small_amount", "is_large_amount",
    # Card behaviour
     "card_amt_mean", "card_amt_std", "card_txn_count",
     "amt_zscore_card", "amt_ratio_card",
    # Velocity
     "card_txn_rank", "card_is_first_txn",
     "time_since_last_txn", "is_rapid_txn",
    # Email
     "email_domains_match", "purchaser_email_risky", "recipient_email_risky",
    # Device
     "is_mobile", "device_seen_before",
    # Address
     "addr_mismatch",
    # card numbers (numeric IDs)
     "card1", "card2", "card3",
    # addr
     "addr1", "addr2"
    ] +
    c_cols +
    d_cols
)
num_cols = [c for c in num_cols if c in df.columns]
 
feature_cols = num_cols + cat_cols
print(f"\nNumeric features    : {len(num_cols)}")
print(f"Categorical features: {len(cat_cols)}")
print(f"Total features      : {len(feature_cols)}")



# After feature engineering:
X = df[feature_cols]
y = df["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Then card_stats must be recomputed on X_train only


# ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ("num",
     SimpleImputer(strategy="median"),
     num_cols),

    ("cat",
     Pipeline([
         ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
         ("encoder", OrdinalEncoder(
             handle_unknown="use_encoded_value",
             unknown_value=-1
         ))
     ]),
     cat_cols)
])

# Step 2: Full Pipeline (preprocessor → SMOTE → LGB) 
# Must use ImbPipeline (not sklearn Pipeline) because SMOTE
# only runs during fit(), not transform()

lgb_pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote",        SMOTE(random_state=42, sampling_strategy=0.3)),
    ("model",        lgb.LGBMClassifier(
                        n_estimators=500,
                        max_depth=6,
                        learning_rate=0.05,
                        num_leaves=63,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1
                    ))
])


print(X_train.shape)
print(X_train.columns)


lgb_pipeline.fit(X_train, y_train)
proba = lgb_pipeline.predict_proba(X_test)[:, 1]
auc   = roc_auc_score(y_test, proba)
print(f"Pipeline LGB AUC: {auc:.4f}")

# ── Step 5: Save single pkl 
joblib.dump(lgb_pipeline, MODEL_DIR / "lgb_pipeline.pkl")
print("Saved: lgb_pipeline.pkl")