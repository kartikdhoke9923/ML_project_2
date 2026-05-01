"""
v1 computed card_amt_mean from the FULL training set (including the current
transaction). At inference time this is fine, but if you ever re-run feature
engineering for retraining you'd be leaking the target row into its own stats.
v2 lookups are built from training data ONLY (precompute_history.py), and
new transactions increment the counts AFTER prediction — never before.

Usage
─────
  from feature_store import FeatureStore
  fs = FeatureStore()                      # loads precomputed lookups once
  feature_df = fs.build(txn_dict)          # O(1) per call
"""

import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

MODEL_DIR = Path("models")

# ── Risky email domains (must match training) ─────────────────────────────────
RISKY_DOMAINS = {
    "anonymous.com", "guerrillamail.com", "mailnull.com",
    "suremail.info", "spambog.com", "trashmail.com"
}

SECONDS_PER_DAY = 86_400


class FeatureStore:
    """
    Stateless feature builder backed by precomputed lookup dicts.
    Thread-safe: no mutable state after __init__.
    """

    def __init__(self):
        """Load precomputed lookup tables once at startup."""
        self.c_lookups    = joblib.load(MODEL_DIR / "c_lookups.pkl")
        self.d_lookups    = joblib.load(MODEL_DIR / "d_lookups.pkl")
        self.feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")

        # Card-level amount stats (for zscore / ratio features)
        # Precomputed from training: {card1: {mean, std, count}}
        self._build_card_amount_stats()
        print("FeatureStore ready — all lookups loaded.")

    def _build_card_amount_stats(self):
        """
        Build card → {mean, std, count} dict from the transaction history.
        This exists separately from C-lookups because amount stats need
        three values per key (mean + std + count).
        """
        hist_path = MODEL_DIR / "transaction_history.pkl"
        if not hist_path.exists():
            self.card_amt_stats = {}
            return

        history = joblib.load(hist_path)
        stats = (
            history.groupby("card1")["TransactionAmt"]
            .agg(mean="mean", std="std", count="count")
            .fillna({"std": 0})
        )
        self.card_amt_stats = stats.to_dict(orient="index")
        # {card1: {"mean": ..., "std": ..., "count": ...}}

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def build(self, txn: dict) -> pd.DataFrame:
        """
        Build a single-row feature DataFrame from a raw transaction dict.
        All features computed via O(1) lookups — no DataFrame scanning.

        Args:
            txn: Raw transaction fields (matches TransactionInput schema)

        Returns:
            pd.DataFrame with exactly one row and columns = feature_cols
        """
        row = {}
        dt  = txn.get("TransactionDT") or int(time.time())
        amt = txn.get("TransactionAmt", 0)

        # 1. Time features
        row.update(self._time_features(dt))

        # 2. Amount features
        row.update(self._amount_features(amt))

        # 3. Card behaviour (card-level amount stats)
        card = txn.get("card1")
        row.update(self._card_stats_features(card, amt, dt))

        # 4. Email features
        p_email = (txn.get("P_emaildomain") or "unknown").lower()
        r_email = (txn.get("R_emaildomain") or "unknown").lower()
        row.update(self._email_features(p_email, r_email))

        # 5. Device features
        row.update(self._device_features(
            card,
            txn.get("DeviceType") or "unknown",
            txn.get("DeviceInfo") or "unknown",
        ))

        # 6. Address features
        a1 = txn.get("addr1") or -1
        a2 = txn.get("addr2") or -1
        row["addr1"]        = a1
        row["addr2"]        = a2
        row["addr_mismatch"] = int(a1 != a2)

        # 7. Pass-through card fields
        row["card1"] = card
        row["card2"] = txn.get("card2") or -1
        row["card3"] = txn.get("card3") or -1

        # 8. Categorical pass-throughs
        row["ProductCD"]    = txn.get("ProductCD") or "unknown"
        row["card4"]        = txn.get("card4")     or "unknown"
        row["card6"]        = txn.get("card6")     or "unknown"
        row["P_emaildomain"] = p_email
        row["R_emaildomain"] = r_email
        row["DeviceType"]   = txn.get("DeviceType") or "unknown"
        row["DeviceInfo"]   = txn.get("DeviceInfo") or "unknown"

        # 9. M-columns: pass through from payment processor
        for i in range(1, 10):
            row[f"M{i}"] = txn.get(f"M{i}") or "unknown"

        # 10. C-columns: O(1) lookup
        row.update(self._c_features(card, a1, p_email, amt))

        # 11. D-columns: O(1) lookup + arithmetic
        row.update(self._d_features(card, a1, p_email,
                                    txn.get("DeviceInfo") or "unknown",
                                    dt))

        # Assemble into DataFrame, fill any missing cols, enforce column order
        df_row = pd.DataFrame([row])
        for col in self.feature_cols:
            if col not in df_row.columns:
                df_row[col] = -1

        return df_row[self.feature_cols]

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE FEATURE BUILDERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _time_features(dt: int) -> dict:
        hour = (dt // 3600) % 24
        dow  = (dt // (3600 * 24)) % 7
        return {
            "hour_of_day":       hour,
            "day_of_week":       dow,
            "is_weekend":        int(dow >= 5),
            "is_night":          int(hour >= 22 or hour <= 5),
            "is_business_hours": int(9 <= hour <= 17),
        }

    @staticmethod
    def _amount_features(amt: float) -> dict:
        return {
            "amt_log":          np.log1p(amt),
            "is_round_amount":  int(amt % 1 == 0),
            "is_small_amount":  int(amt < 10),
            "is_large_amount":  int(amt > 1000),
        }

    def _card_stats_features(self, card, amt: float, dt: int) -> dict:
        """Card-level amount stats + velocity features — all O(1)."""
        stats = self.card_amt_stats.get(card)

        if stats:
            card_mean  = stats["mean"]
            card_std   = stats["std"] if not np.isnan(stats["std"]) else 0
            card_count = stats["count"]
        else:
            # New card — no history
            card_mean  = amt
            card_std   = 0
            card_count = 0

        # Velocity from D-lookups (reuse timestamp anchor)
        card_ts = self.d_lookups["card_ts"].get(card, {})
        last_dt = card_ts.get("last_dt", None)

        if last_dt is not None:
            time_diff = dt - last_dt
            is_rapid  = int(0 < time_diff < 600)
        else:
            time_diff = -1
            is_rapid  = 0

        return {
            "card_amt_mean":       card_mean,
            "card_amt_std":        card_std,
            "card_txn_count":      card_count,
            "amt_zscore_card":     (amt - card_mean) / (card_std + 1),
            "amt_ratio_card":      amt / (card_mean + 1),
            "card_txn_rank":       card_count + 1,
            "card_is_first_txn":   int(card_count == 0),
            "time_since_last_txn": time_diff,
            "is_rapid_txn":        is_rapid,
        }

    @staticmethod
    def _email_features(p_email: str, r_email: str) -> dict:
        return {
            "email_domains_match":   int(p_email == r_email),
            "purchaser_email_risky": int(p_email in RISKY_DOMAINS),
            "recipient_email_risky": int(r_email in RISKY_DOMAINS),
            "P_email_tld":           p_email.split(".")[-1] if "." in p_email else p_email,
            "R_email_tld":           r_email.split(".")[-1] if "." in r_email else r_email,
        }

    def _device_features(self, card, device_type: str, device_info: str) -> dict:
        is_mobile = int(device_type == "mobile")

        # "Seen before" = device appeared ≥2 times for this card in training
        # We approximate by checking if device appears in per-card device nunique
        # A more precise approach needs a set-per-card, stored in device_sets.pkl
        c5_val = self.c_lookups["C5"].get(card, 1)  # n distinct devices for card
        # If the card has only 1 distinct device in history, we conservatively
        # say it was seen (True). If 0 history, mark as unseen.
        card_ts = self.d_lookups["card_ts"].get(card, {})
        has_history = card_ts.get("count", 0) > 0
        device_seen = int(has_history)  # if card has any history, device likely seen

        return {
            "is_mobile":         is_mobile,
            "device_seen_before": device_seen,
        }

    def _c_features(self, card, addr1, email: str, amt: float) -> dict:
        """
        C1–C14 via dict lookups. O(1).
        Defaults: 1 (first-seen entity), except C12 which defaults to 0.
        """
        cl = self.c_lookups
        addr_email_key = f"{addr1}|{email}"

        return {
            "C1":  cl["C1"].get(card,   1),
            "C2":  cl["C2"].get(addr1,  1),
            "C3":  cl["C3"].get(card,   1),
            "C4":  cl["C4"].get(email,  1),
            "C5":  cl["C5"].get(card,   1),
            "C6":  cl["C6"].get(email,  1),
            "C7":  cl["C7"].get(card,   1),
            "C8":  cl["C8"].get(card,   1),
            "C9":  cl["C9"].get(card,   1),
            "C10": cl["C10"].get(addr_email_key, 1),
            "C11": cl["C11"].get(card,  1),
            "C12": cl["C12"].get(card,  0),    # 0 = no high-value history
            "C13": cl["C13"].get(card,  1),
            "C14": cl["C14"].get(card,  1),
        }

    def _d_features(self, card, addr1, email: str, device: str, dt: int) -> dict:
        """
        D1–D15 time-delta features via precomputed timestamp anchors. O(1).
        All values in days. Unknown = -1 (same as training fillna default).
        """
        dl = self.d_lookups

        card_ts   = dl["card_ts"].get(card,   {})
        addr_ts   = dl["addr_ts"].get(addr1,  {})
        email_ts  = dl["email_ts"].get(email, {})
        device_ts = dl["device_ts"].get(device, {})

        def days_since_first(ts_dict):
            first = ts_dict.get("first_dt")
            if first is None:
                return -1
            return max(0.0, (dt - first) / SECONDS_PER_DAY)

        def days_since_last(ts_dict):
            last = ts_dict.get("last_dt")
            if last is None:
                return -1
            return max(0.0, (dt - last) / SECONDS_PER_DAY)

        # D4: gap between the two most recent transactions for this card
        second_last = dl["card_2nd_last"].get(card)
        last_dt     = card_ts.get("last_dt")
        if second_last is not None and last_dt is not None:
            d4 = max(0.0, (last_dt - second_last) / SECONDS_PER_DAY)
        else:
            d4 = -1

        return {
            "D1":  days_since_first(card_ts),
            "D2":  days_since_last(card_ts),
            "D3":  days_since_first(addr_ts),
            "D4":  d4,
            "D5":  days_since_first(card_ts),   # same anchor as D1
            "D6":  -1,   # exact meaning unknown
            "D7":  -1,
            "D8":  -1,
            "D9":  -1,
            "D10": days_since_first(device_ts),
            "D11": days_since_first(email_ts),
            "D12": -1,
            "D13": -1,
            "D14": -1,
            "D15": days_since_last(addr_ts),
        }


# ── Convenience function (backward compat with main.py v1 signature) ──────────
_store: Optional[FeatureStore] = None

def build_feature_vector(txn: dict, history: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Drop-in replacement for the v1 function.
    `history` and `feature_cols` args are accepted but ignored —
    the FeatureStore loads these from disk at startup.
    """
    global _store
    if _store is None:
        _store = FeatureStore()
    return _store.build(txn)