"""
streamlit_app.py — Fraud Detection Dashboard

"""

# adding code for waking up api automatically when streamlit app starts, since render.com free dynos sleep after inactivity
API_BASE = "https://fraud-detection-api-ry7g.onrender.com/"
import threading
import requests
import time

def keep_alive():
    while True:
        try:
            requests.get(f"{API_BASE}/health", timeout=5)
        except:
            pass
        time.sleep(840)  # ping every 14 minutes

# Start in background when Streamlit loads
threading.Thread(target=keep_alive, daemon=True).start()

import streamlit as st
import requests
import json
import pandas as pd
import time
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────


st.set_page_config(
    page_title="Fraud Detection",
    page_icon="",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach API at localhost:8000. Is `uvicorn main:app` running?"
    except Exception as e:
        return None, str(e)

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach API."
    except Exception as e:
        return None, str(e)

def risk_color(tier: str) -> str:
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(tier, "⚪")

def decision_color(decision: str) -> str:
    return "🟢 APPROVE" if decision == "APPROVE" else "🔴 BLOCK"


# ── Sidebar: API status ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("Fraud Detector")
    st.caption("IEEE-CIS LightGBM Model")

    health, err = api_get("/health")
    if health:
        st.success("API Online")
        st.caption(f"Model loaded: {health.get('loaded_at', 'unknown')}")
        st.caption(f"History rows: {health.get('history_rows', 0):,}")
    else:
        st.error(f"API Offline\n{err}")

    st.divider()
    page = st.radio("Navigate", ["Single Prediction", "Batch Test", "History Stats"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
if page == "Single Prediction":
    st.title("Single Transaction Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Details")
        amt     = st.number_input("Transaction Amount ($)", min_value=0.01, value=9200.0, step=0.01)
        product = st.selectbox("Product Code", ["W", "H", "C", "S", "R"], index=0)
        card1   = st.number_input("Card ID (card1)", min_value=1, value=13926, step=1)

        st.subheader("Card Info")
        c2, c3 = st.columns(2)
        with c2:
            card2 = st.number_input("card2", value=100.0, step=1.0)
            card3 = st.number_input("card3", value=150.0, step=1.0)
        with c3:
            card4 = st.selectbox("Network", ["visa", "mastercard", "discover", "american express"])
            card6 = st.selectbox("Type", ["debit", "credit"])

    with col2:
        st.subheader("Address & Email")
        a1 = st.number_input("addr1 (billing zip)", value=10.0, step=1.0)
        a2 = st.number_input("addr2 (country code)", value=999.0, step=1.0)
        p_email = st.text_input("Purchaser email domain", value="anonymous.com")
        r_email = st.text_input("Recipient email domain", value="guerrillamail.com")

        st.subheader("Device")
        dev_type = st.selectbox("Device Type", ["desktop", "mobile", "unknown"])
        dev_info = st.text_input("Device Info", value="Android 4.0")

    with st.expander("M-columns (match flags from payment processor)"):
        mc = st.columns(3)
        m_vals = {}
        options = ["T", "F", "M0", "M1", "M2", "unknown"]
        defaults = ["F", "F", "F", "M0", "F", "F", "unknown", "unknown", "unknown"]
        for i in range(1, 10):
            col_idx = (i - 1) % 3
            with mc[col_idx]:
                m_vals[f"M{i}"] = st.selectbox(f"M{i}", options,
                                                index=options.index(defaults[i-1])
                                                if defaults[i-1] in options else 0)

    # ── Submit ────────────────────────────────────────────────────────────────
    if st.button(" Check Transaction", type="primary", use_container_width=True):
        payload = {
            "TransactionAmt": amt,
            "ProductCD": product,
            "card1": int(card1),
            "card2": card2, "card3": card3,
            "card4": card4, "card6": card6,
            "addr1": a1, "addr2": a2,
            "P_emaildomain": p_email,
            "R_emaildomain": r_email,
            "DeviceType": dev_type,
            "DeviceInfo": dev_info,
            **m_vals,
        }

        with st.spinner("Scoring..."):
            result, err = api_post("/predict", payload)

        if err:
            st.error(err)
        elif result:
            # ── Results ───────────────────────────────────────────────────────
            st.divider()
            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric("Decision", decision_color(result["decision"]))
            with r2:
                prob_pct = round(result["fraud_probability"] * 100, 2)
                st.metric("Fraud Probability", f"{prob_pct}%")
            with r3:
                tier = result["risk_tier"]
                st.metric("Risk Tier", f"{risk_color(tier)} {tier}")
            

            # Probability bar
            st.markdown("**Fraud Score**")
            bar_color = "🟢" if prob_pct < 30 else ("🟡" if prob_pct < 60 else "🔴")
            st.progress(result["fraud_probability"],
                        text=f"{bar_color} {prob_pct}% fraud probability")

            # Card context
            ch = result.get("card_history", {})
            if ch.get("is_new_card"):
                st.info(f"🆕 New card — no prior transaction history.")
            else:
                st.info(f"📋 Card has {ch.get('past_transactions', 0):,} prior transactions in history.")

            # Raw response expander
            with st.expander("Raw API response"):
                st.json(result)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch Test
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Batch Test":
    st.title("Batch Transaction Scoring")

    st.info("Paste or edit JSON below. Array of up to 100 transactions.")

    sample = [
        {"TransactionAmt": 50.0, "ProductCD": "W", "card1": 13926,
         "card4": "visa", "card6": "debit",
         "P_emaildomain": "gmail.com", "R_emaildomain": "gmail.com",
         "DeviceType": "desktop", "DeviceInfo": "Windows"},
        {"TransactionAmt": 8500.0, "ProductCD": "H", "card1": 99999,
         "card4": "mastercard", "card6": "credit",
         "P_emaildomain": "anonymous.com", "R_emaildomain": "guerrillamail.com",
         "DeviceType": "mobile", "DeviceInfo": "Android 9"},
        {"TransactionAmt": 9.99, "ProductCD": "C", "card1": 13926,
         "card4": "visa", "card6": "debit",
         "P_emaildomain": "gmail.com", "R_emaildomain": "gmail.com",
         "DeviceType": "desktop", "DeviceInfo": "Windows"},
    ]

    raw = st.text_area("Transactions (JSON array)", value=json.dumps(sample, indent=2), height=300)

    if st.button("Score Batch", type="primary", use_container_width=True):
        try:
            transactions = json.loads(raw)
            if not isinstance(transactions, list):
                st.error("Input must be a JSON array.")
            else:
                with st.spinner(f"Scoring {len(transactions)} transactions..."):
                    result, err = api_post("/predict/batch", transactions)

                if err:
                    st.error(err)
                elif result:
                    # Summary metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total",   result["count"])
                    m2.metric("✅ Approved", result["approved"])
                    m3.metric("🚫 Blocked",  result["blocked"])

                    # Results table
                    rows = []
                    for i, pred in enumerate(result["predictions"]):
                        rows.append({
                            "#":           i + 1,
                            "Decision":    pred["decision"],
                            "Fraud Prob":  f"{round(pred['fraud_probability']*100,1)}%",
                            "Risk Tier":        pred["risk_tier"],
                            "Latency ms": round(pred["latency_ms"], 1),
                        })

                    df = pd.DataFrame(rows)

                    def color_decision(val):
                        if val == "BLOCK":
                            return "background-color: #ffcccc; color: #7b0000; font-weight: bold"
                        elif val == "APPROVE":
                            return "background-color: #ccffcc; color: #1a5c1a; font-weight: bold"
                        return ""

                    st.dataframe(
                        df.style.map(color_decision, subset=["Decision"]),
                        use_container_width=True
                    )
        except json.JSONDecodeError as e:
            st.error(f"JSON parse error: {e}")



# PAGE 3 — History Stats

elif page == "History Stats":
    st.title("Transaction History Stats")

    stats, err = api_get("/history/stats")

    if err:
        st.error(err)
    elif stats:
        if "message" in stats:
            st.warning(stats["message"])
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Transactions",  f"{stats.get('total_transactions', 0):,}")
            c2.metric("Unique Cards",         f"{stats.get('unique_cards', 0):,}")
            c3.metric("History Span (days)",  stats.get("date_range_days", "N/A"))
            c4.metric("Fraud Rate",           f"{stats.get('fraud_rate_pct', 0)}%")

            st.divider()
            st.markdown("""
            ### What this history is used for

            | Feature group | Uses history? | How |
            |---|---|---|
            | **C-columns** (infrastructure counts) | ✅ Precomputed | Groupby dicts at startup |
            | **D-columns** (time deltas) | ✅ Precomputed | First/last timestamp per entity |
            | **Card amount stats** | ✅ Precomputed | Mean/std per card1 |
            | **Velocity features** | ✅ Precomputed | Last timestamp per card |
            | **M-columns** | ❌ Live input | Payment processor provides at runtime |
            | **Time/Amount** | ❌ No history | Computed from transaction fields only |

            > All C and D features use **O(1) dictionary lookups** — no DataFrame scanning at inference time.
            """)