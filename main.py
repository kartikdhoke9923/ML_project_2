from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field, field_validator
import joblib
import pickle
from typing import List, Literal, Optional
import numpy as np
import pandas as pd

# importing ml model
model = joblib.load('lgb_pipeline.pkl')

app=FastAPI()

# pydantic model for input data validation
class User_input(BaseModel):
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD:      str   = Field(..., description="Product code: W, H, C, S, R")
    TransactionDT:  Optional[int] = Field(None, description="Unix-like timestamp offset. If None, current time is used.")
 
    # ── Card details ──────────────────────────────────────────────────
    card1: int            = Field(..., description="Card identifier 1 (masked)")
    card2: Optional[float] = Field(None)
    card3: Optional[float] = Field(None)
    card4: Optional[str]  = Field(None, description="Card network: visa, mastercard, etc.")
    card6: Optional[str]  = Field(None, description="Card type: credit, debit")
 
    # ── Address ───────────────────────────────────────────────────────
    addr1: Optional[float] = Field(None, description="Billing zip code")
    addr2: Optional[float] = Field(None, description="Billing country code")
 
    # ── Email ─────────────────────────────────────────────────────────
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
 
    # ── Device (from identity file in training; from device fingerprint in prod) ──
    DeviceType: Optional[str] = Field(None, description="mobile or desktop")
    DeviceInfo: Optional[str] = Field(None, description="Device fingerprint string")
 
    # ── M-columns: match flags from payment processor ─────────────────
    # These are provided by Vesta/payment system — not computed by us but still kept 
    M1: Optional[str] = Field(None, description="Name on card vs billing name match")
    M2: Optional[str] = Field(None)
    M3: Optional[str] = Field(None)
    M4: Optional[str] = Field(None)
    M5: Optional[str] = Field(None)
    M6: Optional[str] = Field(None)
    M7: Optional[str] = Field(None)
    M8: Optional[str] = Field(None)
    M9: Optional[str] = Field(None)

@app.post("/predict")
def predict_value(data: User_input):

    # creating a dataframe from the input data
    input_data = data.model_dump()
    input_df = pd.DataFrame([input_data])
    prediction = model.predict_proba(input_df)[:, 1][0]  # Get the probability of the positive class
    if prediction < 0.3:
        risk = "LOW"
    elif prediction < 0.6:
        risk = "MEDIUM"
    else:
        risk = "HIGH"


    return JSONResponse(status_code=200 ,content={"fraud_probability": prediction, "risk": risk})  
