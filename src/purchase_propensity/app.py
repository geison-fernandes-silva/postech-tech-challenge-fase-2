"""Business-facing purchase propensity demonstration interface."""

# ruff: noqa: E501

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.joblib"))

DEFAULT_FEATURES: dict[str, Any] = {
    "Administrative": 2,
    "Administrative_Duration": 45.0,
    "Informational": 1,
    "Informational_Duration": 20.0,
    "ProductRelated": 30,
    "ProductRelated_Duration": 310.0,
    "BounceRates": 0.02,
    "ExitRates": 0.04,
    "PageValues": 10.0,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False,
}


class VisitorSignals(BaseModel):
    """Business inputs exposed by the demonstration screen."""

    product_pages: int = Field(30, ge=0, le=500)
    page_value: float = Field(10.0, ge=0, le=500)
    bounce_rate: float = Field(0.02, ge=0, le=1)
    exit_rate: float = Field(0.04, ge=0, le=1)
    visitor_type: str = "Returning_Visitor"
    month: str = "Nov"
    weekend: bool = False


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load the latest locally generated model once per application process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact was not found at {MODEL_PATH}.")
    return joblib.load(MODEL_PATH)


def make_feature_frame(signals: VisitorSignals) -> pd.DataFrame:
    """Convert simplified business signals into the full model feature set."""
    features = DEFAULT_FEATURES | {
        "ProductRelated": signals.product_pages,
        "PageValues": signals.page_value,
        "BounceRates": signals.bounce_rate,
        "ExitRates": signals.exit_rate,
        "VisitorType": signals.visitor_type,
        "Month": signals.month,
        "Weekend": signals.weekend,
    }
    return pd.DataFrame([features])


def commercial_recommendation(probability: float) -> tuple[str, str]:
    """Translate a prediction into a simple commercial priority and action."""
    if probability >= 0.65:
        return "High priority", "Offer real-time assistance or a targeted incentive now."
    if probability >= 0.35:
        return "Nurture opportunity", "Keep the visitor engaged with relevant product information."
    return "Low priority", "Prioritize scalable remarketing rather than immediate outreach."


app = FastAPI(title="Purchase Propensity Demo", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Serve the interactive business demonstration screen."""
    return PAGE_HTML


@app.get("/health")
def health() -> dict[str, str]:
    """Report whether the model required by the interface is available."""
    return {"status": "ready" if MODEL_PATH.exists() else "model_missing"}


@app.post("/predict")
def predict(signals: VisitorSignals) -> dict[str, str | float]:
    """Return propensity, priority, and recommended commercial response."""
    try:
        probability = float(load_model().predict_proba(make_feature_frame(signals))[0, 1])
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    priority, action = commercial_recommendation(probability)
    return {"probability": round(probability, 4), "priority": priority, "action": action}


def main() -> None:
    """Start the local business demonstration server."""
    uvicorn.run(app, host="0.0.0.0", port=8080)


PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Purchase Propensity | Demo</title>
  <style>
    body { margin: 0; background: #f7f5f2; color: #292523; font-family: Arial, sans-serif; }
    .shell { max-width: 1000px; margin: 0 auto; padding: 48px 28px; }
    .tag { color: #a33125; font-weight: 700; letter-spacing: .1em; font-size: 13px; }
    h1 { font-size: 44px; margin: 10px 0 12px; }
    p { color: #615b57; font-size: 18px; line-height: 1.5; }
    .grid { display: grid; grid-template-columns: 1.15fr .85fr; gap: 24px; margin-top: 32px; }
    .card { background: #fff; border: 1px solid #ded7d0; border-radius: 14px; padding: 28px; }
    label { font-weight: 700; display: block; margin-top: 16px; }
    input, select { box-sizing: border-box; margin-top: 7px; width: 100%; padding: 11px; }
    input, select { border: 1px solid #cfc7c0; border-radius: 7px; font-size: 16px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    button { margin-top: 24px; background: #b63628; color: #fff; border: 0; border-radius: 7px; }
    button { padding: 13px 20px; font-size: 16px; font-weight: 700; cursor: pointer; width: 100%; }
    .result { min-height: 260px; display: flex; flex-direction: column; justify-content: center; }
    .muted { font-size: 15px; }.score { font-size: 56px; font-weight: 700; color: #2f6a53; }
    .priority { font-size: 20px; font-weight: 700; color: #a33125; }
    @media(max-width: 700px) { .grid { grid-template-columns: 1fr; } h1 { font-size: 34px; } }
  </style>
</head>
<body>
  <main class="shell">
    <div class="tag">E-COMMERCE DECISION SUPPORT</div>
    <h1>Purchase propensity advisor</h1>
    <p>Turn visitor behavior into a clear commercial priority and recommended next action.</p>
    <section class="grid">
      <form class="card" id="form">
        <h2>Visitor signals</h2>
        <div class="row">
          <label>Product pages viewed<input name="product_pages" type="number" min="0" value="30"></label>
          <label>Page value<input name="page_value" type="number" min="0" step="0.1" value="10"></label>
        </div>
        <div class="row">
          <label>Bounce rate<input name="bounce_rate" type="number" min="0" max="1" step="0.01" value="0.02"></label>
          <label>Exit rate<input name="exit_rate" type="number" min="0" max="1" step="0.01" value="0.04"></label>
        </div>
        <label>Visitor type<select name="visitor_type"><option value="Returning_Visitor">Returning visitor</option><option value="New_Visitor">New visitor</option><option value="Other">Other</option></select></label>
        <div class="row">
          <label>Month<select name="month"><option>Nov</option><option>May</option><option>Oct</option><option>Dec</option><option>Mar</option><option>Aug</option></select></label>
          <label>Weekend<select name="weekend"><option value="false">No</option><option value="true">Yes</option></select></label>
        </div>
        <button>Assess purchase propensity</button>
      </form>
      <aside class="card result" id="result">
        <div class="tag">COMMERCIAL RECOMMENDATION</div>
        <h2>Ready for assessment</h2>
        <p class="muted">Enter visitor signals to receive a model-based propensity score and suggested action.</p>
      </aside>
    </section>
  </main>
  <script>
    const form = document.getElementById('form');
    const result = document.getElementById('result');
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const data = Object.fromEntries(new FormData(form));
      data.product_pages = Number(data.product_pages);
      ['page_value', 'bounce_rate', 'exit_rate'].forEach((key) => data[key] = Number(data[key]));
      data.weekend = data.weekend === 'true';
      result.innerHTML = '<div class="tag">COMMERCIAL RECOMMENDATION</div><h2>Assessing visitor...</h2>';
      const response = await fetch('/predict', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
      });
      if (!response.ok) {
        result.innerHTML = '<h2>Model unavailable</h2><p>Please run the training pipeline first.</p>';
        return;
      }
      const prediction = await response.json();
      result.innerHTML = `<div class="tag">PURCHASE PROPENSITY</div>
        <div class="score">${(prediction.probability * 100).toFixed(1)}%</div>
        <div class="priority">${prediction.priority}</div><h2>Suggested next action</h2>
        <p>${prediction.action}</p><p class="muted">Prediction generated by the registered model.</p>`;
    });
  </script>
</body>
</html>"""
