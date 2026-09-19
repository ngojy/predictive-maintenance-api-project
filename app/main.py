import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

model_path = os.getenv("MODEL_PATH", "models/model.joblib")

app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting machine failure based on sensor data.",
    version="1.0.0",
)

_artifacts = None

# loads model file once and keep it in memory, to avoid reloading the model for every request
def get_artifacts():
    global _artifacts
    if _artifacts is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        _artifacts = joblib.load(model_path)
    return _artifacts

# define Pydantic models for request and response, if someone sends bad data (example: negative vibration), fastapi will auto-reject it
class SensorReading(BaseModel):
    vibration_mm_s: float = Field(..., ge=0,description="Vibration in mm/s RMS")
    temperature_C: float = Field(..., ge=-40, le=300, description="Temperature in Celsius")
    rpm: float = Field(..., ge=0,description="Shaft speed in RPM")
    load_pct: float = Field(..., ge=0, le=150, description="Load percentage of rated capacity")
    hours_since_maintenance: float = Field(..., ge=0, description="Operating hours since last maintenance")

class PredictionResponse(BaseModel):
    failure_probability: float
    failure_predicted: bool
    risk_level: str

def risk_level_from_prob(prob: float) -> str:
    if prob < 0.2:
        return "low"
    elif prob < 0.5:
        return "moderate"
    elif prob < 0.8:
        return "high"
    else:
        return "critical"

# simple check endpoint to verify that the service is running and the model is loaded
@app.get("/health")
def health_check():
    try:
        get_artifacts()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

# takes a sensor reading, runs the model, and returns the prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    artifacts = get_artifacts()
    model = artifacts["model"]
    features = artifacts["features"]

    x = np.array([[getattr(reading, f) for f in features]])

    prob = model.predict_proba(x)[0][1]
    failure_predicted = prob >= 0.5
    risk_level = risk_level_from_prob(prob)

    return PredictionResponse(
        failure_probability=prob,
        failure_predicted=failure_predicted,
        risk_level=risk_level
    )

@app.get("/")
def root():
    return {"service": "predictive-maintenance-api", "docs": "/docs"}