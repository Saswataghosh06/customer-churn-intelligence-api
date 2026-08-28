# app/services/churn_service.py
import joblib

model = None
scaler = None

def load_ml_models():
    """Loads baked production models directly into memory."""
    global model, scaler
    
    scaler = joblib.load('app/ml_models/scaler.pkl')
    model = joblib.load('app/ml_models/model.pkl')
    
    print("✅ ML models loaded from local production files.")

def predict_churn_risk(features: list) -> dict:
    if not model or not scaler:
        raise Exception("Models not loaded!")
    
    scaled_features = scaler.transform([features])
    probability = model.predict_proba(scaled_features)[0][1]
    
    if probability > 0.7: tier = "Critical"
    elif probability > 0.4: tier = "High"
    elif probability > 0.2: tier = "Medium"
    else: tier = "Safe"
        
    return {
        "churn_probability": round(probability, 4),
        "risk_tier": tier
    }