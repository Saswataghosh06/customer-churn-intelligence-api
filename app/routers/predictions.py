# app/routers/predictions.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services import feature_engineering, churn_service
from pydantic import BaseModel

router = APIRouter(tags=["Predictions"])

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str

@router.get("/predictions/churn/{customer_id}", response_model=PredictionResponse)
async def get_churn_prediction(customer_id: str, db: AsyncSession = Depends(get_db)):
    # 1. Get features from DB
    features = await feature_engineering.calculate_customer_features(db, customer_id)
    
    if features is None:
        raise HTTPException(status_code=404, detail="Customer not found in invoice database.")
    
    # 2. Get prediction from ML Service
    prediction_result = churn_service.predict_churn_risk(features)
    
    # 3. Return combined response
    return PredictionResponse(
        customer_id=customer_id,
        churn_probability=prediction_result["churn_probability"],
        risk_tier=prediction_result["risk_tier"]
    )