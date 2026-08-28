# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import health, predictions
from app.services import churn_service

# Lifespan event: Runs once when the server starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML models into memory before accepting requests
    churn_service.load_ml_models()
    yield
    # Clean up happens here when server shuts down (if needed)

# Create the FastAPI app, passing the lifespan
app = FastAPI(
    title="Customer Churn Intelligence API",
    description="A production-grade microservice for predicting SaaS churn via Stripe data.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow Streamlit frontend to talk to this API later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our routers
app.include_router(health.router)
app.include_router(predictions.router) # Added the new predictions router!