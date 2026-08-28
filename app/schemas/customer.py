# app/schemas/customer.py
from pydantic import BaseModel
from datetime import datetime, date

class CustomerResponse(BaseModel):
    stripe_customer_id: str
    email: str
    company_size: str
    initial_plan: str
    created_at: datetime

    class Config:
        from_attributes = True

class HealthResponse(BaseModel):
    status: str
    database: str