# app/models/customer.py
from sqlalchemy import Column, String, Float, Integer, Date, ForeignKey, DateTime
from sqlalchemy.sql import func
from app.database import Base

class Customer(Base):
    __tablename__ = "customers"

    stripe_customer_id = Column(String(50), primary_key=True, index=True)
    email = Column(String(100))
    company_size = Column(String(20))
    initial_plan = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Invoice(Base):
    __tablename__ = "invoices"

    stripe_invoice_id = Column(String(50), primary_key=True, index=True)
    customer_id = Column(String(50), ForeignKey("customers.stripe_customer_id"))
    amount_paid = Column(Float)
    attempt_count = Column(Integer)
    invoice_status = Column(String(20))
    billing_period_start = Column(Date)
    billing_period_end = Column(Date)