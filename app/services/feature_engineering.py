# app/services/feature_engineering.py
import json
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

async def calculate_customer_features(db: AsyncSession, customer_id: str):
    # Load the exact column order the model expects
    with open('app/ml_models/feature_columns.json', 'r') as f:
        feature_columns = json.load(f)

    # Write the exact same aggregation logic as our training script, but in SQL
    query = text("""
        SELECT 
            SUM(amount_paid) as total_spend,
            AVG(amount_paid) as avg_invoice_amount,
            COUNT(amount_paid) as total_invoices,
            SUM(CASE WHEN attempt_count > 1 THEN 1 ELSE 0 END) as failed_payment_count
        FROM invoices
        WHERE customer_id = :customer_id
        GROUP BY customer_id
    """)
    
    result = await db.execute(query, {"customer_id": customer_id})
    row = result.fetchone()

    if not row:
        return None # Customer not found

    # Map SQL results to match the exact feature order
    feature_dict = {
        "total_spend": float(row[0]) if row[0] else 0.0,
        "avg_invoice_amount": float(row[1]) if row[1] else 0.0,
        "total_invoices": int(row[2]) if row[2] else 0,
        "failed_payment_count": int(row[3]) if row[3] else 0,
        "tenure_months": int(row[2]) if row[2] else 0 # Tenure is just invoice count
    }

    # Return as a list in the EXACT order the model was trained on
    features = [feature_dict[col] for col in feature_columns]
    return features