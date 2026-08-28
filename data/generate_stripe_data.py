import os
import random
import time
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to the PostgreSQL database running in Docker
print("Connecting to PostgreSQL...")
# Automatically strip out +asyncpg if the .env has it, so psycopg2 understands it
clean_url = os.getenv("DATABASE_URL").replace("+asyncpg", "")
conn = psycopg2.connect(clean_url)
cur = conn.cursor()

# --- STEP 1: CREATE TABLES ---
print("Creating tables...")
cur.execute("""
CREATE TABLE IF NOT EXISTS customers (
    stripe_customer_id VARCHAR(50) PRIMARY KEY,
    email VARCHAR(100),
    company_size VARCHAR(20),
    initial_plan VARCHAR(20),
    created_at TIMESTAMP
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS invoices (
    stripe_invoice_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(stripe_customer_id),
    amount_paid FLOAT,
    attempt_count INT,
    invoice_status VARCHAR(20),
    billing_period_start DATE,
    billing_period_end DATE
);
""")
conn.commit()

# --- STEP 2: SIMULATE STRIPE DATA ---
print("Simulating 10,000 SaaS customers over 12 months...")
np.random.seed(42)
random.seed(42)

n_customers = 10000
plans = ['Basic', 'Pro', 'Enterprise']
plan_prices = {'Basic': 29.0, 'Pro': 99.0, 'Enterprise': 299.0}
company_sizes = ['SMB', 'Mid-Market', 'Enterprise']
statuses = ['paid', 'paid', 'paid', 'paid', 'paid', 'paid', 'paid', 'paid', 'void', 'uncollectible'] # 80% paid rate base

customers_data = []
invoices_data = []

for i in range(1, n_customers + 1):
    cust_id = f"cus_mock_{i}"
    email = f"customer{i}@company.com"
    company = random.choices(company_sizes, weights=[70, 20, 10])[0]
    plan = random.choices(plans, weights=[60, 30, 10])[0]
    
    # Assign a random start month (1 to 12)
    start_month = random.randint(1, 12)
    
    customers_data.append((cust_id, email, company, plan, f"2023-{start_month:02d}-01"))

    # Simulate monthly invoices until they churn or reach month 12
    current_plan = plan
    for m in range(start_month, 13):
        # Simulate Churn / Downgrade logic (Senior Engineer touch)
        if random.random() < 0.08: # 8% chance to churn each month
            break
        if current_plan == 'Enterprise' and random.random() < 0.15:
            current_plan = 'Pro' # Downgrade
        elif current_plan == 'Pro' and random.random() < 0.1:
            current_plan = 'Basic' # Downgrade

        amount = plan_prices[current_plan]
        status = random.choice(statuses)
        attempt_count = 1 if status == 'paid' else random.randint(2, 4)
        
        invoices_data.append((
            f"in_mock_{i}_{m}",
            cust_id,
            amount if status == 'paid' else 0.0,
            attempt_count,
            status,
            f"2023-{m:02d}-01",
            f"2023-{m:02d}-28"
        ))

# --- STEP 3: INSERT INTO DATABASE ---
print(f"Inserting {len(customers_data)} customers into DB...")
# psycopg2.extras.execute_values is the fastest way to bulk insert in Python
from psycopg2.extras import execute_values
execute_values(cur, "INSERT INTO customers (stripe_customer_id, email, company_size, initial_plan, created_at) VALUES %s", customers_data)

print(f"Inserting {len(invoices_data)} invoices into DB...")
execute_values(cur, "INSERT INTO invoices (stripe_invoice_id, customer_id, amount_paid, attempt_count, invoice_status, billing_period_start, billing_period_end) VALUES %s", invoices_data)

conn.commit()
cur.close()
conn.close()

print("✅ Phase 0 Complete! Database is populated and ready for Machine Learning.")