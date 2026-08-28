# ml_pipeline/train_model.py
import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Connect to the Database (Using standard sync driver for Pandas)
print("Connecting to database to extract data...")
DB_URL = "postgresql://churn_user:churn_pass@localhost:5432/churn_db"
engine = create_engine(DB_URL)

# 2. Extract Data using a SQL Join
df = pd.read_sql("""
    SELECT 
        c.stripe_customer_id, 
        c.company_size, 
        c.initial_plan,
        i.amount_paid, 
        i.attempt_count, 
        i.invoice_status
    FROM customers c
    JOIN invoices i ON c.stripe_customer_id = i.customer_id
""", engine)

print(f"Extracted {len(df)} raw invoice records.")

# 3. Feature Engineering (Senior Approach: Aggregate to Customer Level)
print("Engineering features...")
customer_features = df.groupby('stripe_customer_id').agg(
    total_spend=('amount_paid', 'sum'),
    avg_invoice_amount=('amount_paid', 'mean'),
    total_invoices=('amount_paid', 'count'),
    failed_payment_count=('attempt_count', lambda x: (x > 1).sum()), # Failed if attempted more than once
    uncollectible_count=('invoice_status', lambda x: (x == 'uncollectible').sum())
).reset_index()

# Add Tenure (total_invoices represents months active in our simulated data)
customer_features['tenure_months'] = customer_features['total_invoices']

# 4. Define the Target Variable (Churn)
# Senior Logic: If a customer had an 'uncollectible' invoice (failed to pay and gave up), they churned.
# Alternatively, if they only lasted 1 or 2 months, they churned. Let's combine these.
def assign_churn(row):
    if row['uncollectible_count'] > 0:
        return 1
    elif row['tenure_months'] <= 2:
        return 1 # Signed up and left immediately
    else:
        return 0

customer_features['is_churned'] = customer_features.apply(assign_churn, axis=1)

# 5. Prepare for Machine Learning
# Drop columns that leak the future or aren't numeric
features_to_drop = ['stripe_customer_id', 'is_churned', 'uncollectible_count']
X = customer_features.drop(columns=features_to_drop)
y = customer_features['is_churned']

# Save the exact column order! The API MUST use this exact order later.
feature_columns = X.columns.tolist()
with open('app/ml_models/feature_columns.json', 'w') as f:
    json.dump(feature_columns, f)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (Important for some models, good practice for all)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Model & Log to MLflow
print("Training Random Forest and logging to MLflow...")
# Tell MLflow where to track the experiment (Our Docker container)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn_prediction_experiment")

with mlflow.start_run():
    # Train
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Predict
    predictions = rf_model.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Log parameters and metrics to MLflow UI
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the model AND the scaler together
    # Log the ML model to MLflow using the new modern syntax
    mlflow.sklearn.log_model(rf_model, name="churn_random_forest")

    # Save the scaler locally as a pickle file (Standard practice)
    import joblib
    joblib.dump(scaler, 'app/ml_models/scaler.pkl')

print("✅ Phase 2 Complete! Check http://localhost:5000 to see your model.")