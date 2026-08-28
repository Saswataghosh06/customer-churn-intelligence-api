# notebooks/generate_deep_eda.py
import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

# 1. Connect & Aggregate
db_url = os.getenv("DATABASE_URL", "postgresql://churn_user:churn_pass@localhost:5432/churn_db").replace("+asyncpg", "")
engine = create_engine(db_url)
df = pd.read_sql("SELECT * FROM invoices", engine)

df_agg = df.groupby('customer_id').agg(
    total_spend=('amount_paid', 'sum'),
    avg_invoice_amount=('amount_paid', 'mean'),
    total_invoices=('amount_paid', 'count'),
    failed_payment_count=('attempt_count', lambda x: (x > 1).sum()),
    uncollectible_count=('invoice_status', lambda x: (x == 'uncollectible').sum())
).reset_index()

df_agg['tenure_months'] = df_agg['total_invoices']
df_agg['is_churned'] = np.where((df_agg['uncollectible_count'] > 0) | (df_agg['tenure_months'] <= 2), 1, 0)

img_dir = 'docs/images/'
os.makedirs(img_dir, exist_ok=True)

# --- CHART 1: Spend Distribution by Churn Status (For Data Analysts) ---
print("Generating EDA Chart 1: Spend Distribution...")
fig, ax = plt.subplots(figsize=(8, 5))
# Filter out extreme outliers for better visualization
df_viz = df_agg[df_agg['total_spend'] < df_agg['total_spend'].quantile(0.95)]
sns.histplot(data=df_viz, x='total_spend', hue='is_churned', kde=True, bins=30, palette=['#2ecc71', '#e74c3c'], ax=ax)
ax.set_title("Distribution of Total Spend by Churn Status (Top 95%)", fontsize=14, fontweight='bold')
ax.set_xlabel("Total Spend ($)")
ax.set_ylabel("Number of Customers")
ax.legend(title='Churned', labels=['Active (0)', 'Churned (1)'])
plt.tight_layout()
plt.savefig(f'{img_dir}eda_spend_distribution.png')
plt.close()

# --- CHART 2: Failed Payments vs Churn (Proving the #1 Driver) ---
print("Generating EDA Chart 2: Failed Payments Impact...")
fig, ax = plt.subplots(figsize=(7, 5))
# Cap at 3 for visualization, group the rest
df_viz2 = df_agg.copy()
df_viz2['failed_payments_grouped'] = df_viz2['failed_payment_count'].apply(lambda x: '3+' if x >= 3 else str(x))
sns.countplot(data=df_viz2, x='failed_payments_grouped', hue='is_churned', palette=['#2ecc71', '#e74c3c'], ax=ax)
ax.set_title("Churn Rate Skyrockets with Failed Payment Attempts", fontsize=14, fontweight='bold')
ax.set_xlabel("Number of Failed Payment Attempts")
ax.set_ylabel("Customer Count")
ax.legend(title='Churned', labels=['Active (0)', 'Churned (1)'])
plt.tight_layout()
plt.savefig(f'{img_dir}eda_failed_payments.png')
plt.close()

# --- CHART 3: Feature Correlation Matrix (For ML Engineers) ---
print("Generating EDA Chart 3: Correlation Matrix...")
with open('app/ml_models/feature_columns.json', 'r') as f:
    features = json.load(f)
    
df_ml = df_agg[features + ['is_churned']]
corr_matrix = df_ml.corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
ax.set_title("Feature Correlation Matrix (ML Inputs)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{img_dir}eda_correlation_matrix.png')
plt.close()

print("✅ Deep EDA charts saved to docs/images/")