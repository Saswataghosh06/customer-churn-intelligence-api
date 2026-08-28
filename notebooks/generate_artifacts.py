# notebooks/generate_readme_artifacts.py
import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# Setup professional chart styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150

# 1. Connect to DB
db_url = os.getenv("DATABASE_URL", "postgresql://churn_user:churn_pass@localhost:5432/churn_db").replace("+asyncpg", "")
engine = create_engine(db_url)
print("Connecting to database...")

# 2. Extract & Aggregate Data
df = pd.read_sql("SELECT * FROM invoices", engine)
customers = pd.read_sql("SELECT * FROM customers", engine)

print("Engineering features for all 10,000 customers...")
df_agg = df.groupby('customer_id').agg(
    total_spend=('amount_paid', 'sum'),
    avg_invoice_amount=('amount_paid', 'mean'),
    total_invoices=('amount_paid', 'count'),
    failed_payment_count=('attempt_count', lambda x: (x > 1).sum()),
    uncollectible_count=('invoice_status', lambda x: (x == 'uncollectible').sum()),
    last_invoice_date=('billing_period_end', 'max')
).reset_index()

df_agg['tenure_months'] = df_agg['total_invoices']
df_agg['is_churned'] = np.where((df_agg['uncollectible_count'] > 0) | (df_agg['tenure_months'] <= 2), 1, 0)

# Calculate MRR (Using their last known invoice amount as current MRR)
mrr_df = df.sort_values('billing_period_end').groupby('customer_id').last().reset_index()[['customer_id', 'amount_paid']]
mrr_df.rename(columns={'amount_paid': 'current_mrr'}, inplace=True)
df_agg = df_agg.merge(mrr_df, on='customer_id', how='left')
df_agg['current_mrr'] = df_agg['current_mrr'].fillna(0)

# 3. Load ML Model & Predict for Everyone
print("Loading ML model and predicting risk for entire customer base...")
with open('app/ml_models/feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

scaler = joblib.load('app/ml_models/scaler.pkl')
model = joblib.load('app/ml_models/model.pkl')

X = df_agg[feature_columns]
X_scaled = scaler.transform(X)
df_agg['churn_probability'] = model.predict_proba(X_scaled)[:, 1]

def assign_tier(prob):
    if prob > 0.7: return 'Critical'
    elif prob > 0.4: return 'High'
    elif prob > 0.2: return 'Medium'
    else: return 'Safe'

df_agg['risk_tier'] = df_agg['churn_probability'].apply(assign_tier)

# 4. Calculate Business Metrics
total_customers = len(df_agg)
total_mrr = df_agg['current_mrr'].sum()
total_arr = total_mrr * 12
true_churn_count = df_agg['is_churned'].sum()
true_churn_rate = true_churn_count / total_customers

at_risk_df = df_agg[df_agg['risk_tier'].isin(['High', 'Critical'])]
at_risk_mrr = at_risk_df['current_mrr'].sum()
at_risk_arr = at_risk_mrr * 12

print("\n" + "="*50)
print("COPY AND PASTE THIS TO YOUR MENTOR:")
print("="*50)
print(f"TOTAL_CUSTOMERS: {total_customers:,}")
print(f"TRUE_CHURN_RATE: {true_churn_rate*100:.1f}%")
print(f"TOTAL_MRR: ${total_mrr:,.2f}")
print(f"TOTAL_ARR: ${total_arr:,.2f}")
print(f"AT_RISK_CUSTOMERS: {len(at_risk_df):,}")
print(f"AT_RISK_ARR: ${at_risk_arr:,.2f}")
print("="*50)

# 5. Generate Charts
img_dir = 'docs/images/'
os.makedirs(img_dir, exist_ok=True)

# Chart 1: Risk Tier Distribution
print("Generating charts...")
tier_counts = df_agg['risk_tier'].value_counts().reindex(['Safe', 'Medium', 'High', 'Critical'])
colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=tier_counts.index, y=tier_counts.values, palette=colors, ax=ax)
ax.set_title("Customer Distribution by Predicted Risk Tier", fontsize=14, fontweight='bold')
ax.set_ylabel("Number of Customers")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{img_dir}risk_tier_distribution.png')
plt.close()

# Chart 2: Revenue at Risk (ARR by Tier)
arr_by_tier = df_agg.groupby('risk_tier')['current_mrr'].sum().reindex(['Safe', 'Medium', 'High', 'Critical']) * 12

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=arr_by_tier.index, y=arr_by_tier.values, palette=colors, ax=ax)
ax.set_title("Annual Recurring Revenue (ARR) Exposed by Risk Tier", fontsize=14, fontweight='bold')
ax.set_ylabel("ARR ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
for p in ax.patches:
    ax.annotate(f'${int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{img_dir}revenue_at_risk.png')
plt.close()

# Chart 3: ML Feature Importances
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importances}).sort_values(by='Importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', ax=ax)
ax.set_title("Random Forest: Top Drivers of Churn", fontsize=14, fontweight='bold')
ax.set_xlabel("Relative Importance")
plt.tight_layout()
plt.savefig(f'{img_dir}feature_importance.png')
plt.close()

print("✅ All charts saved to docs/images/")