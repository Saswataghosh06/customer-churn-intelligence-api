<div align="center">
<img width="1584" height="396" alt="Image" src="https://github.com/user-attachments/assets/64ca71d0-9002-4147-aa04-22e3f03b1686" />
</div>

<h1 align="center">Customer Churn Intelligence API</h1>
<h3 align="center">System Data Dictionary & Schema Reference</h3>

<div align="center">

<img alt="status" src="https://img.shields.io/badge/status-production_ready-1E3A5F?style=flat-square">
<img alt="data" src="https://img.shields.io/badge/data-Simulated%20SaaS%20Billing-8B98AE?style=flat-square">
<img alt="stack" src="https://img.shields.io/badge/stack-FastAPI%20%7C%20MLflow%20%7C%20Docker-1E3A5F?style=flat-square">
<img alt="scale" src="https://img.shields.io/badge/ARR-$5.1M_%7C_At_Risk-$3.9M-e74c3c?style=flat-square">

<br><br>
<b>Saswata Ghosh</b><br>
<a href="https://github.com/Saswataghosh06/customer-churn-intelligence-api">GitHub</a> · <a href="https://www.linkedin.com/in/saswata-ghosh06/">LinkedIn</a> · <a href="mailto:saswataghosh2022@gmail.com">Email</a>
</div>

**Document Version:** 1.0  
**System:** Customer Churn Intelligence API  
**Primary Database:** PostgreSQL 15 (Dockerized)  
**Data Granularity:** Customer-level (Master), Invoice-line level (Transactions)  

---

## 1. Operational Source Tables (Bronze/Silver Layer)

These tables represent the raw ingestion layer mimicking a Stripe data dump. 

### Table: `customers`
**Description:** The customer master record. Contains firmographic data and the initial subscription state. 
*Note: In a true production SaaS environment with plan upgrades/downgrades, this would be a Slowly Changing Dimension (SCD Type 2). For this API, it captures the state at signup.*

| Column Name | Data Type | Nullable | PII Flag | Business Description | Technical / Transformation Logic | Example Values |
|---|---|---|---|---|---|---|
| `stripe_customer_id` | `VARCHAR(50)` | `NO` | Yes (Pseudo) | The unique identifier for the customer, mimicking Stripe's `cus_` prefix format. | Generated sequentially during ETL (`cus_mock_{i}`). Acts as the Primary Key and join key for all downstream tables. | `cus_mock_8451` |
| `email` | `VARCHAR(100)` | `NO` | **Yes (Direct)** | Customer contact email. | Formatted as `customer{i}@company.com` during simulation. Not used in ML model to prevent PII leakage. | `customer8451@company.com` |
| `company_size` | `VARCHAR(20)` | `NO` | No | Firmographic segment representing the size of the customer's business. | Simulated using weighted random choice: `SMB` (70%), `Mid-Market` (20%), `Enterprise` (10%). | `SMB`, `Enterprise` |
| `initial_plan` | `VARCHAR(20)` | `NO` | No | The subscription tier the customer signed up for on Day 1. | Simulated using weighted random choice: `Basic` ($29, 60%), `Pro` ($99, 30%), `Enterprise` ($299, 10%). | `Pro` |
| `created_at` | `TIMESTAMP` | `NO` | No | The exact timestamp the customer record was created in our system. | Defaults to `NOW()` upon ETL insertion. | `2023-06-15 14:22:11` |

<br>

### Table: `invoices`
**Description:** Transactional billing events. One row represents a single monthly billing cycle attempt for a customer.

| Column Name | Data Type | Nullable | PII Flag | Business Description | Technical / Transformation Logic | Example Values |
|---|---|---|---|---|---|---|
| `stripe_invoice_id` | `VARCHAR(50)` | `NO` | No | Unique identifier for the billing attempt. | Generated as `in_mock_{customer_id}_{month}`. Primary Key. | `in_mock_8451_4` |
| `customer_id` | `VARCHAR(50)` | `NO` | Yes (Pseudo) | Foreign Key linking back to the customer. | Must exist in `customers.stripe_customer_id`. | `cus_mock_8451` |
| `amount_paid` | `FLOAT` | `NO` | No | The actual revenue collected for this invoice. | If `invoice_status = 'paid'`, this reflects the plan price (e.g., 99.0). If failed/void, this is `0.0`. Used to calculate `total_spend` and MRR. | `99.00`, `0.00` |
| `attempt_count` | `INTEGER` | `NO` | No | The number of times Stripe attempted to charge the payment method. | `1` = Success on first try. `>1` = Card was declined initially, Stripe's smart retry logic kicked in. **This is the primary feature driving churn.** | `1`, `2`, `3` |
| `invoice_status` | `VARCHAR(20)` | `NO` | No | The final state of the invoice. | Simulated states: <br>`paid` (Success - 80% base rate)<br>`void` (Manual cancellation - rare)<br>`uncollectible` (All retries failed, debt written off - primary churn driver) | `paid`, `uncollectible` |
| `billing_period_start`| `DATE` | `NO` | No | The start date of the service period this invoice covers. | Generated sequentially based on the customer's simulated start month. Used to calculate `tenure_months`. | `2023-09-01` |
| `billing_period_end` | `DATE` | `NO` | No | The end date of the service period. | Always `billing_period_start + 27 days` (simulating a 1-month cycle). Used to anchor recency calculations. | `2023-09-28` |

---

## 2. Logical Feature Store (Gold Layer - Computed On-The-Fly)

*Note: In this architecture, these features are not materialized as a physical table. They are computed dynamically via Async SQL (`feature_engineering.py`) when the API is hit. This ensures real-time scoring without batch lag.*

| Feature Name | Data Type | ML Model Input | Mathematical Definition | Business Interpretation |
|---|---|---|---|---|
| `total_spend` | `FLOAT` | Yes | `SUM(amount_paid)` grouped by `customer_id` | Total lifetime value (historical). |
| `avg_invoice_amount` | `FLOAT` | Yes | `AVG(amount_paid)` grouped by `customer_id` | Average monthly recurring revenue. |
| `total_invoices` | `INTEGER` | Yes | `COUNT(invoice_id)` grouped by `customer_id` | Acts as a proxy for `tenure_months`. |
| `failed_payment_count`| `INTEGER` | Yes | `SUM(CASE WHEN attempt_count > 1 THEN 1 ELSE 0 END)` | Number of months the customer's card failed. **Highest feature importance in Random Forest.** |
| `tenure_months` | `INTEGER` | Yes | Direct copy of `total_invoices` | How long the customer has been active. |

---

## 3. Target Variable Definition

The ML model does not predict `invoice_status`. It predicts a custom engineered target variable to prevent data leakage.

| Target Name | Data Type | Definition | Rationale |
|---|---|---|---|
| `is_churned` | `INTEGER` (0 or 1) | `1` if `uncollectible_count > 0` OR `total_invoices <= 2`, else `0`. | If we just used `uncollectible`, the model would ignore early cancellations. If we used `total_invoices <= 2`, we'd miss people who stuck around for 6 months then failed. This OR logic captures both "I quit early" and "I failed to pay." |

---

## 4. API Payload Schemas (Pydantic Models)

These are the strict JSON validation schemas enforced by FastAPI at the API boundary.

### Response: `PredictionResponse`
**Endpoint:** `GET /predictions/churn/{customer_id}`

| Field | JSON Type | Constraints | Description |
|---|---|---|---|
| `customer_id` | `string` | Must match regex `^cus_mock_\d+$` | Echoed back from the URL path for confirmation. |
| `churn_probability`| `float` | `>= 0.0` and `<= 1.0` | Raw output from `RandomForest.predict_proba()[0][1]`. |
| `risk_tier` | `string` | Must be in: `['Safe', 'Medium', 'High', 'Critical']` | Mapped from probability: `<0.2`, `0.2-0.4`, `0.4-0.7`, `>0.7`. |

**Example Payload:**
```json
{
  "customer_id": "cus_mock_8000",
  "churn_probability": 0.8521,
  "risk_tier": "Critical"
}
```

### Response: `HealthResponse`
**Endpoint:** `GET /health`

| Field | JSON Type | Description |
|---|---|---|
| `status` | `string` | Hardcoded `ok` if server is running. |
| `database` | `string` | Result of a lightweight `SELECT 1` ping to Postgres (`connected` or `disconnected`). |

---

## 5. Data Governance & PII Classification

To ensure this project could be deployed in a real enterprise environment under GDPR/CCPA:

| Data Element | Classification | Action Required for Prod |
|---|---|---|
| `customers.email` | **High Risk (Direct PII)** | Must be encrypted at rest in Postgres (e.g., `pgcrypto`). **Must be excluded** from ML feature engineering to prevent model inversion attacks. |
| `customers.stripe_customer_id`| **Medium Risk (Pseudo-PII)** | Can be used in features/logs, but should be masked in frontend UIs (e.g., `cus_****_8000`). |
| `invoices.amount_paid` | **Low Risk (Aggregated)** | Safe to use. Only risky if a single invoice can uniquely identify a person (rare in B2B). |
| `ML Probabilities` | **Sensitive Business Data** | Predicting a customer will churn is proprietary intel. API must require Authentication (e.g., `X-API-Key` header) to prevent competitors from scraping the endpoint. |
```
