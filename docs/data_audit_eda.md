<div align="center">
<img width="1584" height="396" alt="Image" src="https://github.com/user-attachments/assets/64ca71d0-9002-4147-aa04-22e3f03b1686" />
</div>

<h1 align="center">Customer Churn Intelligence API</h1>
<h3 align="center">Data Audit & Exploratory Data Analysis (EDA)</h3>

<div align="center">

<img alt="status" src="https://img.shields.io/badge/status-production_ready-1E3A5F?style=flat-square">
<img alt="data" src="https://img.shields.io/badge/data-Simulated%20SaaS%20Billing-8B98AE?style=flat-square">
<img alt="stack" src="https://img.shields.io/badge/stack-FastAPI%20%7C%20MLflow%20%7C%20Docker-1E3A5F?style=flat-square">
<img alt="scale" src="https://img.shields.io/badge/ARR-$5.1M_%7C_At_Risk-$3.9M-e74c3c?style=flat-square">

<br><br>
<b>Saswata Ghosh</b><br>
<a href="https://github.com/Saswataghosh06/customer-churn-intelligence-api">GitHub</a> · <a href="https://www.linkedin.com/in/saswata-ghosh06/">LinkedIn</a> · <a href="mailto:saswataghosh2022@gmail.com">Email</a>
</div>

---

**Document Version:** 1.0  
**System:** Customer Churn Intelligence API  
**Primary Database:** PostgreSQL 15 (Dockerized)  
**Data Granularity:** Customer-level (Master), Invoice-line level (Transactions)  

---

## 1. Data Provenance & Baseline Metrics

Unlike messy real-world data extracts, this dataset was generated via a controlled Python ETL script (`data/generate_stripe_data.py`) using fixed random seeds. However, it was audited rigorously to ensure the simulation produced realistic SaaS financial patterns.

| Metric | Value | Notes |
|---|---|---|
| **Total Customers** | 10,000 | Generated at inception. |
| **Customers with Invoices** | 9,217 | 783 customers churned so early (month 1) they generated $0 revenue. Excluded from ML training. |
| **Total Invoice Records** | 45,290 | Represents ~12 months of billing cycles. |
| **Observation Window** | Jan 2023 – Dec 2023 | Fixed end-date used for all recency calculations. |
| **Historical Churn Rate** | **63.0%** | Aggressive definition: `uncollectible` OR `tenure <= 2` months. |

---

## 2. Data Quality & Integrity Assessment

While synthetic, the data was validated against standard SaaS billing anomalies:

| Check | Status | Details |
|---|---|---|
| **Null Values** | ✅ PASS | 0% nulls across all columns. Enforced by `NOT NULL` constraints in Postgres. |
| **Duplicate Invoices** | ✅ PASS | Composite Primary Key (`stripe_invoice_id`) prevents duplicate billing lines. |
| **Negative Revenue** | ⚠️ ADJUSTED | Real Stripe data contains negative `amount_paid` for refunds/credits. This simulation excluded refunds to simplify the CLV heuristic, but a production system *must* account for net revenue. |
| **Temporal Leaks** | ✅ PASS | All invoices have `billing_period_end` strictly before the max dataset date. No future-dated transactions. |
| **Entity Resolution** | ✅ PASS | 100% of invoices successfully map to a valid `customer_id`. No orphaned transactions. |

---

## 3. Exploratory Data Analysis (Deep Dive)

### 3.1 Revenue Distribution & Churn Segmentation
To visualize the heavily right-skewed nature of SaaS revenue, we filtered the top 5% of spenders ("whales") to focus on the core distribution.

![Spend Distribution](images/eda_spend_distribution.png)

**Statistical Takeaways:**
* **The "Micro-SMB" Trap:** The vast majority of customers (both Active and Churned) cluster below $200 in total lifetime spend. 
* **Churn Concentration:** The red distribution (Churned) has a visibly higher peak at the $0-$50 range compared to the green (Active). Low historical spend is a weak but valid predictor of churn.
* **Why we didn't use raw `total_spend` in the model:** Because of the extreme right-skew (a few customers spending $5,000+), raw spend destabilizes tree splits. We rely on `avg_invoice_amount` and `failed_payment_count` instead, which are less prone to outlier distortion.

### 3.2 The Primary Driver: Payment Friction
This is the most critical finding for business stakeholders. We analyzed the relationship between `attempt_count > 1` (Stripe's smart retry logic kicking in) and the churn label.

![Failed Payments](images/eda_failed_payments.png)

**Statistical Takeaways:**
* **The "0 Failed" Baseline:** Customers who never experience a card decline are overwhelmingly Active (green > red).
* **The Tipping Point:** At `1` failed attempt, the ratio begins to invert. By `2` or `3+` failed attempts, the red bars (Churned) heavily dominate.
* **ML Validation:** This chart empirically proves *why* the Random Forest model assigned a 34% relative importance to `failed_payment_count`. It is not a statistical artifact; it is a direct reflection of SaaS billing physics—payment friction is the strongest leading indicator of cancellation.

### 3.3 Multicollinearity & Linear Separability
Before selecting our model, we analyzed the Pearson correlation matrix of the 5 engineered ML features.

![Correlation Matrix](images/eda_correlation_matrix.png)

**Statistical Takeaways:**
* **Moderate Collinearity (0.69):** `total_spend` and `avg_invoice_amount` are naturally correlated (customers who spend more per month naturally have a higher total). 
    * *Impact on Modeling:* If we were using Logistic Regression, this 0.69 correlation would inflate variance and make coefficients unstable. However, Random Forests are immune to multicollinearity—they simply pick the best split between the two and ignore the redundant one.
* **Weak Linear Predictors:** Notice that no single feature has a strong linear correlation with `is_churned` (all are `< 0.20`). 
    * *Impact on Modeling:* This proves that churn is a **non-linear** problem. Customers don't churn based on a single metric moving linearly; they churn based on a *combination* of events (e.g., moderate spend + 1 failed payment + short tenure). This justifies bypassing simple Logistic Regression in favor of a non-linear ensemble model like Random Forest.

---

## 4. Target Variable Analysis & Class Imbalance

The base churn rate in this dataset is **63.0%**. 

### Is this a data error?
No. In the ETL phase, we defined churn aggressively:
`1` = Customer had an `uncollectible` invoice (failed all retries) **OR** customer lifespan was `<= 2` months (signed up and quit immediately).

### Impact on the Machine Learning Model
1. **Model Bias:** The model is slightly biased toward predicting "Churned" because it sees that state 63% of the time during training.
2. **Threshold Calibration:** Because of this imbalance, a standard `0.50` probability threshold doesn't make sense. If the model outputs `0.45`, it is essentially saying "I'm not sure, but given the base rate, I lean churned." 
3. **The "Missing Middle" Effect:** This class imbalance is the exact reason why the Executive Summary shows almost no customers in the "Medium" risk tier (20-40% probability). The model's probabilities are polarized—it is highly confident in its predictions, pushing customers to the extremes (Safe vs. Critical) rather than sitting on the fence.

### Why we didn't use SMOTE (Synthetic Oversampling):
While common for imbalanced datasets, SMOTE generates fake customers. In a financial/regulatory context, explaining a business decision based on a "fake" customer is unacceptable. We opted to keep the raw imbalance and adjust our business threshold tiers instead.

---

## 5. Feature Engineering Decisions

| Feature | Raw Data | Transformation Logic | Why? |
|---|---|---|---|
| `failed_payment_count` | `attempt_count` (e.g., 1, 2, 3) | `SUM(CASE WHEN attempt_count > 1 THEN 1 ELSE 0 END)` | We only care if a failure *occurred*, not the exact number of retries. Binarizing this (or capping it) prevents a single customer with 10 failed attempts from skewing the entire feature scale. |
| `tenure_months` | Individual invoice dates | `COUNT(invoices)` | In a monthly billing cycle, the number of successful invoices perfectly mirrors the number of active months. This avoids complex date-diff logic. |
| `total_spend` | `amount_paid` per invoice | `SUM(amount_paid)` | Included in the EDA matrix, but ultimately *dropped* from the final ML input list in favor of `avg_invoice_amount` to reduce the impact of extreme whale outliers on tree splits. |
```
