# Limitations & Responsible-Use Considerations (`ChurnLens`)

Understanding what a machine learning pipeline **cannot** do is essential for engineering integrity and technical interviews.

## 1. Predictive Association vs. Causal Impact

Both Logistic Regression coefficients and Random Forest / permutation feature importances measure **statistical association** within historical observational data—not causation.
- For example, observing that `InternetService == "Fiber optic"` or `PaymentMethod == "Electronic check"` is positively associated with churn does **not** mean that switching a customer's payment method or downgrading their internet speed will cause them to stay. Those variables often act as proxies for unobserved factors (such as competitor fiber promotions or regional service outages).

## 2. Prediction vs. Retention Uplift

A churn model predicts $P(\text{Churn} = 1 \mid X)$, the baseline risk of a customer leaving.
- In real business operations, the highest-risk customers ($P > 0.85$, sometimes called "sure things" or "lost causes") may have already decided to leave regardless of any discount, while certain medium-risk customers ("persuadables") might respond strongly to an outreach call.
- Worse, contacting some customers who were quietly paying month-to-month ("sleeping dogs") can remind them to cancel.
- Measuring true business ROI requires a **randomized controlled trial (A/B test)** or an **uplift model** estimating $P(\text{Churn} \mid \text{Outreach}) - P(\text{Churn} \mid \text{No Outreach})$.

## 3. Single Cross-Sectional Snapshot (No Temporal Dynamics)

The IBM Telco Customer Churn dataset is a static cross-sectional snapshot without timestamps, longitudinal usage logs, customer support call transcripts, or billing history trajectories.
- In production telecom environments, features evolve over time (e.g., a sudden 40% drop in data usage over the last 30 days or three unresolved support tickets), and models must be evaluated using **temporal (out-of-time) splits** rather than random stratified splits to prevent temporal leakage.

## 4. Synthetic Fixture vs. Real IBM Telco Dataset

When outbound network access is blocked (such as in isolated CI containers or offline build sandboxes), `ChurnLens` supports running against `generate_synthetic_telco_fixture` (`--use-synthetic`).
- While the synthetic fixture preserves the 21-column schema, blank-string `TotalCharges` edge case, and ~22–27% churn rate to verify pipeline correctness, its feature-target relationships are generated from a known parametric log-odds formula.
- Therefore, synthetic fixture metrics (`data_source: "synthetic_fixture"`) must **never** be presented as empirical findings on the real `7,043`-row IBM Telco dataset (`data_source: "ibm_telco_csv"`).

## 5. Fairness & Demographic Sensitivity

The dataset includes demographic attributes such as `gender`, `SeniorCitizen`, `Partner`, and `Dependents`. Before using any churn score to allocate financial discounts or service tiers in the real world, teams must audit false-positive and false-negative rates across demographic subgroups to ensure that automated retention policies do not unfairly exclude protected groups from beneficial offers.
