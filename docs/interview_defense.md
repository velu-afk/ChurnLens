# Interview Defense Guide & 5-Hour Study Plan (`ChurnLens`)

Every answer in this guide reflects the exact code in `src/churnlens/` and the verified experimental outputs in `results/`.

---

## 1. Truthful 30-Second Project Pitch

> "I built **ChurnLens**, an end-to-end Python and `scikit-learn` customer churn prediction pipeline modeled on the 21-column IBM Telco Customer Churn schema. It compares an interpretable $L_2$-regularized Logistic Regression baseline against a depth-constrained Random Forest inside a leakage-safe `ColumnTransformer` and `Pipeline`. I implemented automated data cleaning for whitespace-corrupted numeric fields (`TotalCharges`), a stratified 80/20 train/test split with an isolated validation split for threshold tuning, held-out permutation importance, and an illustrative risk-tier prioritization analysis, backed by 11 offline `pytest` unit tests and CI."

---

## 2. Two-Minute Technical Walkthrough

> "In subscription telecom businesses, predicting which customers are likely to cancel (`Churn == Yes`, encoded as `1`) helps retention teams prioritize outreach. However, two common mistakes ruin student churn projects: data leakage from preprocessing before splitting, and relying on raw accuracy under ~22–27% class imbalance.
>
> To solve this cleanly in **ChurnLens**:
> 1. **Data Cleaning & Leakage Guards (`src/churnlens/data.py`)**: Raw Telco data stores `TotalCharges` as strings because new customers (`tenure == 0`) have blank spaces (`" "`). My loader converts blank strings to `NaN` without computing global medians, strips `customerID` and any target-leakage columns (`Churn Score`, `Churn Reason`), and encodes `Churn` (`Yes -> 1`, `No -> 0`).
> 2. **Stratified Three-Way Split (`src/churnlens/train.py`)**: I first isolate a stratified 20% held-out test set (`random_state=42`). From the remaining 80% training data, I carve out a stratified 20% validation split strictly to tune the classification threshold $\tau$ for positive-class F1, keeping the test set completely untouched.
> 3. **Leakage-Safe Pipelines (`src/churnlens/preprocessing.py` & `models.py`)**: I use a `ColumnTransformer` that applies `SimpleImputer(strategy='median')` and `StandardScaler()` to the 4 numeric columns, and `SimpleImputer(strategy='most_frequent')` plus `OneHotEncoder(handle_unknown='ignore')` to the 15 categorical columns. This transformer is wrapped inside a `Pipeline` with either `LogisticRegression(class_weight='balanced')` or `RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=3, class_weight='balanced')` so all statistics are learned strictly on training folds.
> 4. **Evaluation & Business Translation (`src/churnlens/evaluate.py`)**: On the held-out test split, I evaluate Accuracy, Precision, Recall, F1, ROC-AUC, and confusion matrices at both $\tau = 0.50$ and the validation-tuned threshold. I also compute held-out permutation importance (`scoring='roc_auc'`) alongside Random Forest Gini importance to correct for high-cardinality numeric bias, and rank test customers into top-10%, top-20%, and top-30% risk tiers to show lift over random outreach."

---

## 3. The End-to-End Pipeline in Plain English

1. **Read & Validate**: Load the 21-column CSV (or synthetic fixture when offline), coerce `" "` in `TotalCharges` to `NaN`, remove `customerID` and `Churn` from `X`, and encode `Churn` as `1` (`Yes`) / `0` (`No`).
2. **Split**: Hold out 20% of customers via stratified sampling for final testing; split the remaining 80% into subtrain (64%) and validation (16%).
3. **Fit & Tune Thresholds**: Fit both pipelines on subtrain, sweep probability thresholds $\tau \in [0.20, 0.80]$ on validation to maximize F1, then refit both pipelines on the full 80% training set.
4. **Evaluate & Explain**: Score the 20% test set once, compute metrics at $\tau = 0.50$ and the validation-selected threshold, run held-out permutation importance, and save CSV/Markdown tables and 5 matplotlib plots.

---

## 4. 24 Likely Interview Questions & Concise Technical Answers

1. **Q: Why did you choose Logistic Regression and Random Forest instead of jumping straight to deep learning or XGBoost?**
   **A:** On tabular customer data with 19 features, regularized Logistic Regression provides a fast, well-calibrated linear baseline with directly interpretable log-odds coefficients, while a depth-controlled Random Forest captures nonlinear feature interactions (like month-to-month contracts combined with short tenure) without requiring heavy hyperparameter tuning. Comparing a linear model against a tree ensemble directly tests whether nonlinear interactions improve ranking (ROC-AUC).

2. **Q: What happens to `TotalCharges` in the IBM Telco dataset if you load it with `pd.read_csv()` directly?**
   **A:** Because rows where `tenure == 0` contain a single whitespace character `" "`, pandas parses the entire `TotalCharges` column as `object` (string). If you don't convert it via `pd.to_numeric(..., errors="coerce")`, `ColumnTransformer` will treat `TotalCharges` as a categorical column and create thousands of useless one-hot columns!

3. **Q: Why didn't you fill the `NaN` values in `TotalCharges` inside `clean_blank_numeric_columns()` before splitting the data?**
   **A:** Computing the median of `TotalCharges` across the entire DataFrame before `train_test_split` leaks test-set distribution statistics into the training set. Instead, `clean_blank_numeric_columns` only converts `" "` to `np.nan`, and `SimpleImputer(strategy="median")` inside the `Pipeline` learns the median strictly from the training split.

4. **Q: Why do you drop `customerID` before training?**
   **A:** `customerID` is a unique row identifier with zero generalizable relationship to churn. Leaving it in `X` would cause `OneHotEncoder` to create one indicator column per customer in the training set, bloating dimensionality and causing overfitting.

5. **Q: Why is Accuracy a poor primary metric for customer churn?**
   **A:** Because churn is imbalanced (~22–27% positive class). A trivial model that predicts `"No Churn" (0)` for every single customer achieves ~73–78% accuracy while catching 0% of actual churners ($\text{Recall} = 0$, $\text{F1} = 0$).

6. **Q: In churn prediction, what is the business difference between a False Positive (FP) and a False Negative (FN)?**
   **A:** A **False Positive** is a loyal customer mistakenly flagged as about to churn—if the business sends them a $15 retention discount, that outreach cost is wasted. A **False Negative** is a truly churning customer whom the model misses—the business fails to intervene and loses the customer's recurring subscription revenue.

7. **Q: When would a retention team prefer higher Recall over higher Precision, and vice versa?**
   **A:** If the intervention is a cheap automated email or low-cost check-in call and losing a customer is very expensive (high customer lifetime value), the team prefers **high Recall** (lower threshold $\tau$). If the intervention is an expensive account credit or human specialist call with a strict budget, the team prefers **high Precision** (higher threshold $\tau$ or top-$K\%$ ranking).

8. **Q: How did you choose the alternative classification threshold without leaking test data?**
   **A:** Inside `train_models_and_tune_thresholds` (`src/churnlens/train.py`), I split the 80% training set into 64% subtrain and 16% validation. I fit the pipeline on subtrain, evaluated 25 candidate thresholds in $[0.20, 0.80]$ on the validation set (`select_threshold_on_validation`), locked the threshold with the highest validation F1, and then applied that locked threshold to the test set.

9. **Q: Why did you report both the default 0.50 threshold and the validation-tuned threshold?**
   **A:** Because both models use `class_weight="balanced"`, their raw predicted probabilities are already shifted upward toward the minority class. Reporting both $\tau = 0.50$ and the validation-selected $\tau$ shows transparently how threshold tuning trades precision for recall on held-out data.

10. **Q: What did the actual model comparison show on the executed synthetic verification benchmark (`n=600`, test size `120`)?**
    **A:** At the default $\tau = 0.50$ threshold:
    - **Logistic Regression** achieved **0.8457 ROC-AUC**, **0.6154 F1**, **0.9231 Recall** (`24/26` churners caught), **0.4615 Precision**, and **0.7500 Accuracy**.
    - **Random Forest** achieved a higher **0.8629 ROC-AUC**, **0.6122 F1**, higher **0.6522 Precision** (`15/23` flagged were true churners), **0.5769 Recall** (`15/26` caught), and **0.8417 Accuracy**.
    - When applying the validation-tuned threshold ($\tau = 0.38$) to Random Forest, its **Recall increased from 0.5769 to 0.7692** (`20/26` churners caught) while Precision adjusted to **0.4444**.

11. **Q: Why does `LogisticRegression(class_weight="balanced")` have 92.3% recall at $\tau = 0.50$ while `RandomForestClassifier(class_weight="balanced")` has 57.7% recall at $\tau = 0.50$?**
    **A:** In Logistic Regression, weighting positive examples by $\frac{N}{2 N_1}$ directly shifts the linear intercept $b$ upward so any customer with slightly above-average risk gets $\hat{p} > 0.50$. In a Random Forest of depth-8 trees, leaf probabilities are averaged across 120 bootstrap trees, which pulls extreme probabilities back toward the center and makes $\hat{p} \ge 0.50$ a much stricter cutoff.

12. **Q: Why did you use `StandardScaler` on numeric features when Random Forest is invariant to monotonic feature scaling?**
    **A:** While tree splits in Random Forest are invariant to monotonic linear scaling, Logistic Regression with $L_2$ regularization (`C=1.0`) is highly sensitive to feature scale. Using a unified `ColumnTransformer` factory (`build_preprocessor`) ensures consistent preprocessing across both pipelines without hurting Random Forest.

13. **Q: Why did you set `handle_unknown="ignore"` in `OneHotEncoder`?**
    **A:** If a new category appears in the test split or during production scoring (`churnlens predict`) that was never seen in `X_train`, `handle_unknown="error"` (the default) crashes with a `ValueError`. Setting `handle_unknown="ignore"` encodes unseen categories as a vector of all zeros so inference remains resilient.

14. **Q: Why did you include held-out Permutation Importance alongside Random Forest's built-in Gini importance?**
    **A:** Random Forest's Gini importance (`feature_importances_`) is computed on training splits and is biased toward continuous numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) because continuous variables offer many more split points than binary one-hot dummies. Permutation importance (`sklearn.inspection.permutation_importance`) shuffles each raw feature column on the **held-out test set** and measures the actual drop in test ROC-AUC.

15. **Q: Did Permutation Importance and Gini Importance differ in your actual experiment?**
    **A:** Yes! In our verification run, Random Forest's internal Gini importance ranked continuous numeric features highest (`num__tenure`: `0.1303`, `num__MonthlyCharges`: `0.1000`, `num__TotalCharges`: `0.0923`) ahead of `cat__Contract_Month-to-month` (`0.0727`). However, held-out **permutation importance** on test ROC-AUC ranked **`Contract` #1 (`0.1124` AUC drop)**, ahead of `tenure` (`0.0264`) and `MonthlyCharges` (`0.0200`).

16. **Q: Which features had the strongest positive and negative coefficients in Logistic Regression?**
    **A:** The strongest positive churn drivers were `cat__Contract_Month-to-month` ($+1.101$) and `num__MonthlyCharges` ($+0.968$). The strongest protective (negative) drivers were `num__tenure` ($-0.971$), `cat__Contract_Two year` ($-0.942$), and `cat__OnlineSecurity_Yes` ($-0.502$).

17. **Q: Does `Contract_Month-to-month` having the highest importance prove that forcing customers onto two-year contracts will reduce churn?**
    **A:** No. Feature importance and regression coefficients measure **predictive association**, not causation. Customers who self-select into two-year contracts are already more committed to the service; forcing an unhappy customer into a long contract could cause them to refuse sign-up altogether.

18. **Q: How did you control overfitting in the Random Forest model?**
    **A:** Instead of letting trees grow until every leaf is pure (`max_depth=None, min_samples_leaf=1`), I configured `max_depth=8`, `min_samples_split=6`, `min_samples_leaf=3`, and `max_features="sqrt"` across `n_estimators=120` trees.

19. **Q: How does `build_illustrative_retention_plan` translate model probabilities into a business workflow?**
    **A:** It ranks all test-set customers from highest to lowest predicted churn probability $\hat{p}_i$ and evaluates contacting the Top 10%, Top 20%, and Top 30% highest-risk customers. In our test run, contacting the **Top 20% highest-risk tier** captured **57.69% of all true churners** (`15` of `26`), achieving a **2.88x lift** over random outreach.

20. **Q: Why did you label the outreach cost ($15/customer) and retention tiers as "illustrative"?**
    **A:** Because the dataset contains no data on intervention costs, customer lifetime value, or whether contacted customers actually accept a retention offer. Claiming real dollar savings from an offline classification dataset would be scientifically dishonest.

21. **Q: How do your automated unit tests verify that preprocessing does not leak test-set statistics?**
    **A:** In `tests/test_preprocessing.py` (`test_preprocessor_fits_exclusively_on_training_statistics`), `X_train` has `tenure = [10, 20, 30, NaN]` (training median = `20.0`, training mean = `20.0`), while `X_test` has `tenure = [1000.0, NaN]`. After fitting on `X_train` and transforming `X_test`, the test verifies that the missing value in `X_test` is imputed as `20.0` (scaling to `0.0`), proving the `1000.0` test outlier never influenced the imputer or scaler!

22. **Q: What was executed in the build environment vs. what requires local execution?**
    **A:** All source modules, CLI commands, 11 `pytest` unit tests, Ruff lint checks, and the full 600-row synthetic Telco verification experiment (`results/synthetic_verification/`) were executed and verified in the build environment. Because the build sandbox had no outbound internet access to download the 7,043-row CSV from GitHub, the real-data download (`python scripts/download_data.py`) is marked as blocked in the sandbox and ready to run with a single command on a networked machine.

23. **Q: How can a user score new customer records using a trained model?**
    **A:** By running `python -m churnlens predict --model-path results/models/random_forest.joblib --input-csv path/to/new_customers.csv --output-csv results/scored_customers.csv`, which loads the serialized `Pipeline` (including fitted imputers, scalers, and encoders) and outputs `predicted_churn_probability` and `predicted_churn_label`.

24. **Q: Why is `ROC-AUC` invariant to the classification threshold $\tau$?**
    **A:** Because `ROC-AUC` integrates the True Positive Rate against the False Positive Rate across *all* possible thresholds $\tau \in (-\infty, +\infty)$. Changing $\tau$ from `0.50` to `0.38` moves you to a different operating point $(FPR, TPR)$ along the exact same ROC curve, leaving the area under the curve unchanged (`0.8457` for Logistic Regression and `0.8629` for Random Forest).

---

## 5. 12 Challenging Follow-Up Questions (Deep-Dive Defense)

1. **Follow-up: Why didn't you use `drop="first"` in `OneHotEncoder` to avoid multicollinearity (the dummy variable trap) in Logistic Regression?**
   - **Defense**: In unregularized Ordinary Least Squares or unpenalized logistic regression, collinear dummy columns ($x_1 + x_2 + x_3 = 1$) make the design matrix singular because they sum to the intercept column. However, our `LogisticRegression` uses **$L_2$ regularization (`penalty="l2", C=1.0`)**, which strictly convexifies the objective ($\lambda \sum_j w_j^2$) and uniquely determines every category's weight while allowing `handle_unknown="ignore"` to map unseen categories cleanly to all-zeros.

2. **Follow-up: Why does validation-tuned threshold selection sometimes yield a slightly lower F1 on the test set than on the validation set?**
   - **Defense**: With finite sample sizes (e.g., $96$ validation rows and $120$ test rows in the synthetic fixture), the empirical Precision-Recall curve is a step function with sample variance. The threshold $\tau^* = 0.38$ that maximizes F1 on the validation split is an estimate subject to finite-sample variance when applied to the independent test split. Still, isolating threshold selection on validation is mandatory because picking $\tau^*$ on the test set would be test-set overfitting.

3. **Follow-up: If `TotalCharges` is approximately `tenure * MonthlyCharges`, does including all three columns hurt Logistic Regression or Random Forest?**
   - **Defense**: Because `TotalCharges` is a nonlinear product interaction (`tenure * MonthlyCharges`), including it gives linear Logistic Regression access to a bilinear interaction term it couldn't otherwise represent! While `TotalCharges` is correlated with `tenure`, $L_2$ regularization prevents coefficient blow-up, and in Random Forest, correlated features share Gini importance (which is why held-out permutation importance is used alongside Gini importance).

4. **Follow-up: What is the difference between `ROC-AUC` and `PR-AUC` (Average Precision), and when is `PR-AUC` better?**
   - **Defense**: `ROC-AUC` uses False Positive Rate ($\frac{\text{FP}}{\text{FP} + \text{TN}}$) on the x-axis. When the negative class ($\text{TN}$) is huge (e.g., $99\%$ non-churners), a large spike in False Positives barely changes FPR, making `ROC-AUC` look overly optimistic. `PR-AUC` plots Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) vs. Recall and does not use $\text{TN}$ at all, making it more sensitive to false alarms under severe class imbalance.

5. **Follow-up: How does `class_weight="balanced"` affect probability calibration?**
   - **Defense**: Upweighting the positive minority class by $\frac{N}{2 N_1}$ shifts predicted probabilities upward toward $0.50$ (approximating a balanced $50/50$ prior rather than the true ~25% base rate). This preserves the **ranking order** of customers (so `ROC-AUC` and top-$K\%$ lift are unaffected), but if exact calibrated probabilities matching the empirical base rate are needed, one would wrap the model in `CalibratedClassifierCV` (Platt scaling or isotonic regression).

6. **Follow-up: Why did you use `SimpleImputer(strategy="median")` instead of `"mean"` for numeric features?**
   - **Defense**: Billing and tenure distributions (`TotalCharges`) are right-skewed. The sample median is robust to high-spending outliers, whereas the sample mean is pulled upward by long-tenured high-bill subscribers.

7. **Follow-up: What happens if a feature is strongly correlated with another feature when computing Permutation Importance?**
   - **Defense**: If two features carry duplicate signal (such as `tenure` and `TotalCharges`), shuffling `tenure` alone still leaves `TotalCharges` intact for the model to use, which can make both features appear slightly less important individually than they are jointly.

8. **Follow-up: Why is a random stratified split potentially optimistic for real-world production churn deployment?**
   - **Defense**: If customer behavior shifts over time (concept drift, seasonal promotions, price hikes), a random split mixes customers from the same time period across train and test. In a production database with signup and cancellation timestamps, we would use an **out-of-time temporal split** (e.g., train on Q1–Q3 cohorts, validate on Q4, test on Q1 of the next year).

9. **Follow-up: How would you turn this batch prediction pipeline into a real-world retention experiment?**
   - **Defense**: Score active customers weekly, take the top 20% highest-risk tier, and randomly assign 50% of that tier to receive the retention outreach (treatment group) and 50% to receive no special outreach (control group). Comparing the 30-day churn rate between treatment and control measures the true **causal uplift** of the retention campaign.

10. **Follow-up: Why did you serialize the entire `Pipeline` with `joblib` instead of saving only the classifier weights?**
    - **Defense**: Saving only `classifier` weights without the fitted `ColumnTransformer` means you cannot score raw customer records at inference time without recomputing medians, means, standard deviations, and one-hot category mappings—which risks train-serve skew. Saving the unified `Pipeline` guarantees identical transformations during `churnlens predict`.

11. **Follow-up: Are `.joblib` / `.pkl` files safe to load from untrusted internet sources?**
    - **Defense**: No—`joblib` uses Python's `pickle` module under the hood, which can execute arbitrary code during deserialization. Only load `.joblib` artifacts generated locally by your own trusted pipeline.

12. **Follow-up: If you had 2 more hours to extend the modeling scope on the real 7,043-row IBM Telco dataset, what would you add?**
    - **Defense**: I would add 5-fold `StratifiedKFold` cross-validation on the 80% training split to report $\text{mean} \pm \text{std}$ validation ROC-AUC and F1 before evaluating on the held-out 20% test split, and include a reliability diagram (calibration curve) comparing predicted probabilities against empirical churn frequencies.

---

## 6. Resume Bullet Walkthrough & Verified Claims

### Three Verified, ATS-Friendly Resume Bullets (Supported Immediately by This Repository)

1. **Built an end-to-end customer churn prediction pipeline in Python and scikit-learn (`ChurnLens`)** comparing $L_2$-regularized Logistic Regression and Random Forest classifiers across 19 customer demographic, service, and billing features using leakage-safe `ColumnTransformer` and `Pipeline` architectures.
2. **Engineered automated data-cleaning, stratified train/validation/test isolation, and threshold tuning workflows**, coercing whitespace-corrupted numeric fields, stripping identifier/target-leakage columns, and isolating threshold selection on a validation split backed by 11 offline `pytest` unit tests and GitHub Actions CI.
3. **Implemented model explainability and retention-prioritization evaluation modules**, combining held-out ROC-AUC permutation importance with impurity/coefficient analysis and probability-ranked outreach lift tables to quantify precision-recall trade-offs under class imbalance.

### Metrics to Plug In After Running `python scripts/download_data.py && python scripts/run_experiment.py` on Your Laptop
Once you run the two commands above on your networked machine with the 7,043-row IBM Telco CSV, open `results/model_comparison.md` and `results/evaluation_metrics.json` and insert the exact real-dataset numbers into Bullet 1 or Bullet 3:
- *Example template*: `"...achieving [X.XX] ROC-AUC and [X.XX] Recall on the 7,043-record IBM Telco dataset, capturing [XX.X]% of churning customers within the top 20% risk tier ([X.XX]x lift)."`
- *(Note: On our 600-row offline synthetic verification fixture, Random Forest achieved `0.8629` ROC-AUC and `0.7692` validation-tuned Recall, while Logistic Regression achieved `0.8457` ROC-AUC and `0.9231` Recall, with both models achieving `2.88x` lift in the top 20% risk tier).*

---

## 7. Prioritized 5-Hour / Two-Day Study Schedule

| Session | Time Budget | Files to Inspect | Core Focus & Practical Check |
| :--- | :--- | :--- | :--- |
| **Day 1 — Part 1: Data Hygiene & Leakage Prevention** | **1.0 hr** | `src/churnlens/data.py`, `src/churnlens/preprocessing.py`, `tests/test_preprocessing.py` | Understand why `" "` in `TotalCharges` becomes `NaN`, why `customerID` is dropped, and how `ColumnTransformer` inside `Pipeline` prevents test-set leakage. Run `python -m pytest tests/test_data.py tests/test_preprocessing.py -v`. |
| **Day 1 — Part 2: Logistic Regression vs. Random Forest** | **1.5 hrs** | `src/churnlens/models.py`, `src/churnlens/train.py`, `docs/interview_guide.md` (Sections 4–8) | Understand sigmoid log-odds, $L_2$ regularization (`StandardScaler`), tree bagging, `max_depth=8`, `class_weight="balanced"`, and the 3-way stratified split (`subtrain`, `val`, `test`). |
| **Day 2 — Part 1: Metrics, Thresholds & Importance** | **1.5 hrs** | `src/churnlens/evaluate.py`, `results/synthetic_verification/model_comparison.md`, `docs/interview_guide.md` (Sections 9–15) | Master the confusion matrix (`TP, FP, FN, TN`), why accuracy fails under imbalance, how lowering $\tau$ from `0.50` to `0.38` trades precision for recall, and why permutation importance beats Gini importance. |
| **Day 2 — Part 2: Mock Interview Defense** | **1.0 hr** | `docs/interview_defense.md`, `docs/limitations.md` | Practice the 30-second pitch, 2-minute walkthrough, and answer 15 questions aloud without looking at notes. Run `python scripts/run_experiment.py --use-synthetic` and explain every generated plot in `results/`. |
