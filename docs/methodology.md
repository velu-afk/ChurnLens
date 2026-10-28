# Experimental Methodology (`ChurnLens`)

## 1. Problem Formulation

Customer churn occurs when an existing subscriber cancels or fails to renew their service. In subscription telecommunications businesses, acquiring a new customer typically costs several times more than retaining an existing one. However, contacting every customer with a retention discount wastes budget on satisfied subscribers.

**ChurnLens** formulates churn prediction as a supervised **binary classification and probability-ranking problem**:
- **Input ($x_i$)**: $19$ historical customer demographic, service subscription, contract, and billing features.
- **Target ($y_i \in \{0, 1\}$)**:
  - $y_i = 1$ (`Churn == "Yes"`, **explicitly documented positive class**)
  - $y_i = 0$ (`Churn == "No"`, negative class)
- **Model Output ($\hat{p}_i = P(y_i = 1 \mid x_i) \in [0, 1]$)**: Estimated churn probability used both for binary classification at a chosen threshold $\tau$ ($\hat{y}_i = \mathbb{I}(\hat{p}_i \ge \tau)$) and for ranking customers by risk tier.

> **Important Scope Boundary**: Predicting churn probability $\hat{p}_i$ identifies *who is at risk of leaving*; it does not prove that a specific retention offer will prevent them from churning. Measuring real-world churn reduction or revenue impact requires a separate randomized controlled intervention (uplift) study.

---

## 2. Data Hygiene & Leakage Prevention Protocol

### 2.1 Identifier & Target Leakage Removal (`src/churnlens/data.py`)
Before splitting or modeling, `prepare_features_and_target` removes:
1. **Unique Customer Identifiers**: `customerID` (and any column matching `*_id`), which carry zero generalizable signal and can cause memorization if left in tree splits or one-hot encodings.
2. **Target & Post-Hoc Leakage Columns**: `Churn` as well as any extended Telco columns that directly encode the outcome (`Churn Value`, `Churn Score`, `Churn Reason`, `Customer Status`). A runtime guard (`if target_column in X.columns: raise RuntimeError`) guarantees the target is never included in `X`.

### 2.2 Handling Blank Numeric Strings (`TotalCharges`)
In the IBM Telco Customer Churn dataset (`7,043` rows), `11` rows corresponding to new customers (`tenure == 0`) contain a single whitespace string `" "` in `TotalCharges`, causing pandas to parse the entire column as `object` (string).
- `clean_blank_numeric_columns` uses `pd.to_numeric(..., errors="coerce")` to convert `" "` into `np.nan` so `TotalCharges` has a valid `float64` dtype.
- Crucial leakage safeguard: `clean_blank_numeric_columns` **does not compute medians or fill missing values on the full dataset**. Imputation happens strictly inside the `SimpleImputer(strategy="median")` step of the `scikit-learn` `ColumnTransformer` fitted only on training data.

### 2.3 Three-Way Stratified Partition (`src/churnlens/train.py`)
To evaluate models fairly and tune classification thresholds without touching the final test set:
1. **Held-Out Test Split ($20\%$ of total rows)**: Created via `train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)`. Kept completely untouched during preprocessing, training, and threshold selection.
2. **Training Split ($80\%$ of total rows)**: Further partitioned via stratified `train_test_split(..., test_size=0.20, stratify=y_train_full, random_state=42)` into:
   - **Subtrain ($64\%$ of total rows)**: Used to fit candidate pipelines during threshold tuning.
   - **Validation ($16\%$ of total rows)**: Used exclusively by `select_threshold_on_validation` to evaluate thresholds $\tau \in [0.20, 0.80]$ and pick the threshold maximizing positive-class F1-score.
3. **Final Training Fit**: Once the validation threshold is locked, the final pipelines are fitted on the full $80\%$ training split (`X_train_full, y_train_full`) and evaluated once on `X_test, y_test`.

---

## 3. Baseline Models Compared

### Model A: L2-Regularized Logistic Regression (`LogisticRegression`)
- **Role**: Linear, calibrated, interpretable baseline.
- **Pipeline**:
  - Numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`): `SimpleImputer(strategy="median") -> StandardScaler()`.
  - Categorical features ($15$ columns): `SimpleImputer(strategy="most_frequent") -> OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
  - Classifier: `LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)`.
- **Why `StandardScaler` is essential**: Without zero-mean, unit-variance scaling, features with large numerical ranges (`TotalCharges` up to $8,684$) experience disproportionate $L_2$ penalty updates compared with binary indicators (`0`/`1`), slowing convergence and distorting coefficient magnitudes.

### Model B: Regularized Random Forest (`RandomForestClassifier`)
- **Role**: Nonlinear ensemble capable of capturing feature interactions (e.g., `Contract == "Month-to-month"` combined with short `tenure` and high `MonthlyCharges`).
- **Pipeline**: Uses the same `ColumnTransformer` wrapper paired with `RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_split=6, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", random_state=42)`.
- **Why depth and leaf regularization matter**: Unconstrained decision trees (`max_depth=None, min_samples_leaf=1`) memorize training noise and produce poorly calibrated 0/1 leaf probabilities. Constraining `max_depth=8` and `min_samples_leaf=3` prevents overfitting.

---

## 4. Evaluation Metrics & Why Accuracy Alone Is Misleading

In customer churn datasets, only ~22–27% of customers churn ($y = 1$), while ~73–78% stay ($y = 0$).
- **The Accuracy Paradox**: A trivial dummy classifier that predicts `"No Churn" (0)` for every single customer achieves **73.5%–78.2% accuracy** while catching **0% of actual churners** ($\text{Recall} = 0.0$, $\text{F1} = 0.0$).
- Therefore, `src/churnlens/evaluate.py` reports:
  1. **Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$)**: Fraction of customers flagged as churn risks who actually churn (controls wasted retention budget).
  2. **Recall ($\frac{\text{TP}}{\text{TP} + \text{FN}}$)**: Fraction of all true churners successfully identified (controls missed churners).
  3. **F1-Score ($\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$)**: Harmonic mean balancing precision and recall.
  4. **ROC-AUC**: Threshold-independent probability that the model ranks a randomly chosen churner higher than a randomly chosen non-churner.
  5. **Confusion Matrix ($\text{TN}, \text{FP}, \text{FN}, \text{TP}$)**: Explicit counts of correct and incorrect decisions at both the default $0.50$ threshold and the validation-tuned threshold.

---

## 5. Feature Importance: Impurity vs. Permutation Importance

Random Forest's built-in Gini impurity importance (`feature_importances_`) is computed on training splits and is known to inflate the importance of continuous numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) because continuous variables offer many more candidate split thresholds than binary one-hot categories.

To provide a more reliable assessment, `extract_feature_importances` in `src/churnlens/evaluate.py` computes **held-out permutation importance** on `X_test, y_test` (`scoring="roc_auc", n_repeats=5`), measuring how much test ROC-AUC drops when each raw input feature column is randomly shuffled. Both methods measure **predictive association**, not causal impact.
