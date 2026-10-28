# Beginner-to-Interview Machine Learning Guide (`ChurnLens`)

This guide explains the **15 core machine learning concepts** behind **ChurnLens** in clear, everyday language tied directly to our code (`src/churnlens/`). You can study this entire guide in roughly 2 hours.

---

## 1. What Customer Churn Prediction Is
- **Concept**: Customer churn is when a subscriber cancels their service. Churn prediction uses historical customer attributes (contract type, tenure, monthly bill, add-on services) to estimate the probability $\hat{p} \in [0, 1]$ that each customer will leave.
- **Why It Matters**: It lets a retention team rank customers by risk and focus limited outreach budgets on at-risk subscribers, though verifying whether outreach actually *saves* those customers requires a separate A/B test.

## 2. Features, Labels, and Binary Classification
- **Features ($X$)**: The $19$ input columns describing each customer in `src/churnlens/data.py` (4 numeric: `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`; 15 categorical: `Contract`, `InternetService`, `TechSupport`, etc.). We drop `customerID` because an arbitrary ID string has no predictive meaning for new customers.
- **Label / Target ($y$)**: The outcome we want to predict (`Churn`). In `prepare_features_and_target`, we encode `"Yes" -> 1` (the **positive class**) and `"No" -> 0` (the **negative class**).
- **Binary Classification**: Learning a function that maps feature vector $x_i$ to a probability $\hat{p}_i = P(y_i = 1 \mid x_i)$ and a binary decision $\hat{y}_i \in \{0, 1\}$.

## 3. Training, Validation, and Test Sets
- **Why We Split Data**: If you evaluate a model on the same rows it used to learn its parameters, a complex model can memorize those rows and look artificially accurate while failing on new customers.
- **How `ChurnLens` Splits Data (`src/churnlens/train.py`)**:
  - **Training Set ($80\%$ of rows)**: Used to fit preprocessing statistics and model weights.
  - **Validation Set ($20\%$ of the training set = $16\%$ of total rows)**: Isolated from the training set to tune the probability classification threshold $\tau$ without peeking at the test set.
  - **Held-Out Test Set ($20\%$ of rows)**: Locked away until the very end and evaluated **once** to report unbiased final metrics.
- **What "Stratified" Means (`stratify=y`)**: Ensures that the training, validation, and test splits all have the exact same percentage of churners (~22–27%) as the overall dataset.

## 4. Logistic Regression Intuition & Sigmoid Probabilities
- **Intuition**: Logistic Regression (Model A) computes a weighted sum of the input features (the **log-odds** or **logit** $z$) and squashes it into a probability between $0$ and $1$ using the **sigmoid function**:
  $$z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d, \quad \hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Interpreting Weights (`coef_`)**:
  - If a feature has a **positive weight** ($w_j > 0$), increasing that feature increases predicted churn risk. In our experiment, `Contract_Month-to-month` has $w = +1.101$, meaning month-to-month contracts strongly increase churn odds.
  - If a feature has a **negative weight** ($w_j < 0$), increasing that feature decreases churn risk. For example, `tenure` has $w = -0.971$, meaning longer-tenured customers are much less likely to churn.
- **L2 Regularization (`C=1.0`)**: Penalizes large weights ($\frac{1}{2C}\sum_j w_j^2$) so no single correlated feature dominates the prediction.

## 5. Random Forest Intuition: Trees, Bagging, and Feature Subsampling
- **Decision Trees**: A tree asks a sequence of if-else questions (e.g., *"Is `Contract == Month-to-month`? If yes, is `tenure <= 6` months?"*) to split customers into purer groups (measured by **Gini impurity**). Deep single trees easily overfit (memorize) training noise.
- **Bagging (Bootstrap Aggregating)**: A Random Forest (`n_estimators=120`) trains $120$ trees in parallel. Each tree is trained on a random **bootstrap sample** (sampling rows with replacement) of the training set, and the forest averages the predicted probabilities across all $120$ trees.
- **Feature Subsampling (`max_features="sqrt"`)**: At every split inside a tree, the algorithm only considers a random subset of $\sqrt{d}$ features instead of all $d$ features. This decorrelates the trees so that one dominant feature (like `Contract`) isn't chosen at the root of every single tree.
- **Hyperparameter Controls in `ChurnLens`**: Setting `max_depth=8` and `min_samples_leaf=3` prevents individual trees from creating tiny 1-customer leaves.

## 6. Numerical Scaling & One-Hot Encoding
- **Numerical Scaling (`StandardScaler`)**: Transforms each numeric feature (`tenure`, `MonthlyCharges`, `TotalCharges`) to have mean $0$ and standard deviation $1$:
  $$x_{\text{scaled}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
  Required for regularized Logistic Regression so features with large dollar amounts (`TotalCharges`) and small integers (`tenure`) are penalized fairly.
- **One-Hot Encoding (`OneHotEncoder(handle_unknown="ignore")`)**: Converts categorical strings like `Contract` (`"Month-to-month"`, `"One year"`, `"Two year"`) into separate binary $0/1$ columns. We never assign arbitrary integers like `0, 1, 2` to unordered categories because that would falsely tell Logistic Regression that `"Two year"` is twice `"One year"`.

## 7. Missing Values & Data Leakage Prevention
- **The `TotalCharges` Quirk**: New customers (`tenure == 0`) have `" "` (a blank space) in `TotalCharges`. `clean_blank_numeric_columns` converts `" "` to `NaN`, and `SimpleImputer(strategy="median")` fills `NaN` with the **training set median**.
- **What Data Leakage Is**: Using information from the test set (or from the future/target) during training.
  - *Preprocessing Leakage*: Calling `StandardScaler().fit_transform(full_df)` before splitting leaks the test set's mean and variance into training. We prevent this by wrapping `ColumnTransformer` inside a `scikit-learn` `Pipeline` (`src/churnlens/models.py`).
  - *Target Leakage*: Leaving a column like `Churn Reason` or `Customer Status` in `X`. We explicitly strip all target-derived columns in `prepare_features_and_target`.

## 8. Class Imbalance
- **The Problem**: In telecom churn, ~73–78% of customers stay ($0$) and only ~22–27% churn ($1$). A model trained without adjustments may favor predicting the majority class (`0`).
- **Our Solution (`class_weight="balanced"`)**: Automatically weights each class inversely proportional to its frequency in the training set ($w_c = \frac{N}{2 N_c}$), penalizing missed churners ($\text{FN}$) roughly $3\times$ more than false alarms ($\text{FP}$) during training.

## 9. The Confusion Matrix
For binary churn prediction ($1 = \text{Churn}$, $0 = \text{No Churn}$):

| | **Predicted: No Churn ($0$)** | **Predicted: Churn ($1$)** |
| :--- | :--- | :--- |
| **Actual: No Churn ($0$)** | **True Negative (TN)**: Loyal customer correctly predicted to stay. | **False Positive (FP)**: Loyal customer mistakenly flagged as churn risk (*wasted retention offer*). |
| **Actual: Churn ($1$)** | **False Negative (FN)**: Churning customer missed by the model (*lost customer*). | **True Positive (TP)**: Churning customer correctly identified (*opportunity to retain*). |

## 10. Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- **Accuracy ($\frac{\text{TP} + \text{TN}}{\text{Total}}$)**: Overall percent correct. **Misleading under imbalance** because predicting $0$ for everyone gives ~75% accuracy with $0$ churners caught!
- **Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$)**: *"When the model flags a customer as a churn risk, how often is it right?"* High precision minimizes wasted outreach money on loyal customers.
- **Recall / Sensitivity ($\frac{\text{TP}}{\text{TP} + \text{FN}}$)**: *"Out of all customers who actually churned, what fraction did the model catch?"* High recall minimizes missed churners.
- **F1-Score ($\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$)**: The harmonic mean of Precision and Recall. Unlike a simple average, F1 is low if either Precision or Recall is near zero.
- **ROC-AUC (Area Under the Receiver Operating Characteristic Curve)**: Plots True Positive Rate ($\text{Recall}$) vs. False Positive Rate ($\frac{\text{FP}}{\text{FP} + \text{TN}}$) across **all possible thresholds** from $0$ to $1$. An AUC of $0.50$ is random guessing; $1.00$ is perfect ranking. Intuitively, an ROC-AUC of $0.86$ means that if you pick one churner and one non-churner at random, the model assigns a higher churn probability to the true churner $86\%$ of the time.

## 11. Classification Thresholds ($\tau$)
- By default, `.predict(X)` uses threshold $\tau = 0.50$: if $\hat{p} \ge 0.50$, predict `Churn (1)`.
- **Lowering $\tau$ (e.g., from $0.50$ to $0.38$)**: Flags more customers as at-risk $\to$ **Recall goes up**, **Precision goes down**.
- **Raising $\tau$ (e.g., from $0.50$ to $0.65$)**: Only flags the most certain churners $\to$ **Precision goes up**, **Recall goes down**.
- In `src/churnlens/train.py`, `select_threshold_on_validation` sweeps thresholds on the **validation set** (never the test set!) and reports both the default $0.50$ result and the validation-selected result.

## 12. Overfitting and Generalization
- **Overfitting**: When a model fits noise specific to the training set (e.g., an unpruned decision tree with `max_depth=None` achieving $100\%$ training accuracy but poor test ROC-AUC).
- **How We Control It**: $L_2$ regularization (`C=1.0`) in Logistic Regression, and depth/leaf limits (`max_depth=8, min_samples_leaf=3`) plus bootstrap feature subsampling (`max_features="sqrt"`) in Random Forest.

## 13. Feature Importance and Its Limitations
- **Impurity (Gini) Importance**: Measures how much each feature reduces node impurity across all trees in the Random Forest during training. *Caveat*: Biased toward continuous numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) because they have more split points than binary one-hot columns.
- **Permutation Importance (`sklearn.inspection.permutation_importance`)**: Randomly shuffles one column of the held-out test set at a time and measures how much test ROC-AUC drops. In our experiment, permutation importance identifies `Contract` as the #1 feature for Random Forest (`-0.1124` ROC-AUC drop when shuffled).
- **Crucial Limitation**: Feature importance shows what the model relies on for prediction (**association**), not **causation**.

## 14. Reproducibility and Random Seeds
- Setting `random_seed: 42` in `configs/default.yaml` fixes the stratified `train_test_split`, the `RandomForestClassifier` bootstrap and feature sampling RNG, and the `permutation_importance` shuffles so every run produces identical tables and plots.

## 15. Business Interpretation: Probability Ranking & Lift
- Instead of treating churn prediction as an all-or-nothing binary label, a retention team with a budget to call $20\%$ of customers can sort subscribers by $\hat{p}_i$ descending (`build_illustrative_retention_plan` in `src/churnlens/evaluate.py`).
- In our verification run, contacting the **Top 20% highest-risk customers** captures **$57.7\%$ of all actual churners** ($2.88\times$ lift over random outreach).
