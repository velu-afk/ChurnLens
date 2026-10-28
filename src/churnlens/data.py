"""Data loading, schema inspection, blank-string cleaning, and leakage prevention."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import expit

from churnlens.config import DataConfig

KNOWN_NUMERIC_CANDIDATES = ("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen")


def generate_synthetic_telco_fixture(
    n_samples: int = 600,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic 21-column synthetic Telco customer churn fixture for offline testing.

    Important Integrity Note:
        This fixture mirrors the exact 21-column schema, data types, blank-string `TotalCharges`
        edge cases (`" "`), and ~26.5% positive churn rate of the IBM Telco Customer Churn
        dataset so unit tests and offline smoke verifications can run without internet access.
        Outputs generated from this fixture are explicitly tagged `"synthetic_fixture"` and must
        never be reported as measurements from the real IBM Telco CSV.
    """
    if n_samples < 20:
        raise ValueError(f"n_samples must be >= 20, got {n_samples}.")

    rng = np.random.default_rng(int(random_state))

    customer_ids = [f"{1000 + i:04d}-SYNTH" for i in range(n_samples)]
    gender = rng.choice(["Female", "Male"], size=n_samples)
    senior = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])

    # Tenure in [0, 72] months, with a few 0-month new customers that have blank TotalCharges
    tenure = rng.integers(0, 73, size=n_samples)
    tenure[:4] = 0

    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["No", "Yes"], size=n_samples, p=[0.53, 0.47]),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22]
    )

    def _internet_addon(prob_yes: float) -> np.ndarray:
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["No", "Yes"], size=n_samples, p=[1.0 - prob_yes, prob_yes]),
        )

    online_security = _internet_addon(0.29)
    online_backup = _internet_addon(0.34)
    device_protection = _internet_addon(0.34)
    tech_support = _internet_addon(0.29)
    streaming_tv = _internet_addon(0.38)
    streaming_movies = _internet_addon(0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.55, 0.21, 0.24],
    )
    paperless = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n_samples,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    base_charge = np.where(
        internet_service == "No",
        20.0,
        np.where(internet_service == "DSL", 50.0, 78.0),
    )
    addon_count = (
        (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
    )
    monthly_charges = np.round(
        np.clip(base_charge + 5.0 * addon_count + rng.normal(0.0, 4.5, size=n_samples), 18.25, 118.75),
        2,
    )

    total_charges_num = np.round(
        np.maximum(0.0, tenure * monthly_charges + rng.normal(0.0, 15.0, size=n_samples)),
        2,
    )
    # Represent TotalCharges as strings with whitespace " " when tenure == 0 (mirrors IBM Telco CSV)
    total_charges_str = [
        " " if int(t) == 0 else f"{val:.2f}"
        for t, val in zip(tenure, total_charges_num)
    ]

    # Ground-truth latent log-odds for realistic churn behavior (~26-28% positive rate)
    logit = (
        -1.35
        + 1.25 * (contract == "Month-to-month").astype(float)
        - 0.95 * (contract == "Two year").astype(float)
        + 0.75 * (internet_service == "Fiber optic").astype(float)
        - 0.65 * (internet_service == "No").astype(float)
        + 0.45 * (payment_method == "Electronic check").astype(float)
        - 0.40 * (online_security == "Yes").astype(float)
        - 0.35 * (tech_support == "Yes").astype(float)
        + 0.30 * senior.astype(float)
        - 0.035 * tenure.astype(float)
        + 0.012 * (monthly_charges - 60.0)
        + rng.normal(0.0, 0.55, size=n_samples)
    )
    probs = expit(logit)
    churn_bin = (rng.random(n_samples) < probs).astype(int)
    churn_str = np.where(churn_bin == 1, "Yes", "No")

    return pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges_str,
            "Churn": churn_str,
        }
    )


def load_churn_dataset(data_cfg: DataConfig, random_seed: int = 42) -> Tuple[pd.DataFrame, str]:
    """Load the raw Telco churn CSV or generate the synthetic fixture if configured.

    Returns:
        Tuple of (raw_dataframe, data_source_label).
    """
    if data_cfg.use_synthetic_fixture:
        df = generate_synthetic_telco_fixture(
            n_samples=data_cfg.synthetic_n_samples,
            random_state=random_seed,
        )
        return df, "synthetic_fixture"

    csv_path = Path(data_cfg.raw_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Real dataset file not found at '{csv_path}'. "
            "Run `python scripts/download_data.py` to download the IBM Telco Customer Churn CSV, "
            "or pass `--use-synthetic` to run offline verification on the synthetic fixture."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Loaded CSV at '{csv_path}' is empty.")
    return df, "ibm_telco_csv"


def clean_blank_numeric_columns(
    df: pd.DataFrame,
    numeric_candidates: Sequence[str] = KNOWN_NUMERIC_CANDIDATES,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Coerce whitespace/blank string entries in known numeric columns (e.g., TotalCharges) to NaN.

    Important Note:
        This step only converts string representations like `" "` into `np.nan` so the column has
        a consistent `float64` dtype. It does NOT impute missing values; actual median/mean
        imputation happens strictly inside the training-fitted `SimpleImputer` pipeline.
    """
    cleaned = df.copy()
    coerced_missing_counts: Dict[str, int] = {}

    for col in numeric_candidates:
        if col in cleaned.columns:
            before_na = int(cleaned[col].isna().sum())
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            after_na = int(cleaned[col].isna().sum())
            coerced_missing_counts[col] = after_na - before_na

    return cleaned, coerced_missing_counts


def inspect_dataframe(
    df: pd.DataFrame,
    target_column: str = "Churn",
    positive_label_str: str = "Yes",
) -> Dict[str, Any]:
    """Inspect dataset dimensions, missing values, duplicates, and target distribution."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame columns.")

    cleaned_df, blank_coercions = clean_blank_numeric_columns(df)
    missing_by_col = {
        str(col): int(cnt)
        for col, cnt in cleaned_df.isna().sum().items()
        if int(cnt) > 0
    }

    target_series = cleaned_df[target_column]
    value_counts = {str(k): int(v) for k, v in target_series.value_counts(dropna=False).items()}
    if target_series.dtype == object or str(target_series.dtype).startswith("string"):
        pos_count = int((target_series.astype(str) == positive_label_str).sum())
    else:
        pos_count = int((target_series == 1).sum())

    total_rows = int(len(cleaned_df))
    pos_rate = round(pos_count / max(1, total_rows), 6)

    id_col = "customerID" if "customerID" in cleaned_df.columns else None
    duplicate_ids = (
        int(cleaned_df[id_col].duplicated().sum()) if id_col is not None else 0
    )

    return {
        "n_rows": total_rows,
        "n_columns": int(cleaned_df.shape[1]),
        "columns": [str(c) for c in cleaned_df.columns],
        "duplicate_rows": int(cleaned_df.duplicated().sum()),
        "duplicate_customer_ids": duplicate_ids,
        "blank_numeric_strings_coerced_to_nan": blank_coercions,
        "missing_values_by_column": missing_by_col,
        "total_missing_values": int(sum(missing_by_col.values())),
        "target_column": target_column,
        "positive_label": positive_label_str,
        "target_counts": value_counts,
        "positive_class_count": pos_count,
        "negative_class_count": total_rows - pos_count,
        "positive_class_rate": pos_rate,
    }


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str = "Churn",
    positive_label_str: str = "Yes",
    negative_label_str: str = "No",
    id_columns: Optional[Sequence[str]] = None,
    leakage_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Separate predictive features X and binary target y (0/1), dropping ID and leakage columns.

    Returns:
        Tuple of (X_dataframe, y_binary_array, dropped_columns_list).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not present in dataset.")

    cleaned, _ = clean_blank_numeric_columns(df)

    # Encode binary target explicitly and validate labels
    raw_target = cleaned[target_column]
    if raw_target.isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")

    if raw_target.dtype == object or str(raw_target.dtype).startswith("string"):
        str_vals = raw_target.astype(str).str.strip()
        valid_set = {positive_label_str, negative_label_str, "1", "0"}
        unexpected = set(str_vals.unique()) - valid_set
        if unexpected:
            raise ValueError(
                f"Unexpected target labels {unexpected} in '{target_column}'. "
                f"Expected '{positive_label_str}'/'{negative_label_str}'."
            )
        y = np.where((str_vals == positive_label_str) | (str_vals == "1"), 1, 0).astype(
            np.int64
        )
    else:
        num_vals = pd.to_numeric(raw_target, errors="raise").to_numpy(dtype=np.int64)
        unexpected_num = set(np.unique(num_vals).tolist()) - {0, 1}
        if unexpected_num:
            raise ValueError(f"Numeric target must be binary {{0, 1}}, got {unexpected_num}.")
        y = num_vals

    # Identify all identifier and target-leakage columns to drop from X
    ids_to_drop = set(id_columns or ["customerID", "CustomerID"])
    leaks_to_drop = set(
        leakage_columns
        or [
            target_column,
            "Churn Value",
            "Churn Score",
            "Churn Reason",
            "Customer Status",
        ]
    )
    leaks_to_drop.add(target_column)

    # Also auto-detect any column whose lowercased name ends with '_id' or equals 'id'
    for col in cleaned.columns:
        col_low = str(col).strip().lower()
        if col_low in ("id", "customer_id", "customerid") or col_low.startswith("churn"):
            ids_to_drop.add(col)

    all_drop_candidates = ids_to_drop | leaks_to_drop
    dropped_cols = [c for c in cleaned.columns if c in all_drop_candidates]
    X = cleaned.drop(columns=dropped_cols).copy()

    if target_column in X.columns:
        raise RuntimeError("Target leakage guard triggered: target column remained in X.")
    if X.shape[1] == 0:
        raise ValueError("No predictive feature columns remain after dropping ID/leakage columns.")

    return X, y, dropped_cols
