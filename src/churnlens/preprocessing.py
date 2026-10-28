"""Feature preprocessing pipeline using scikit-learn ColumnTransformer and Pipeline."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def identify_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Partition feature DataFrame columns into numeric and categorical lists.

    Note on `SeniorCitizen`:
        In the IBM Telco dataset, `SeniorCitizen` is stored as integers `0` and `1`.
        Treating binary/numeric columns consistently while one-hot encoding string/object
        columns keeps the feature space clean and interpretable.
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in X.columns:
        dtype = X[col].dtype
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            numeric_cols.append(str(col))
        else:
            categorical_cols.append(str(col))

    return numeric_cols, categorical_cols


def build_preprocessor(
    X_sample: pd.DataFrame,
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
    scale_numeric: bool = True,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Construct an unfitted ColumnTransformer for numeric and categorical features.

    Leakage Prevention Guarantee:
        `X_sample` is used ONLY to inspect column names and data types (`numeric_cols` vs.
        `categorical_cols`). The returned `ColumnTransformer` is completely unfitted—its
        imputers, scalers, and one-hot encoders are fitted exclusively when `.fit(X_train, y_train)`
        is called on the training split.
    """
    numeric_cols, categorical_cols = identify_column_types(X_sample)

    num_steps = [("imputer", SimpleImputer(strategy=numeric_impute_strategy))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_impute_strategy)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor, numeric_cols, categorical_cols
