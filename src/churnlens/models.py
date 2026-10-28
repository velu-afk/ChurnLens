"""Model pipeline builders for Logistic Regression (Model A) and Random Forest (Model B)."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churnlens.config import ChurnLensConfig
from churnlens.preprocessing import build_preprocessor


def build_logistic_regression_pipeline(
    X_sample: pd.DataFrame,
    cfg: ChurnLensConfig,
) -> Pipeline:
    """Build Model A: Preprocessing + L2-regularized Logistic Regression pipeline."""
    preprocessor, _, _ = build_preprocessor(
        X_sample=X_sample,
        numeric_impute_strategy=cfg.preprocessing.numeric_impute_strategy,
        categorical_impute_strategy=cfg.preprocessing.categorical_impute_strategy,
        scale_numeric=cfg.preprocessing.scale_numeric,
    )
    lr_cfg = cfg.models.logistic_regression
    classifier = LogisticRegression(
        C=lr_cfg.C,
        class_weight=lr_cfg.class_weight,
        max_iter=lr_cfg.max_iter,
        solver=lr_cfg.solver,
        random_state=cfg.random_seed,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_random_forest_pipeline(
    X_sample: pd.DataFrame,
    cfg: ChurnLensConfig,
) -> Pipeline:
    """Build Model B: Preprocessing + regularized Random Forest Classifier pipeline."""
    preprocessor, _, _ = build_preprocessor(
        X_sample=X_sample,
        numeric_impute_strategy=cfg.preprocessing.numeric_impute_strategy,
        categorical_impute_strategy=cfg.preprocessing.categorical_impute_strategy,
        scale_numeric=cfg.preprocessing.scale_numeric,
    )
    rf_cfg = cfg.models.random_forest
    classifier = RandomForestClassifier(
        n_estimators=rf_cfg.n_estimators,
        max_depth=rf_cfg.max_depth,
        min_samples_split=rf_cfg.min_samples_split,
        min_samples_leaf=rf_cfg.min_samples_leaf,
        max_features=rf_cfg.max_features,
        class_weight=rf_cfg.class_weight,
        random_state=cfg.random_seed,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_model_pipelines(
    X_sample: pd.DataFrame,
    cfg: ChurnLensConfig,
) -> Dict[str, Pipeline]:
    """Return a dictionary containing both unfitted scikit-learn model pipelines."""
    return {
        "logistic_regression": build_logistic_regression_pipeline(X_sample, cfg),
        "random_forest": build_random_forest_pipeline(X_sample, cfg),
    }
