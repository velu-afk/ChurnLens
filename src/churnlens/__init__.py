"""ChurnLens: An End-to-End Customer Churn Prediction and Model Evaluation Pipeline."""

from __future__ import annotations

__version__ = "0.1.0"

from churnlens.config import ChurnLensConfig, load_config
from churnlens.data import (
    generate_synthetic_telco_fixture,
    inspect_dataframe,
    load_churn_dataset,
    prepare_features_and_target,
)
from churnlens.models import build_model_pipelines
from churnlens.preprocessing import build_preprocessor

__all__ = [
    "__version__",
    "ChurnLensConfig",
    "load_config",
    "generate_synthetic_telco_fixture",
    "inspect_dataframe",
    "load_churn_dataset",
    "prepare_features_and_target",
    "build_preprocessor",
    "build_model_pipelines",
]
