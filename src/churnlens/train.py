"""Stratified dataset splitting, model training, validation threshold tuning, and serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churnlens.config import ChurnLensConfig
from churnlens.data import prepare_features_and_target
from churnlens.models import build_model_pipelines
from churnlens.preprocessing import identify_column_types


@dataclass
class SplitBundle:
    """Container holding stratified train, validation (from train), and untouched test splits."""

    X_train_full: pd.DataFrame
    y_train_full: np.ndarray
    X_subtrain: pd.DataFrame
    y_subtrain: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    numeric_cols: List[str]
    categorical_cols: List[str]
    dropped_cols: List[str]


def create_stratified_splits(
    df: pd.DataFrame,
    cfg: ChurnLensConfig,
) -> SplitBundle:
    """Create a stratified 80/20 train/test split and a training-derived validation split.

    Leakage & Isolation Protocol:
        1. `prepare_features_and_target` removes `customerID` and target/leakage columns.
        2. `train_test_split(..., test_size=0.20, stratify=y, random_state=seed)` isolates
           `X_test, y_test` (20% of data). `X_test` is untouched until final evaluation.
        3. `X_train_full, y_train_full` (80% of data) is further split using stratification into
           `X_subtrain, y_subtrain` and `X_val, y_val` (`val_size_from_train=0.20`) strictly for
           classification threshold selection.
    """
    X, y, dropped_cols = prepare_features_and_target(
        df=df,
        target_column=cfg.data.target_column,
        positive_label_str=cfg.data.positive_label_str,
        negative_label_str=cfg.data.negative_label_str,
        id_columns=cfg.data.id_columns,
        leakage_columns=cfg.data.leakage_columns,
    )
    numeric_cols, categorical_cols = identify_column_types(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        stratify=y,
        random_state=cfg.random_seed,
    )

    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=cfg.data.val_size_from_train,
        stratify=y_train_full,
        random_state=cfg.random_seed,
    )

    return SplitBundle(
        X_train_full=X_train_full.reset_index(drop=True),
        y_train_full=np.asarray(y_train_full, dtype=np.int64),
        X_subtrain=X_subtrain.reset_index(drop=True),
        y_subtrain=np.asarray(y_subtrain, dtype=np.int64),
        X_val=X_val.reset_index(drop=True),
        y_val=np.asarray(y_val, dtype=np.int64),
        X_test=X_test.reset_index(drop=True),
        y_test=np.asarray(y_test, dtype=np.int64),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        dropped_cols=dropped_cols,
    )


def select_threshold_on_validation(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    grid_min: float = 0.20,
    grid_max: float = 0.80,
    grid_steps: int = 25,
) -> Tuple[float, List[Dict[str, float]]]:
    """Evaluate candidate classification thresholds on the validation split and pick best F1.

    Never call this function on `y_test`!
    """
    thresholds = np.linspace(grid_min, grid_max, grid_steps)
    records: List[Dict[str, float]] = []
    best_thresh = 0.50
    best_f1 = -1.0

    for thr in thresholds:
        preds = (val_probs >= thr).astype(np.int64)
        prec = float(precision_score(y_val, preds, pos_label=1, average="binary", zero_division=0))
        rec = float(recall_score(y_val, preds, pos_label=1, average="binary", zero_division=0))
        f1 = float(f1_score(y_val, preds, pos_label=1, average="binary", zero_division=0))
        records.append(
            {
                "threshold": round(float(thr), 4),
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "f1": round(f1, 6),
            }
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thr)

    return round(best_thresh, 4), records


def train_models_and_tune_thresholds(
    splits: SplitBundle,
    cfg: ChurnLensConfig,
) -> Dict[str, Dict[str, Any]]:
    """Train Logistic Regression and Random Forest pipelines and tune thresholds on validation.

    Procedure for each model:
        1. Fit a validation pipeline on `(X_subtrain, y_subtrain)` and predict probabilities on
           `X_val` to select `selected_threshold` and record the validation threshold curve.
        2. Fit the final model pipeline on the full 80% training set `(X_train_full, y_train_full)`
           so all training data is used while `X_test` remains completely untouched.
    """
    val_pipelines = build_model_pipelines(splits.X_subtrain, cfg)
    final_pipelines = build_model_pipelines(splits.X_train_full, cfg)

    trained_artifacts: Dict[str, Dict[str, Any]] = {}

    for model_name in ("logistic_regression", "random_forest"):
        # Step 1: Fit on subtrain and tune threshold on validation split
        val_pipe = val_pipelines[model_name]
        val_pipe.fit(splits.X_subtrain, splits.y_subtrain)
        val_probs = val_pipe.predict_proba(splits.X_val)[:, 1]
        best_thr, thr_curve = select_threshold_on_validation(
            y_val=splits.y_val,
            val_probs=val_probs,
            grid_min=cfg.evaluation.threshold_grid_min,
            grid_max=cfg.evaluation.threshold_grid_max,
            grid_steps=cfg.evaluation.threshold_grid_steps,
        )

        # Step 2: Fit final pipeline on full training split
        final_pipe = final_pipelines[model_name]
        final_pipe.fit(splits.X_train_full, splits.y_train_full)

        trained_artifacts[model_name] = {
            "pipeline": final_pipe,
            "selected_threshold": best_thr,
            "validation_threshold_curve": thr_curve,
        }

    return trained_artifacts


def save_model_pipeline(pipeline: Pipeline, path: Union[str, Path]) -> Path:
    """Serialize a fitted scikit-learn Pipeline to disk using joblib."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    return out_path


def load_model_pipeline(path: Union[str, Path]) -> Pipeline:
    """Load a serialized scikit-learn Pipeline from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Saved model file not found: '{p}'")
    return joblib.load(p)
