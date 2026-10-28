"""Unit tests for binary classification metrics, threshold selection, and result export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from churnlens.config import ChurnLensConfig
from churnlens.evaluate import build_illustrative_retention_plan, compute_binary_metrics
from churnlens.main import run_full_pipeline
from churnlens.train import select_threshold_on_validation


def test_compute_binary_metrics_known_example() -> None:
    """Verify exact Accuracy, Precision, Recall, F1, Confusion Matrix, and ROC-AUC on known arrays."""
    # Ground truth: 2 positives (1), 2 negatives (0)
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_probs = np.array([0.10, 0.60, 0.40, 0.90], dtype=np.float64)

    # At threshold 0.50: y_pred = [0, 1, 0, 1] -> TN=1, FP=1, FN=1, TP=1
    m = compute_binary_metrics(y_true, y_probs, threshold=0.50)
    assert m["tn"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tp"] == 1
    assert m["accuracy"] == pytest.approx(0.50)
    assert m["precision"] == pytest.approx(0.50)
    assert m["recall"] == pytest.approx(0.50)
    assert m["f1"] == pytest.approx(0.50)
    assert m["roc_auc"] == pytest.approx(0.75)


def test_validation_threshold_selection_and_retention_plan() -> None:
    """Verify threshold selection on validation probabilities and illustrative retention tiers."""
    y_val = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    val_probs = np.array([0.10, 0.15, 0.20, 0.25, 0.35, 0.40], dtype=np.float64)

    # At default 0.50, recall is 0.0. At threshold ~0.35, precision=1.0 and recall=1.0 (F1=1.0).
    best_thr, curve = select_threshold_on_validation(
        y_val, val_probs, grid_min=0.20, grid_max=0.60, grid_steps=9
    )
    assert 0.30 <= best_thr <= 0.40
    assert len(curve) == 9

    plan = build_illustrative_retention_plan(y_val, val_probs, top_k_fraction=0.33)
    assert plan["test_set_actual_churners"] == 2
    assert len(plan["tiers"]) == 3


def test_end_to_end_pipeline_exports_artifacts(tmp_path: Path) -> None:
    """Verify that run_full_pipeline generates all expected CSV, Markdown, JSON, and PNG files."""
    cfg = ChurnLensConfig(random_seed=42)
    cfg.data.use_synthetic_fixture = True
    cfg.data.synthetic_n_samples = 120
    cfg.models.random_forest.n_estimators = 15
    cfg.evaluation.permutation_importance_repeats = 2
    cfg.evaluation.output_dir = str(tmp_path / "results")

    meta = run_full_pipeline(cfg)
    assert meta["data_source"] == "synthetic_fixture"
    assert Path(meta["artifacts"]["comparison_csv"]).exists()
    assert Path(meta["artifacts"]["comparison_markdown"]).exists()
    assert Path(meta["artifacts"]["test_predictions_csv"]).exists()
    for plot_path in meta["artifacts"]["plots"].values():
        assert Path(plot_path).exists()
