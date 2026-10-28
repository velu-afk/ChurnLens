"""Unit tests for Logistic Regression and Random Forest model fitting, probabilities, and serialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from churnlens.config import ChurnLensConfig
from churnlens.data import generate_synthetic_telco_fixture
from churnlens.models import build_model_pipelines
from churnlens.train import (
    create_stratified_splits,
    load_model_pipeline,
    save_model_pipeline,
    train_models_and_tune_thresholds,
)


def test_both_models_fit_and_output_valid_probabilities() -> None:
    """Verify both models fit on training data and output probabilities in [0, 1] summing to 1."""
    cfg = ChurnLensConfig()
    cfg.models.random_forest.n_estimators = 25
    df = generate_synthetic_telco_fixture(n_samples=150, random_state=42)
    splits = create_stratified_splits(df, cfg)

    pipelines = build_model_pipelines(splits.X_train_full, cfg)
    for name, pipe in pipelines.items():
        pipe.fit(splits.X_train_full, splits.y_train_full)
        probs = pipe.predict_proba(splits.X_test)
        preds = pipe.predict(splits.X_test)

        assert probs.shape == (len(splits.X_test), 2), f"Shape mismatch for {name}"
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        assert preds.shape == (len(splits.X_test),)
        assert set(np.unique(preds)).issubset({0, 1})


def test_reproducibility_and_model_serialization(tmp_path: Path) -> None:
    """Verify identical predictions for fixed random_seed and joblib save/load roundtrip."""
    cfg = ChurnLensConfig(random_seed=42)
    cfg.models.random_forest.n_estimators = 20
    df = generate_synthetic_telco_fixture(n_samples=120, random_state=42)
    splits = create_stratified_splits(df, cfg)

    art1 = train_models_and_tune_thresholds(splits, cfg)
    art2 = train_models_and_tune_thresholds(splits, cfg)

    for model_name in ("logistic_regression", "random_forest"):
        p1 = art1[model_name]["pipeline"].predict_proba(splits.X_test)[:, 1]
        p2 = art2[model_name]["pipeline"].predict_proba(splits.X_test)[:, 1]
        assert np.allclose(p1, p2, atol=1e-7)

        saved_path = save_model_pipeline(
            art1[model_name]["pipeline"], tmp_path / f"{model_name}.joblib"
        )
        loaded_pipe = load_model_pipeline(saved_path)
        p_loaded = loaded_pipe.predict_proba(splits.X_test)[:, 1]
        assert np.allclose(p1, p_loaded, atol=1e-7)
