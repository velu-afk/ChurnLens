"""Unit tests for preprocessing, train/test isolation, and unseen category handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnlens.config import ChurnLensConfig
from churnlens.data import generate_synthetic_telco_fixture
from churnlens.preprocessing import build_preprocessor
from churnlens.train import create_stratified_splits


def test_stratified_splits_preserve_class_balance_and_disjoint_sizes() -> None:
    """Verify 80/20 train/test split and training-derived validation split proportions."""
    cfg = ChurnLensConfig()
    df = generate_synthetic_telco_fixture(n_samples=200, random_state=42)
    splits = create_stratified_splits(df, cfg)

    assert len(splits.X_train_full) == 160
    assert len(splits.X_test) == 40
    assert len(splits.X_subtrain) == 128
    assert len(splits.X_val) == 32
    assert "Churn" not in splits.X_train_full.columns
    assert "customerID" not in splits.X_train_full.columns

    # Class ratios should match closely due to stratification
    train_rate = float(splits.y_train_full.mean())
    test_rate = float(splits.y_test.mean())
    assert abs(train_rate - test_rate) < 0.05


def test_preprocessor_fits_exclusively_on_training_statistics() -> None:
    """Verify that SimpleImputer and StandardScaler use only training-set statistics."""
    X_train = pd.DataFrame(
        {
            "tenure": [10.0, 20.0, 30.0, np.nan],
            "Contract": ["Month-to-month", "Month-to-month", "One year", None],
        }
    )
    # Test set has extreme numeric values and an unseen category
    X_test = pd.DataFrame(
        {
            "tenure": [1000.0, np.nan],
            "Contract": ["Two year", "Unseen_Contract_Type"],
        }
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    assert num_cols == ["tenure"]
    assert cat_cols == ["Contract"]

    Xt_train = preprocessor.fit_transform(X_train)
    Xt_test = preprocessor.transform(X_test)

    assert not np.isnan(Xt_train).any()
    assert not np.isnan(Xt_test).any()

    # In X_train, median of [10, 20, 30] is 20.0.
    # Therefore the missing value in X_test[1] is imputed as 20.0, which scales to 0.0!
    assert Xt_test[1, 0] == pytest.approx(0.0, abs=1e-6)
    # And the unseen category in X_test[1] produces all-zero one-hot columns without error
    assert np.all(Xt_test[1, 1:] == 0.0)
