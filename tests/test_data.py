"""Unit tests for data loading, blank-string cleaning, ID removal, and target leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnlens.data import (
    clean_blank_numeric_columns,
    generate_synthetic_telco_fixture,
    inspect_dataframe,
    prepare_features_and_target,
)


def test_synthetic_fixture_schema_and_blank_total_charges() -> None:
    """Verify the synthetic Telco fixture has 21 columns and whitespace strings in TotalCharges."""
    df = generate_synthetic_telco_fixture(n_samples=120, random_state=42)
    assert df.shape == (120, 21)
    assert "customerID" in df.columns
    assert "Churn" in df.columns
    assert (df["TotalCharges"] == " ").sum() >= 1

    summary = inspect_dataframe(df)
    assert summary["n_rows"] == 120
    assert summary["n_columns"] == 21
    assert summary["blank_numeric_strings_coerced_to_nan"]["TotalCharges"] >= 1
    assert 0.10 <= summary["positive_class_rate"] <= 0.50


def test_clean_blank_numeric_columns_coerces_whitespace_to_nan() -> None:
    """Verify that blank strings ' ' in TotalCharges become NaN float values."""
    raw = pd.DataFrame(
        {
            "tenure": [0, 12, 24],
            "MonthlyCharges": [20.0, 55.5, 80.0],
            "TotalCharges": [" ", "666.00", "1920.50"],
            "Churn": ["No", "Yes", "No"],
        }
    )
    cleaned, counts = clean_blank_numeric_columns(raw)
    assert counts["TotalCharges"] == 1
    assert np.isnan(cleaned.loc[0, "TotalCharges"])
    assert cleaned.loc[1, "TotalCharges"] == pytest.approx(666.0)


def test_prepare_features_removes_identifiers_and_target_leakage() -> None:
    """Verify that customerID, Churn, and any leakage columns are excluded from X."""
    df = generate_synthetic_telco_fixture(n_samples=60, random_state=7)
    # Inject a synthetic leakage column and extra ID column
    df["Churn Score"] = np.where(df["Churn"] == "Yes", 95, 10)
    df["Churn Reason"] = np.where(df["Churn"] == "Yes", "Competitor", "None")
    df["account_id"] = range(len(df))

    X, y, dropped = prepare_features_and_target(df)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert "Churn Score" not in X.columns
    assert "Churn Reason" not in X.columns
    assert "customerID" in dropped
    assert "Churn" in dropped
    assert set(np.unique(y)).issubset({0, 1})
    assert len(X) == len(y) == 60


def test_invalid_target_values_raise_value_error() -> None:
    """Verify that unexpected target values raise a clear ValueError."""
    df = generate_synthetic_telco_fixture(n_samples=30, random_state=42)
    df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Unexpected target labels"):
        _ = prepare_features_and_target(df)
