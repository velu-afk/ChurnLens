"""Configuration dataclasses and YAML loader for ChurnLens."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class DataConfig:
    """Dataset paths, schema settings, leakage exclusions, and split ratios."""

    raw_csv_path: str = "data/raw/Telco-Customer-Churn.csv"
    source_url: str = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    )
    use_synthetic_fixture: bool = False
    synthetic_n_samples: int = 600
    target_column: str = "Churn"
    positive_label_str: str = "Yes"
    negative_label_str: str = "No"
    id_columns: List[str] = field(default_factory=lambda: ["customerID", "CustomerID"])
    leakage_columns: List[str] = field(
        default_factory=lambda: [
            "Churn",
            "Churn Value",
            "Churn Score",
            "Churn Reason",
            "Customer Status",
        ]
    )
    test_size: float = 0.20
    val_size_from_train: float = 0.20


@dataclass
class PreprocessingConfig:
    """Imputation and scaling configuration."""

    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    scale_numeric: bool = True


@dataclass
class LogisticRegressionConfig:
    """Hyperparameters for Model A (Logistic Regression baseline)."""

    C: float = 1.0
    class_weight: Optional[str] = "balanced"
    max_iter: int = 1000
    solver: str = "lbfgs"


@dataclass
class RandomForestConfig:
    """Hyperparameters for Model B (Random Forest Classifier)."""

    n_estimators: int = 120
    max_depth: Optional[int] = 8
    min_samples_split: int = 6
    min_samples_leaf: int = 3
    max_features: str = "sqrt"
    class_weight: Optional[str] = "balanced"


@dataclass
class ModelsConfig:
    """Container for Model A and Model B configurations."""

    logistic_regression: LogisticRegressionConfig = field(
        default_factory=LogisticRegressionConfig
    )
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)


@dataclass
class EvaluationConfig:
    """Threshold analysis, permutation importance, and output artifact settings."""

    default_threshold: float = 0.50
    threshold_grid_min: float = 0.20
    threshold_grid_max: float = 0.80
    threshold_grid_steps: int = 25
    threshold_selection_metric: str = "f1"
    permutation_importance_repeats: int = 5
    output_dir: str = "results"
    save_models: bool = True
    illustrative_outreach_top_k_fraction: float = 0.20
    illustrative_outreach_cost_per_customer_usd: float = 15.0


@dataclass
class ChurnLensConfig:
    """Top-level configuration object for ChurnLens."""

    project_name: str = "ChurnLens"
    random_seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        """Validate configuration parameters and raise ValueError on invalid settings."""
        if self.random_seed < 0:
            raise ValueError(f"random_seed must be >= 0, got {self.random_seed}.")
        if not (0.05 <= self.data.test_size <= 0.50):
            raise ValueError(
                f"data.test_size must be in [0.05, 0.50], got {self.data.test_size}."
            )
        if not (0.05 <= self.data.val_size_from_train <= 0.50):
            raise ValueError(
                "data.val_size_from_train must be in [0.05, 0.50], "
                f"got {self.data.val_size_from_train}."
            )
        if self.models.logistic_regression.C <= 0.0:
            raise ValueError("models.logistic_regression.C must be strictly positive.")
        if self.models.random_forest.n_estimators <= 0:
            raise ValueError("models.random_forest.n_estimators must be > 0.")
        if not (0.0 < self.evaluation.default_threshold < 1.0):
            raise ValueError("evaluation.default_threshold must lie in (0, 1).")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration into a plain dictionary."""
        return asdict(self)


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    use_synthetic: Optional[bool] = None,
    output_dir: Optional[str] = None,
    raw_csv_path: Optional[str] = None,
) -> ChurnLensConfig:
    """Load configuration from YAML and apply optional CLI overrides."""
    raw: Dict[str, Any] = {}
    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: '{p}'")
        with open(p, "r", encoding="utf-8") as fp:
            loaded = yaml.safe_load(fp)
            if isinstance(loaded, dict):
                raw = loaded

    data_cfg = DataConfig(**raw.get("data", {}))
    prep_cfg = PreprocessingConfig(**raw.get("preprocessing", {}))
    models_raw = raw.get("models", {})
    lr_cfg = LogisticRegressionConfig(**models_raw.get("logistic_regression", {}))
    rf_cfg = RandomForestConfig(**models_raw.get("random_forest", {}))
    eval_cfg = EvaluationConfig(**raw.get("evaluation", {}))

    cfg = ChurnLensConfig(
        project_name=raw.get("project_name", "ChurnLens"),
        random_seed=int(raw.get("random_seed", 42)),
        data=data_cfg,
        preprocessing=prep_cfg,
        models=ModelsConfig(logistic_regression=lr_cfg, random_forest=rf_cfg),
        evaluation=eval_cfg,
    )

    if use_synthetic is not None:
        cfg.data.use_synthetic_fixture = bool(use_synthetic)
    if output_dir is not None:
        cfg.evaluation.output_dir = str(output_dir)
    if raw_csv_path is not None:
        cfg.data.raw_csv_path = str(raw_csv_path)

    cfg.validate()
    return cfg
