"""Command-Line Interface (CLI) and workflow orchestrator for ChurnLens."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from churnlens.config import ChurnLensConfig, load_config
from churnlens.data import inspect_dataframe, load_churn_dataset, prepare_features_and_target
from churnlens.evaluate import evaluate_all_models, export_comparison_tables
from churnlens.train import (
    create_stratified_splits,
    load_model_pipeline,
    save_model_pipeline,
    train_models_and_tune_thresholds,
)
from churnlens.visualization import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_metric_comparison,
    plot_roc_curves,
    plot_threshold_tradeoff,
)


def get_environment_summary() -> Dict[str, Any]:
    """Inspect OS, Python version, available disk space, and installed library versions."""
    import joblib
    import matplotlib
    import scipy
    import sklearn
    import yaml

    disk = shutil.disk_usage(Path.cwd())
    return {
        "os_platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count() or 1,
        "disk_free_gb": round(disk.free / (1024.0**3), 2),
        "packages": {
            "numpy": getattr(np, "__version__", "unknown"),
            "scipy": getattr(scipy, "__version__", "unknown"),
            "pandas": getattr(pd, "__version__", "unknown"),
            "scikit_learn": getattr(sklearn, "__version__", "unknown"),
            "matplotlib": getattr(matplotlib, "__version__", "unknown"),
            "pyyaml": getattr(yaml, "__version__", "unknown"),
            "joblib": getattr(joblib, "__version__", "unknown"),
        },
    }


def run_full_pipeline(cfg: ChurnLensConfig) -> Dict[str, Any]:
    """Execute dataset validation, stratified splitting, training, evaluation, and plotting."""
    cfg.validate()
    out_dir = Path(cfg.evaluation.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df, data_source_label = load_churn_dataset(cfg.data, random_seed=cfg.random_seed)
    data_summary = inspect_dataframe(
        raw_df,
        target_column=cfg.data.target_column,
        positive_label_str=cfg.data.positive_label_str,
    )
    splits = create_stratified_splits(raw_df, cfg)
    trained_artifacts = train_models_and_tune_thresholds(splits, cfg)
    eval_results = evaluate_all_models(trained_artifacts, splits, cfg)

    csv_path, md_path = export_comparison_tables(eval_results, out_dir, data_source_label)

    # Save predicted probabilities on the test set
    probs_df = pd.DataFrame(
        {
            "y_true": splits.y_test,
            "logistic_regression_prob": eval_results["logistic_regression"][
                "all_test_probabilities"
            ],
            "random_forest_prob": eval_results["random_forest"]["all_test_probabilities"],
        }
    )
    probs_csv = out_dir / "test_predictions.csv"
    probs_df.to_csv(probs_csv, index=False)

    # Remove full array before JSON serialization
    serializable_eval: Dict[str, Any] = {}
    for m_name, m_dict in eval_results.items():
         serializable_eval[m_name] = {
            k: v for k, v in m_dict.items() if k != "all_test_probabilities"
        }

    # Generate all required plots
    plots = {
        "metric_comparison": str(
            plot_metric_comparison(eval_results, out_dir / "model_comparison.png")
        ),
        "confusion_matrices": str(
            plot_confusion_matrices(eval_results, out_dir / "confusion_matrices.png")
        ),
        "roc_curves": str(plot_roc_curves(eval_results, out_dir / "roc_curves.png")),
        "feature_importance": str(
            plot_feature_importance(eval_results, out_dir / "feature_importance.png")
        ),
        "threshold_analysis": str(
            plot_threshold_tradeoff(eval_results, out_dir / "threshold_analysis.png")
        ),
    }

    saved_models: Dict[str, str] = {}
    if cfg.evaluation.save_models:
        models_dir = out_dir / "models"
        for m_name, item in trained_artifacts.items():
            m_path = save_model_pipeline(item["pipeline"], models_dir / f"{m_name}.joblib")
            saved_models[m_name] = str(m_path)

    run_metadata = {
        "project_name": cfg.project_name,
        "data_source": data_source_label,
        "is_synthetic_fixture": bool(cfg.data.use_synthetic_fixture),
        "random_seed": cfg.random_seed,
        "dataset_summary": data_summary,
        "split_sizes": {
            "train_full": int(len(splits.y_train_full)),
            "subtrain": int(len(splits.y_subtrain)),
            "validation": int(len(splits.y_val)),
            "test": int(len(splits.y_test)),
            "numeric_features": splits.numeric_cols,
            "categorical_features": splits.categorical_cols,
            "dropped_columns": splits.dropped_cols,
        },
        "config": cfg.to_dict(),
        "environment": get_environment_summary(),
        "models": serializable_eval,
        "artifacts": {
            "comparison_csv": str(csv_path),
            "comparison_markdown": str(md_path),
            "test_predictions_csv": str(probs_csv),
            "plots": plots,
            "saved_models": saved_models,
        },
    }

    metrics_json_path = out_dir / "evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as fp:
        json.dump(run_metadata, fp, indent=2)

    return run_metadata


def predict_customer_records(
    model_path: Union[str, Path],
    input_csv: Union[str, Path],
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Load a saved model pipeline and score new customer records from a CSV file."""
    pipe = load_model_pipeline(model_path)
    df_in = pd.read_csv(input_csv)
    # Coerce blank strings in numeric columns if present
    from churnlens.data import clean_blank_numeric_columns

    df_clean, _ = clean_blank_numeric_columns(df_in)
    probs = pipe.predict_proba(df_clean)[:, 1]
    preds = (probs >= float(threshold)).astype(int)

    out = df_in.copy()
    out["predicted_churn_probability"] = np.round(probs, 4)
    out["predicted_churn_label"] = np.where(preds == 1, "Yes", "No")
    return out


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ChurnLens."""
    parser = argparse.ArgumentParser(
        prog="churnlens",
        description="ChurnLens CLI: Customer Churn Prediction and Model Evaluation Pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. inspect-env
    sub.add_parser("inspect-env", help="Inspect OS, Python, disk space, and ML library versions.")

    # 2. validate-data
    val_p = sub.add_parser(
        "validate-data",
        help="Validate dataset schema, missing values, duplicates, and target distribution.",
    )
    val_p.add_argument("--config", type=str, default="configs/default.yaml")
    val_p.add_argument("--use-synthetic", action="store_true")
    val_p.add_argument("--data-path", type=str, default=None)

    # 3. train (runs training + evaluation + artifact generation)
    tr_p = sub.add_parser(
        "train",
        help="Train Logistic Regression and Random Forest pipelines and save evaluation results.",
    )
    tr_p.add_argument("--config", type=str, default="configs/default.yaml")
    tr_p.add_argument("--use-synthetic", action="store_true")
    tr_p.add_argument("--data-path", type=str, default=None)
    tr_p.add_argument("--output-dir", type=str, default=None)

    # 4. evaluate
    ev_p = sub.add_parser(
        "evaluate",
        help="Run or display evaluation metrics and comparison table from results directory.",
    )
    ev_p.add_argument("--results-dir", type=str, default="results")

    # 5. predict
    pr_p = sub.add_parser(
        "predict",
        help="Load a saved model pipeline and predict churn probabilities on a CSV file.",
    )
    pr_p.add_argument("--model-path", type=str, required=True)
    pr_p.add_argument("--input-csv", type=str, required=True)
    pr_p.add_argument("--output-csv", type=str, default=None)
    pr_p.add_argument("--threshold", type=float, default=0.50)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for ChurnLens."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.command

    if cmd == "inspect-env":
        print(json.dumps(get_environment_summary(), indent=2))
        return 0

    if cmd == "validate-data":
        cfg = load_config(
            args.config,
            use_synthetic=True if args.use_synthetic else None,
            raw_csv_path=args.data_path,
        )
        df, source_label = load_churn_dataset(cfg.data, random_seed=cfg.random_seed)
        summary = inspect_dataframe(
            df,
            target_column=cfg.data.target_column,
            positive_label_str=cfg.data.positive_label_str,
        )
        X, y, dropped = prepare_features_and_target(
            df,
            target_column=cfg.data.target_column,
            positive_label_str=cfg.data.positive_label_str,
            negative_label_str=cfg.data.negative_label_str,
            id_columns=cfg.data.id_columns,
            leakage_columns=cfg.data.leakage_columns,
        )
        summary["data_source"] = source_label
        summary["predictive_feature_count"] = int(X.shape[1])
        summary["dropped_columns"] = dropped
        print(json.dumps(summary, indent=2))
        return 0

    if cmd == "train":
        cfg = load_config(
            args.config,
            use_synthetic=True if args.use_synthetic else None,
            output_dir=args.output_dir,
            raw_csv_path=args.data_path,
        )
        meta = run_full_pipeline(cfg)
        md_table = Path(meta["artifacts"]["comparison_markdown"]).read_text(encoding="utf-8")
        print(f"=== ChurnLens Experiment Completed (data_source: {meta['data_source']}) ===")
        print(md_table)
        return 0

    if cmd == "evaluate":
        res_dir = Path(args.results_dir)
        md_file = res_dir / "model_comparison.md"
        json_file = res_dir / "evaluation_metrics.json"
        if not md_file.exists() or not json_file.exists():
            raise FileNotFoundError(
                f"No evaluation artifacts found in '{res_dir}'. Run `python -m churnlens train` first."
            )
        print(md_file.read_text(encoding="utf-8"))
        return 0

    if cmd == "predict":
        scored = predict_customer_records(
            model_path=args.model_path,
            input_csv=args.input_csv,
            threshold=args.threshold,
        )
        if args.output_csv:
            out_p = Path(args.output_csv)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(out_p, index=False)
            print(f"Saved {len(scored)} predictions to '{out_p}'.")
        else:
            print(
                scored[
                    ["predicted_churn_probability", "predicted_churn_label"]
                ].head(10).to_string(index=False)
            )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
