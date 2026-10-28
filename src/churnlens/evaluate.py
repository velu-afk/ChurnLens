"""Evaluation metrics, feature/permutation importance, and retention prioritization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from churnlens.config import ChurnLensConfig
from churnlens.train import SplitBundle


def compute_binary_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Compute binary classification metrics at a specified probability threshold.

    Positive class is explicitly `1` (`Churn == "Yes"`).
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_probs, dtype=np.float64)
    if y_t.shape[0] != probs.shape[0] or y_t.size == 0:
        raise ValueError("y_true and y_probs must be non-empty 1D arrays of equal length.")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("Predicted probabilities must lie in [0.0, 1.0].")

    y_pred = (probs >= float(threshold)).astype(np.int64)
    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_t, y_pred))
    prec = float(precision_score(y_t, y_pred, pos_label=1, average="binary", zero_division=0))
    rec = float(recall_score(y_t, y_pred, pos_label=1, average="binary", zero_division=0))
    f1 = float(f1_score(y_t, y_pred, pos_label=1, average="binary", zero_division=0))
    roc_auc = float(roc_auc_score(y_t, probs)) if len(np.unique(y_t)) == 2 else 0.0

    report_str = classification_report(
        y_t,
        y_pred,
        labels=[0, 1],
        target_names=["No Churn (0)", "Churn (1)"],
        zero_division=0,
    )
    report_dict = classification_report(
        y_t,
        y_pred,
        labels=[0, 1],
        target_names=["No Churn (0)", "Churn (1)"],
        zero_division=0,
        output_dict=True,
    )

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "roc_auc": round(roc_auc, 6),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "classification_report_text": report_str,
        "classification_report": report_dict,
    }


def extract_feature_importances(
    pipeline: Pipeline,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract held-out permutation importance (input features) and internal model weights/Gini.

    Caveat:
        Both permutation importance and tree Gini / logistic coefficients measure predictive
        association within the fitted model, NOT causal effect in the real world.
    """
    # 1. Held-out permutation importance over raw input columns of X_eval
    perm = permutation_importance(
        pipeline,
        X_eval,
        y_eval,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
    )
    perm_records: List[Dict[str, Any]] = []
    for col_name, mean_val, std_val in zip(
        X_eval.columns, perm.importances_mean, perm.importances_std
    ):
        perm_records.append(
            {
                "feature": str(col_name),
                "importance_mean": round(float(mean_val), 6),
                "importance_std": round(float(std_val), 6),
            }
        )
    perm_records.sort(key=lambda r: r["importance_mean"], reverse=True)

    # 2. Internal transformed feature importances (Gini for RF, |coef| for LogisticRegression)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed_names = [str(n) for n in preprocessor.get_feature_names_out()]

    internal_records: List[Dict[str, Any]] = []
    if hasattr(classifier, "feature_importances_"):
        for name, val in zip(transformed_names, classifier.feature_importances_):
            internal_records.append(
                {
                    "transformed_feature": name,
                    "importance": round(float(val), 6),
                    "direction": "unsigned_gini",
                }
            )
        internal_records.sort(key=lambda r: r["importance"], reverse=True)
    elif hasattr(classifier, "coef_"):
        coefs = classifier.coef_[0]
        for name, val in zip(transformed_names, coefs):
            internal_records.append(
                {
                    "transformed_feature": name,
                    "importance": round(float(abs(val)), 6),
                    "signed_coefficient": round(float(val), 6),
                    "direction": "increases_churn_risk" if val > 0 else "decreases_churn_risk",
                }
            )
        internal_records.sort(key=lambda r: r["importance"], reverse=True)

    return {
        "permutation_importance": perm_records,
        "internal_transformed_importance": internal_records[:15],
    }


def build_illustrative_retention_plan(
    y_test: np.ndarray,
    test_probs: np.ndarray,
    top_k_fraction: float = 0.20,
    illustrative_cost_per_customer_usd: float = 15.0,
) -> Dict[str, Any]:
    """Show how a retention team could rank test customers by predicted churn probability.

    Disclaimer:
        All outreach budget and cost figures are strictly illustrative to demonstrate probability
        ranking. Predicting churn probability does not guarantee that an intervention will prevent
        churn without a randomized controlled uplift study.
    """
    n_total = int(len(y_test))
    total_actual_churners = int(np.sum(y_test == 1))
    overall_churn_rate = total_actual_churners / max(1, n_total)

    order = np.argsort(test_probs)[::-1]
    y_sorted = y_test[order]
    probs_sorted = test_probs[order]

    tiers = []
    for frac, label in (
        (0.10, "Top 10% Highest Risk"),
        (top_k_fraction, f"Top {int(round(top_k_fraction * 100))}% Highest Risk (Default Budget)"),
        (0.30, "Top 30% Highest Risk"),
    ):
        k = max(1, int(round(n_total * frac)))
        captured_churners = int(np.sum(y_sorted[:k] == 1))
        precision_in_tier = captured_churners / float(k)
        recall_of_total_churn = captured_churners / max(1, total_actual_churners)
        lift = precision_in_tier / max(1e-9, overall_churn_rate)
        tiers.append(
            {
                "tier_label": label,
                "fraction_contacted": round(float(frac), 2),
                "customers_contacted": k,
                "min_probability_in_tier": round(float(probs_sorted[k - 1]), 4),
                "true_churners_in_tier": captured_churners,
                "precision_in_tier": round(precision_in_tier, 4),
                "share_of_all_churners_captured": round(recall_of_total_churn, 4),
                "lift_over_random_outreach": round(lift, 2),
                "illustrative_outreach_cost_usd": round(
                    k * float(illustrative_cost_per_customer_usd), 2
                ),
            }
        )

    return {
        "disclaimer": (
            "Illustrative prioritization analysis only. Outreach costs ($15/customer) and "
            "tier sizes are hypothetical examples; real-world churn reduction requires an "
            "experimental A/B retention intervention study."
        ),
        "test_set_customers": n_total,
        "test_set_actual_churners": total_actual_churners,
        "baseline_churn_rate": round(overall_churn_rate, 4),
        "tiers": tiers,
    }


def evaluate_all_models(
    trained_artifacts: Dict[str, Dict[str, Any]],
    splits: SplitBundle,
    cfg: ChurnLensConfig,
) -> Dict[str, Any]:
    """Evaluate all trained models on the held-out test set at both 0.50 and validation thresholds."""
    results: Dict[str, Any] = {}

    for model_name, item in trained_artifacts.items():
        pipe: Pipeline = item["pipeline"]
        sel_thr: float = item["selected_threshold"]

        test_probs = pipe.predict_proba(splits.X_test)[:, 1]
        default_metrics = compute_binary_metrics(
            splits.y_test, test_probs, threshold=cfg.evaluation.default_threshold
        )
        tuned_metrics = compute_binary_metrics(
            splits.y_test, test_probs, threshold=sel_thr
        )
        fpr, tpr, _ = roc_curve(splits.y_test, test_probs, pos_label=1)

        importances = extract_feature_importances(
            pipeline=pipe,
            X_eval=splits.X_test,
            y_eval=splits.y_test,
            n_repeats=cfg.evaluation.permutation_importance_repeats,
            random_state=cfg.random_seed,
        )
        retention_plan = build_illustrative_retention_plan(
            y_test=splits.y_test,
            test_probs=test_probs,
            top_k_fraction=cfg.evaluation.illustrative_outreach_top_k_fraction,
            illustrative_cost_per_customer_usd=cfg.evaluation.illustrative_outreach_cost_per_customer_usd,
        )

        results[model_name] = {
            "model_name": model_name,
            "default_threshold_metrics": default_metrics,
            "validation_selected_threshold": sel_thr,
            "validation_tuned_threshold_metrics": tuned_metrics,
            "validation_threshold_curve": item["validation_threshold_curve"],
            "roc_curve": {
                "fpr": [round(float(x), 6) for x in fpr],
                "tpr": [round(float(y), 6) for y in tpr],
            },
            "feature_importance": importances,
            "illustrative_retention_prioritization": retention_plan,
            "predicted_probabilities_sample": [round(float(p), 6) for p in test_probs[:20]],
            "all_test_probabilities": test_probs,
        }

    return results


def export_comparison_tables(
    evaluation_results: Dict[str, Any],
    output_dir: Union[str, Path],
    data_source_label: str,
) -> Tuple[Path, Path]:
    """Write tidy CSV and Markdown comparison tables for both models (at 0.50 and tuned thresholds)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for model_name, res in evaluation_results.items():
        for mode_label, m in (
            ("Default (0.50)", res["default_threshold_metrics"]),
            ("Val-Tuned", res["validation_tuned_threshold_metrics"]),
        ):
            rows.append(
                {
                    "model": model_name,
                    "threshold_policy": mode_label,
                    "threshold": m["threshold"],
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "roc_auc": m["roc_auc"],
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                    "tn": m["tn"],
                    "data_source": data_source_label,
                }
            )

    df_cmp = pd.DataFrame(rows)
    csv_path = out_dir / "model_comparison.csv"
    df_cmp.to_csv(csv_path, index=False)

    headers = [
        "Model",
        "Threshold Policy",
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "TP / FP / FN / TN",
        "Data Source",
    ]
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        counts_str = f"{r['tp']} / {r['fp']} / {r['fn']} / {r['tn']}"
        md_lines.append(
            f"| `{r['model']}` | {r['threshold_policy']} | {r['threshold']:.2f} | "
            f"{r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1']:.4f} | {r['roc_auc']:.4f} | {counts_str} | `{r['data_source']}` |"
        )

    md_path = out_dir / "model_comparison.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path
