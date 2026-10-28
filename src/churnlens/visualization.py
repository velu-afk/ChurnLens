"""Matplotlib visualizations for model comparison, confusion matrices, ROC curves, and importance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_metric_comparison(
    evaluation_results: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Plot side-by-side comparison of Accuracy, Precision, Recall, F1, and ROC-AUC (at 0.50)."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    lr_m = evaluation_results["logistic_regression"]["default_threshold_metrics"]
    rf_m = evaluation_results["random_forest"]["default_threshold_metrics"]

    lr_vals = [float(lr_m[k]) for k in metrics]
    rf_vals = [float(rf_m[k]) for k in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    b1 = ax.bar(x - width / 2, lr_vals, width, label="Logistic Regression", color="#1b4f72")
    b2 = ax.bar(x + width / 2, rf_vals, width, label="Random Forest", color="#d35400")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(
        "Model Performance Comparison on Held-Out Test Split (Threshold = 0.50)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrices(
    evaluation_results: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Plot side-by-side confusion matrices for Logistic Regression and Random Forest."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    class_labels = ["No Churn (0)", "Churn (1)"]

    for ax, (model_key, title) in zip(
        axes,
        [
            ("logistic_regression", "Logistic Regression (thr=0.50)"),
            ("random_forest", "Random Forest (thr=0.50)"),
        ],
    ):
        cm = np.asarray(
            evaluation_results[model_key]["default_threshold_metrics"]["confusion_matrix"],
            dtype=int,
        )
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)

        thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
        tag_names = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{tag_names[i][j]}\n{cm[i, j]}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "#1a1a1a",
                    fontsize=10,
                    fontweight="bold",
                )

    fig.suptitle("Held-Out Test Set Confusion Matrices", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curves(
    evaluation_results: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Plot ROC curves and AUC scores for both models."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    styles = {
        "logistic_regression": ("Logistic Regression", "#1b4f72", "-"),
        "random_forest": ("Random Forest", "#d35400", "--"),
    }

    for key, (label, color, ls) in styles.items():
        roc_data = evaluation_results[key]["roc_curve"]
        auc_val = evaluation_results[key]["default_threshold_metrics"]["roc_auc"]
        ax.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            linestyle=ls,
            linewidth=2.2,
            color=color,
            label=f"{label} (ROC-AUC = {auc_val:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle=":", color="#666666", label="Random Classifier (0.500)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=11, fontweight="bold")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate / Recall (TPR)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_feature_importance(
    evaluation_results: Dict[str, Any],
    output_path: Union[str, Path],
    top_n: int = 10,
) -> Path:
    """Plot held-out permutation importance (ROC-AUC drop) and Random Forest Gini importance."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rf_imp = evaluation_results["random_forest"]["feature_importance"]
    perm_list = rf_imp["permutation_importance"][:top_n][::-1]
    gini_list = rf_imp["internal_transformed_importance"][:top_n][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # Left: Held-out Permutation Importance
    p_names = [r["feature"] for r in perm_list]
    p_vals = [r["importance_mean"] for r in perm_list]
    p_errs = [r["importance_std"] for r in perm_list]
    axes[0].barh(p_names, p_vals, xerr=p_errs, color="#1b4f72", alpha=0.9)
    axes[0].set_title(
        "Held-Out Permutation Importance (ROC-AUC Drop)",
        fontsize=10,
        fontweight="bold",
    )
    axes[0].set_xlabel("Mean Decrease in Test ROC-AUC")
    axes[0].grid(True, axis="x", linestyle="--", alpha=0.4)

    # Right: Random Forest Impurity (Gini) Importance
    g_names = [r["transformed_feature"].replace("num__", "").replace("cat__", "") for r in gini_list]
    g_vals = [r["importance"] for r in gini_list]
    axes[1].barh(g_names, g_vals, color="#d35400", alpha=0.9)
    axes[1].set_title(
        "Random Forest Impurity (Gini) Importance (Top 10)",
        fontsize=10,
        fontweight="bold",
    )
    axes[1].set_xlabel("Relative Gini Importance")
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Feature Importance Analysis (Predictive Association, Not Causation)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_threshold_tradeoff(
    evaluation_results: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Plot validation Precision, Recall, and F1 across classification thresholds."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)

    for ax, (model_key, title) in zip(
        axes,
        [
            ("logistic_regression", "Logistic Regression (Validation Split)"),
            ("random_forest", "Random Forest (Validation Split)"),
        ],
    ):
        curve = evaluation_results[model_key]["validation_threshold_curve"]
        sel_thr = evaluation_results[model_key]["validation_selected_threshold"]
        thrs = [r["threshold"] for r in curve]
        precs = [r["precision"] for r in curve]
        recs = [r["recall"] for r in curve]
        f1s = [r["f1"] for r in curve]

        ax.plot(thrs, precs, label="Precision", color="#1b4f72", linewidth=2)
        ax.plot(thrs, recs, label="Recall", color="#2e8b57", linewidth=2)
        ax.plot(thrs, f1s, label="F1-Score", color="#d35400", linewidth=2)
        ax.axvline(0.50, color="#555555", linestyle=":", label="Default (0.50)")
        ax.axvline(sel_thr, color="#c0392b", linestyle="--", label=f"Selected ({sel_thr:.2f})")

        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Classification Probability Threshold")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, frameon=True)

    fig.suptitle(
        "Precision-Recall-F1 Threshold Analysis (Tuned Strictly on Validation Split)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
