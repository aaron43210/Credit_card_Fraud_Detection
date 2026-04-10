"""
evaluation.py — Stage 7: Model Evaluation & Comparison
=========================================================
Comprehensive evaluation with precision-recall focus, cost-benefit
analysis, and side-by-side model comparison visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
)
from src.config import (
    TARGET_COL, EVAL_DIR, COLORS, FIGSIZE_LARGE, FIGSIZE_MEDIUM, DPI,
    COST_FN, COST_FP, COST_TP, COST_TN,
)
from src.utils import get_logger, set_plot_style, save_figure

logger = get_logger("Evaluation")


def get_predictions(model, X, model_name="model", hetero_data=None):
    """
    Get probability predictions from any model type (sklearn or PyTorch).
    """
    if isinstance(model, torch.nn.Module):
        model.eval()
        device = next(model.parameters()).device

        # Graph-based inference path when hetero_data is provided.
        # If model is a tabular torch model (e.g., FraudNet MLP), this call
        # raises TypeError and we fall back to dense feature inference below.
        if hetero_data is not None:
            try:
                hetero_data = hetero_data.to(device)
                with torch.no_grad():
                    tx_decay = getattr(hetero_data['transaction'], 'time_decay', None)
                    final_logits = model(
                        hetero_data.x_dict,
                        hetero_data.edge_index_dict,
                        tx_time_decay=tx_decay,
                    )
                    val_logits = final_logits[hetero_data['transaction'].test_mask]
                    return torch.sigmoid(val_logits).squeeze().cpu().numpy()
            except TypeError:
                pass

        # Tabular torch model path (e.g., FraudNet MLP).
        try:
            with torch.no_grad():
                X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
                X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
                if X_t.ndim == 1:
                    X_t = X_t.unsqueeze(0)
                logits = model(X_t)
                return torch.sigmoid(logits).squeeze().cpu().numpy()
        except TypeError as exc:
            raise ValueError(
                "hetero_data is required for graph-based neural network evaluation."
            ) from exc
    else:
        return model.predict_proba(X)[:, 1]


def compute_metrics(y_true, y_proba, threshold=0.5):
    """
    Compute comprehensive classification metrics.

    Returns
    -------
    dict
        Dictionary of all metrics including accuracy, precision,
        recall, F1, ROC-AUC, AUPRC, and confusion matrix counts.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "auprc": average_precision_score(y_true, y_proba),
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "total_cost": fn * COST_FN + fp * COST_FP + tp * COST_TP + tn * COST_TN,
    }
    return metrics


def find_optimal_threshold(y_true, y_proba, method="f1"):
    """
    Find the optimal classification threshold.

    Methods:
    - 'f1': Maximize F1-score
    - 'cost': Minimize total cost (FN=$500, FP=$10)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    if method == "f1":
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        logger.info(f"  Optimal threshold (max F1): {best_threshold:.4f} → F1={f1_scores[best_idx]:.4f}")
    elif method == "cost":
        costs = []
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (y_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            cost = fn * COST_FN + fp * COST_FP + tp * COST_TP + tn * COST_TN
            costs.append((t, cost))
        best_threshold, best_cost = min(costs, key=lambda x: x[1])
        logger.info(f"  Optimal threshold (min cost): {best_threshold:.4f} → Cost=${best_cost:,.0f}")

    return best_threshold


def evaluate_all_models(models: dict, X_test, y_test, hetero_data=None):
    """
    Evaluate all models on the test set and return comparison table.
    """
    logger.info("=" * 60)
    logger.info("     MODEL EVALUATION ON TEST SET")
    logger.info("=" * 60)

    results = {}
    probas = {}

    for name, model in models.items():
        logger.info(f"\n  📊 Evaluating: {name}")
        y_proba = get_predictions(model, X_test, model_name=name, hetero_data=hetero_data)
        probas[name] = y_proba

        # Find optimal threshold
        opt_threshold = find_optimal_threshold(y_test, y_proba, method="f1")

        # Compute metrics at both default and optimal threshold
        metrics_default = compute_metrics(y_test, y_proba, threshold=0.5)
        metrics_optimal = compute_metrics(y_test, y_proba, threshold=opt_threshold)

        results[name] = {
            "default": metrics_default,
            "optimal": metrics_optimal,
            "optimal_threshold": opt_threshold,
        }

        logger.info(f"    @ threshold=0.50: Precision={metrics_default['precision']:.4f}, "
                    f"Recall={metrics_default['recall']:.4f}, F1={metrics_default['f1_score']:.4f}")
        logger.info(f"    @ threshold={opt_threshold:.2f}: Precision={metrics_optimal['precision']:.4f}, "
                    f"Recall={metrics_optimal['recall']:.4f}, F1={metrics_optimal['f1_score']:.4f}")

    return results, probas


def plot_precision_recall_curves(y_test, probas: dict):
    """Plot PR curves for all models overlaid."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)

    model_colors = {
        "Decision Tree": COLORS["accent"],
        "XGBoost": COLORS["primary"],
        "Neural Network": COLORS["fraud"],
    }

    for name, y_proba in probas.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        auprc = average_precision_score(y_test, y_proba)
        color = model_colors.get(name, COLORS["secondary"])
        ax.plot(recall, precision, color=color, linewidth=2,
                label=f"{name} (AUPRC={auprc:.4f})")

    # Baseline (random classifier)
    baseline = y_test.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5,
               label=f"Random (AUPRC={baseline:.4f})")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Model Comparison",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, os.path.join(EVAL_DIR, "01_precision_recall_curves.png"))
    logger.info("  ✅ Precision-Recall curves saved")


def plot_roc_curves(y_test, probas: dict):
    """Plot ROC curves for all models overlaid."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)

    model_colors = {
        "Decision Tree": COLORS["accent"],
        "XGBoost": COLORS["primary"],
        "Neural Network": COLORS["fraud"],
    }

    for name, y_proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        color = model_colors.get(name, COLORS["secondary"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, os.path.join(EVAL_DIR, "02_roc_curves.png"))
    logger.info("  ✅ ROC curves saved")


def plot_confusion_matrices(y_test, probas: dict, results: dict):
    """Plot confusion matrices side by side for all models."""
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, (name, y_proba) in enumerate(probas.items()):
        threshold = results[name]["optimal_threshold"]
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt=",d", cmap="YlOrRd",
                    xticklabels=["Legitimate", "Fraud"],
                    yticklabels=["Legitimate", "Fraud"],
                    ax=axes[i], annot_kws={"size": 14})
        axes[i].set_title(f"{name}\n(threshold={threshold:.2f})",
                         fontsize=12, fontweight="bold")
        axes[i].set_ylabel("Actual")
        axes[i].set_xlabel("Predicted")

    fig.suptitle("Confusion Matrices — Optimal Thresholds",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(EVAL_DIR, "03_confusion_matrices.png"))
    logger.info("  ✅ Confusion matrices saved")


def plot_cost_benefit_analysis(y_test, probas: dict):
    """
    Cost-benefit analysis: plot total cost vs. threshold for each model.

    Costs: FN=$500 (missed fraud), FP=$10 (false alarm),
           TP=-$500 (caught fraud = saved loss), TN=$0
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)

    model_colors = {
        "Decision Tree": COLORS["accent"],
        "XGBoost": COLORS["primary"],
        "Neural Network": COLORS["fraud"],
    }

    thresholds = np.arange(0.05, 0.95, 0.01)

    for name, y_proba in probas.items():
        costs = []
        for t in thresholds:
            preds = (y_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            cost = fn * COST_FN + fp * COST_FP + tp * COST_TP + tn * COST_TN
            costs.append(cost)

        color = model_colors.get(name, COLORS["secondary"])
        ax.plot(thresholds, costs, color=color, linewidth=2, label=name)

        # Mark minimum cost point
        min_idx = np.argmin(costs)
        ax.scatter(thresholds[min_idx], costs[min_idx], color=color,
                   s=100, zorder=5, edgecolors="white")
        ax.annotate(f"${costs[min_idx]:,.0f}\nt={thresholds[min_idx]:.2f}",
                   (thresholds[min_idx], costs[min_idx]),
                   textcoords="offset points", xytext=(10, -20),
                   fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Total Cost ($)", fontsize=12)
    ax.set_title(f"Cost-Benefit Analysis\n(FN=${COST_FN}, FP=${COST_FP}, TP=${COST_TP})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, os.path.join(EVAL_DIR, "04_cost_benefit_analysis.png"))
    logger.info("  ✅ Cost-benefit analysis saved")


def plot_model_comparison_table(results: dict):
    """Create a summary comparison table and save as image."""
    set_plot_style()
    rows = []
    for name, res in results.items():
        opt = res["optimal"]
        rows.append({
            "Model": name,
            "Accuracy": f"{opt['accuracy']:.4f}",
            "Precision": f"{opt['precision']:.4f}",
            "Recall": f"{opt['recall']:.4f}",
            "F1-Score": f"{opt['f1_score']:.4f}",
            "ROC-AUC": f"{opt['roc_auc']:.4f}",
            "AUPRC": f"{opt['auprc']:.4f}",
            "Threshold": f"{res['optimal_threshold']:.2f}",
            "FN (Missed)": opt['fn'],
            "FP (False Alarm)": opt['fp'],
            "Total Cost ($)": f"${opt['total_cost']:,}",
        })
    comp_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.axis("off")
    table = ax.table(
        cellText=comp_df.values,
        colLabels=comp_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4A5568")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(COLORS["bg_card"])
            cell.set_text_props(color=COLORS["text"])
        cell.set_edgecolor(COLORS["grid"])

    ax.set_title("Model Performance Comparison (Optimal Thresholds)",
                 fontsize=14, fontweight="bold", pad=20, color=COLORS["text"])

    plt.tight_layout()
    save_figure(fig, os.path.join(EVAL_DIR, "05_comparison_table.png"))
    logger.info("  ✅ Comparison table saved")

    return comp_df


def run_evaluation_pipeline(models: dict, X_test, y_test, hetero_data=None):
    """
    Execute the complete evaluation pipeline.
    """
    os.makedirs(EVAL_DIR, exist_ok=True)
    logger.info("=" * 60)
    logger.info("     EVALUATION PIPELINE")
    logger.info("=" * 60)

    results, probas = evaluate_all_models(models, X_test, y_test, hetero_data=hetero_data)

    plot_precision_recall_curves(y_test, probas)
    plot_roc_curves(y_test, probas)
    plot_confusion_matrices(y_test, probas, results)
    plot_cost_benefit_analysis(y_test, probas)
    comp_df = plot_model_comparison_table(results)

    logger.info(f"\n  📊 All evaluation plots saved to {EVAL_DIR}")
    return results, probas, comp_df
