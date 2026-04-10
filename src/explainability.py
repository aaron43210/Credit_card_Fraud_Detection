"""
explainability.py — Stage 8: Model Interpretation & Explainability
=====================================================================
SHAP analysis, feature importance comparison, partial dependence plots,
and decision tree visualization for interpretable fraud detection.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import torch
from sklearn.tree import export_graphviz
from sklearn.inspection import PartialDependenceDisplay
from src.config import EXPLAIN_DIR, COLORS, FIGSIZE_LARGE, FIGSIZE_MEDIUM
from src.utils import get_logger, set_plot_style, save_figure, Timer

logger = get_logger("Explainability")


def compute_shap_tree(model, X_sample, model_name="Model"):
    """
    Compute SHAP values for tree-based models (DT, XGBoost).
    Uses TreeExplainer for exact, fast SHAP computation.
    """
    logger.info(f"  Computing SHAP values for {model_name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return explainer, shap_values


def compute_shap_nn(model, X_sample):
    """
    Compute SHAP values for neural network using KernelExplainer.
    Uses a subsample for background data (KernelSHAP is slow).
    """
    logger.info("  Computing SHAP values for Neural Network (KernelExplainer)...")
    device = next(model.parameters()).device
    model.eval()

    def predict_fn(x):
        with torch.no_grad():
            tensor = torch.FloatTensor(x).to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        return probs

    # Use small background for speed
    bg_size = min(100, len(X_sample))
    background = X_sample[:bg_size]
    explainer = shap.KernelExplainer(predict_fn, background.values if isinstance(background, pd.DataFrame) else background)

    sample_size = min(200, len(X_sample))
    X_explain = X_sample[:sample_size]
    shap_values = explainer.shap_values(X_explain.values if isinstance(X_explain, pd.DataFrame) else X_explain, nsamples=100)

    return explainer, shap_values


def plot_shap_summary(shap_values, X_sample, model_name, max_display=20):
    """Plot SHAP summary (beeswarm) plot."""
    set_plot_style()
    plt.style.use("default")  # SHAP plots better with default style

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
    shap.summary_plot(shap_values, X_sample, max_display=max_display,
                      show=False, plot_size=None)
    plt.title(f"SHAP Feature Importance — {model_name}",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(EXPLAIN_DIR, f"01_shap_summary_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✅ SHAP summary plot saved for {model_name}")

    set_plot_style()  # Restore dark style


def plot_shap_bar(shap_values, X_sample, model_name, max_display=20):
    """Plot SHAP mean absolute value bar chart."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    shap.summary_plot(shap_values, X_sample, plot_type="bar",
                      max_display=max_display, show=False, plot_size=None)
    plt.title(f"Mean |SHAP Value| — {model_name}",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(EXPLAIN_DIR, f"02_shap_bar_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✅ SHAP bar plot saved for {model_name}")
    set_plot_style()


def plot_feature_importance_comparison(models: dict, feature_names: list):
    """
    Compare feature importance across all models side-by-side.
    Uses built-in importance for tree models and permutation for NN.
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    importance_data = {}

    # Decision Tree importance
    if "Decision Tree" in models:
        dt = models["Decision Tree"]
        imp = pd.Series(dt.feature_importances_, index=feature_names)
        importance_data["Decision Tree"] = imp.sort_values(ascending=False).head(15)

    # XGBoost importance
    if "XGBoost" in models:
        xgb = models["XGBoost"]
        imp = pd.Series(xgb.feature_importances_, index=feature_names)
        importance_data["XGBoost"] = imp.sort_values(ascending=False).head(15)

    colors_list = [COLORS["accent"], COLORS["primary"]]
    for i, (name, imp) in enumerate(importance_data.items()):
        axes[i].barh(range(len(imp)), imp.values[::-1], color=colors_list[i],
                    edgecolor="white", linewidth=0.3)
        axes[i].set_yticks(range(len(imp)))
        axes[i].set_yticklabels(imp.index[::-1], fontsize=9)
        axes[i].set_title(f"{name} — Top 15 Features", fontweight="bold")
        axes[i].set_xlabel("Feature Importance")

    fig.suptitle("Feature Importance Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(EXPLAIN_DIR, "03_feature_importance_comparison.png"))
    logger.info("  ✅ Feature importance comparison saved")


def plot_decision_tree_structure(model, feature_names, max_depth=4):
    """Export decision tree visualization (limited depth)."""
    try:
        from sklearn.tree import plot_tree
        set_plot_style()
        fig, ax = plt.subplots(figsize=(24, 12))
        plot_tree(model, max_depth=max_depth, feature_names=feature_names,
                  class_names=["Legitimate", "Fraud"], filled=True,
                  rounded=True, ax=ax, fontsize=7, proportion=True)
        ax.set_title(f"Decision Tree Structure (max_depth={max_depth} shown)",
                     fontsize=16, fontweight="bold")
        save_figure(fig, os.path.join(EXPLAIN_DIR, "04_decision_tree_structure.png"))
        logger.info("  ✅ Decision tree structure saved")
    except Exception as e:
        logger.warning(f"  ⚠️ Could not plot tree: {e}")


def plot_partial_dependence(model, X_sample, feature_names, top_n=6, model_name="XGBoost"):
    """
    Plot partial dependence for top features.
    Shows how each feature influences the prediction marginally.
    """
    try:
        # Get top features by importance
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_names)
            top_features = imp.sort_values(ascending=False).head(top_n).index.tolist()
            top_indices = [feature_names.index(f) for f in top_features]
        else:
            top_indices = list(range(min(top_n, len(feature_names))))

        set_plot_style()
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        display = PartialDependenceDisplay.from_estimator(
            model, X_sample, top_indices,
            feature_names=feature_names, ax=axes,
            kind="average", grid_resolution=50,
        )
        fig.suptitle(f"Partial Dependence Plots — {model_name}",
                     fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_figure(fig, os.path.join(EXPLAIN_DIR, f"05_partial_dependence_{model_name.lower().replace(' ', '_')}.png"))
        logger.info(f"  ✅ Partial dependence plots saved for {model_name}")
    except Exception as e:
        logger.warning(f"  ⚠️ Could not plot PDP for {model_name}: {e}")


def run_explainability_pipeline(models: dict, X_test, feature_names: list):
    """
    Execute the full explainability pipeline.
    """
    os.makedirs(EXPLAIN_DIR, exist_ok=True)
    logger.info("=" * 60)
    logger.info("     EXPLAINABILITY PIPELINE")
    logger.info("=" * 60)

    # Use a sample for SHAP (full dataset is too large)
    sample_size = min(1000, len(X_test))
    X_sample = X_test.iloc[:sample_size] if isinstance(X_test, pd.DataFrame) else X_test[:sample_size]

    # SHAP for Decision Tree
    if "Decision Tree" in models:
        with Timer("SHAP — Decision Tree", logger):
            _, shap_vals = compute_shap_tree(models["Decision Tree"], X_sample, "Decision Tree")
            plot_shap_summary(shap_vals, X_sample, "Decision Tree")
            plot_shap_bar(shap_vals, X_sample, "Decision Tree")
            plot_decision_tree_structure(models["Decision Tree"], feature_names)

    # SHAP for XGBoost
    if "XGBoost" in models:
        with Timer("SHAP — XGBoost", logger):
            _, shap_vals = compute_shap_tree(models["XGBoost"], X_sample, "XGBoost")
            plot_shap_summary(shap_vals, X_sample, "XGBoost")
            plot_shap_bar(shap_vals, X_sample, "XGBoost")
            plot_partial_dependence(models["XGBoost"], X_sample, feature_names, model_name="XGBoost")

    # Feature importance comparison
    plot_feature_importance_comparison(models, feature_names)

    logger.info(f"\n  🔍 All explainability plots saved to {EXPLAIN_DIR}")
