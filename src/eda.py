"""
eda.py — Stage 4: Exploratory Data Analysis
==============================================
Generates publication-quality EDA visualizations for the IEEE-CIS
Fraud Detection dataset. All plots are saved to outputs/eda/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from src.config import TARGET_COL, EDA_DIR, COLORS, FIGSIZE_LARGE, FIGSIZE_MEDIUM, DPI
from src.utils import get_logger, set_plot_style, save_figure, format_number, format_pct

logger = get_logger("EDA")


def plot_class_distribution(df: pd.DataFrame):
    """Plot 1: Class distribution — bar + pie showing severe imbalance."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    counts = df[TARGET_COL].value_counts()
    labels = ["Legitimate", "Fraudulent"]
    colors = [COLORS["legit"], COLORS["fraud"]]

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.5, width=0.6)
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                     format_number(count), ha="center", va="bottom", fontweight="bold", fontsize=13)
    axes[0].set_title("Transaction Counts by Class", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Transactions")
    axes[0].grid(axis="y", alpha=0.3)

    # Pie chart
    fraud_pct = counts[1] / counts.sum() * 100
    legit_pct = counts[0] / counts.sum() * 100
    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=labels, colors=colors, autopct="%1.2f%%",
        startangle=90, explode=(0, 0.1), shadow=True,
        textprops={"color": COLORS["text"], "fontsize": 12}
    )
    autotexts[1].set_fontweight("bold")
    axes[1].set_title("Class Imbalance Visualization", fontsize=14, fontweight="bold")

    fig.suptitle("Severe Class Imbalance in IEEE-CIS Fraud Dataset",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(EDA_DIR, "01_class_distribution.png"))
    logger.info("  ✅ Class distribution plot saved")


def plot_transaction_amount(df: pd.DataFrame):
    """Plot 2: Transaction amount distribution by class (log scale)."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    for cls, color, label in [(0, COLORS["legit"], "Legitimate"), (1, COLORS["fraud"], "Fraudulent")]:
        subset = df[df[TARGET_COL] == cls]["TransactionAmt"]
        axes[0].hist(subset, bins=100, alpha=0.7, color=color, label=label, density=True)

    axes[0].set_xlabel("Transaction Amount ($)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Amount Distribution (Linear Scale)")
    axes[0].legend()
    axes[0].set_xlim(0, df["TransactionAmt"].quantile(0.99))

    # Log scale
    for cls, color, label in [(0, COLORS["legit"], "Legitimate"), (1, COLORS["fraud"], "Fraudulent")]:
        subset = df[df[TARGET_COL] == cls]["TransactionAmt"].clip(lower=1)
        axes[1].hist(np.log1p(subset), bins=100, alpha=0.7, color=color, label=label, density=True)

    axes[1].set_xlabel("log(Transaction Amount + 1)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Amount Distribution (Log Scale)")
    axes[1].legend()

    fig.suptitle("Transaction Amount Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(EDA_DIR, "02_transaction_amount.png"))
    logger.info("  ✅ Transaction amount plot saved")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot 3: Correlation heatmap of top features with target."""
    set_plot_style()
    numeric_df = df.select_dtypes(include=[np.number])
    corr_with_target = numeric_df.corr()[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
    top_features = corr_with_target.head(25).index.tolist()
    top_features.append(TARGET_COL)

    corr_matrix = numeric_df[top_features].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7},
                cbar_kws={"label": "Correlation Coefficient"})
    ax.set_title("Correlation Heatmap — Top 25 Features vs isFraud",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, os.path.join(EDA_DIR, "03_correlation_heatmap.png"))
    logger.info("  ✅ Correlation heatmap saved")


def plot_missing_values(df: pd.DataFrame):
    """Plot 4: Missing value heatmap by feature group."""
    set_plot_style()
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]

    if len(missing) == 0:
        logger.info("  ⚠️ No missing values — skipping plot")
        return

    # Take top 40 columns with most missing
    top_missing = missing.head(40)

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(top_missing)), top_missing.values,
                   color=COLORS["secondary"], edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(top_missing)))
    ax.set_yticklabels(top_missing.index, fontsize=8)
    ax.set_xlabel("Fraction Missing")
    ax.set_title(f"Top {len(top_missing)} Columns by Missing Value Percentage",
                 fontsize=14, fontweight="bold")
    ax.axvline(x=0.5, color=COLORS["fraud"], linestyle="--", label="50% threshold")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    save_figure(fig, os.path.join(EDA_DIR, "04_missing_values.png"))
    logger.info("  ✅ Missing values plot saved")


def plot_time_pattern(df: pd.DataFrame):
    """Plot 5: Fraud rate over time (TransactionDT)."""
    set_plot_style()
    if "TransactionDT" not in df.columns:
        logger.info("  ⚠️ TransactionDT not found — skipping time pattern plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_MEDIUM)

    # Convert TransactionDT to hours
    df_temp = df.copy()
    df_temp["hour"] = (df_temp["TransactionDT"] / 3600).astype(int)
    hourly = df_temp.groupby("hour").agg(
        total=("isFraud", "count"), fraud=("isFraud", "sum")
    )
    hourly["fraud_rate"] = hourly["fraud"] / hourly["total"]

    axes[0].plot(hourly.index, hourly["total"], color=COLORS["primary"], alpha=0.7, linewidth=0.5)
    axes[0].fill_between(hourly.index, hourly["total"], alpha=0.3, color=COLORS["primary"])
    axes[0].set_title("Transaction Volume Over Time", fontweight="bold")
    axes[0].set_ylabel("Transactions per Hour")

    axes[1].plot(hourly.index, hourly["fraud_rate"], color=COLORS["fraud"], alpha=0.8, linewidth=0.5)
    axes[1].fill_between(hourly.index, hourly["fraud_rate"], alpha=0.2, color=COLORS["fraud"])
    axes[1].set_title("Fraud Rate Over Time", fontweight="bold")
    axes[1].set_ylabel("Fraud Rate")
    axes[1].set_xlabel("Time (hours since start)")

    fig.suptitle("Temporal Patterns in Fraud Detection", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(EDA_DIR, "05_time_pattern.png"))
    logger.info("  ✅ Time pattern plot saved")


def plot_product_analysis(df: pd.DataFrame):
    """Plot 6: Fraud rate by ProductCD."""
    set_plot_style()
    if "ProductCD" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    product_stats = df.groupby("ProductCD").agg(
        count=(TARGET_COL, "count"), fraud=(TARGET_COL, "sum")
    )
    product_stats["fraud_rate"] = product_stats["fraud"] / product_stats["count"]
    product_stats = product_stats.sort_values("count", ascending=False)

    bars = axes[0].bar(product_stats.index.astype(str), product_stats["count"],
                       color=COLORS["primary"], edgecolor="white")
    axes[0].set_title("Transaction Count by Product", fontweight="bold")
    axes[0].set_ylabel("Count")

    bars = axes[1].bar(product_stats.index.astype(str), product_stats["fraud_rate"],
                       color=COLORS["fraud"], edgecolor="white")
    axes[1].set_title("Fraud Rate by Product", fontweight="bold")
    axes[1].set_ylabel("Fraud Rate")
    for bar, val in zip(bars, product_stats["fraud_rate"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.2%}", ha="center", fontsize=10)

    fig.suptitle("Product Category Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(EDA_DIR, "06_product_analysis.png"))
    logger.info("  ✅ Product analysis plot saved")


def plot_card_analysis(df: pd.DataFrame):
    """Plot 7: Fraud rate by card type (card4 and card6)."""
    set_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    for i, col in enumerate(["card4", "card6"]):
        if col not in df.columns:
            continue
        card_stats = df.groupby(col).agg(
            count=(TARGET_COL, "count"), fraud=(TARGET_COL, "sum")
        )
        card_stats["fraud_rate"] = card_stats["fraud"] / card_stats["count"]
        card_stats = card_stats[card_stats["count"] > 100].sort_values("fraud_rate", ascending=False)

        bars = axes[i].bar(card_stats.index.astype(str), card_stats["fraud_rate"],
                           color=[COLORS["fraud"] if r > 0.05 else COLORS["accent"] for r in card_stats["fraud_rate"]],
                           edgecolor="white")
        axes[i].set_title(f"Fraud Rate by {col}", fontweight="bold")
        axes[i].set_ylabel("Fraud Rate")
        axes[i].tick_params(axis="x", rotation=45)

    fig.suptitle("Card Type Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(EDA_DIR, "07_card_analysis.png"))
    logger.info("  ✅ Card analysis plot saved")


def plot_email_analysis(df: pd.DataFrame):
    """Plot 8: Top email domains and their fraud rates."""
    set_plot_style()
    if "P_emaildomain" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    email_stats = df.groupby("P_emaildomain").agg(
        count=(TARGET_COL, "count"), fraud=(TARGET_COL, "sum")
    )
    email_stats["fraud_rate"] = email_stats["fraud"] / email_stats["count"]

    # Top 10 by volume
    top10 = email_stats.sort_values("count", ascending=False).head(10)
    axes[0].barh(top10.index.astype(str), top10["count"], color=COLORS["primary"])
    axes[0].set_title("Top 10 Email Domains by Volume", fontweight="bold")
    axes[0].invert_yaxis()

    # Top 10 by fraud rate (min 100 transactions)
    top_fraud = email_stats[email_stats["count"] > 100].sort_values("fraud_rate", ascending=False).head(10)
    axes[1].barh(top_fraud.index.astype(str), top_fraud["fraud_rate"], color=COLORS["fraud"])
    axes[1].set_title("Top 10 Email Domains by Fraud Rate", fontweight="bold")
    axes[1].invert_yaxis()

    fig.suptitle("Email Domain Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(EDA_DIR, "08_email_analysis.png"))
    logger.info("  ✅ Email analysis plot saved")


def plot_feature_distributions(df: pd.DataFrame):
    """Plot 9: Box plots of select V-features by class."""
    set_plot_style()
    v_features = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()][:10]
    if not v_features:
        return

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, col in enumerate(v_features):
        data_legit = df[df[TARGET_COL] == 0][col].dropna()
        data_fraud = df[df[TARGET_COL] == 1][col].dropna()
        bp = axes[i].boxplot(
            [data_legit.values, data_fraud.values],
            labels=["Legit", "Fraud"],
            patch_artist=True,
            boxprops=dict(linewidth=0.5),
            medianprops=dict(color="white", linewidth=1.5),
        )
        bp["boxes"][0].set_facecolor(COLORS["legit"])
        bp["boxes"][1].set_facecolor(COLORS["fraud"])
        axes[i].set_title(col, fontsize=10, fontweight="bold")
        axes[i].tick_params(labelsize=8)

    fig.suptitle("Feature Distributions: Legitimate vs Fraudulent (V1-V10)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(EDA_DIR, "09_feature_distributions.png"))
    logger.info("  ✅ Feature distributions plot saved")


def plot_amount_boxplot(df: pd.DataFrame):
    """Plot 10: Statistical comparison of transaction amounts per class."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)

    data = [
        df[df[TARGET_COL] == 0]["TransactionAmt"].values,
        df[df[TARGET_COL] == 1]["TransactionAmt"].values,
    ]
    bp = ax.boxplot(data, labels=["Legitimate", "Fraudulent"], patch_artist=True,
                    showfliers=False, medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(COLORS["legit"])
    bp["boxes"][1].set_facecolor(COLORS["fraud"])
    ax.set_ylabel("Transaction Amount ($)")
    ax.set_title("Transaction Amount by Class (Outliers Hidden)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, os.path.join(EDA_DIR, "10_amount_boxplot.png"))
    logger.info("  ✅ Amount boxplot saved")


def run_eda_pipeline(df: pd.DataFrame):
    """Execute all EDA visualizations."""
    os.makedirs(EDA_DIR, exist_ok=True)
    logger.info("=" * 60)
    logger.info("     EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    plot_class_distribution(df)
    plot_transaction_amount(df)
    plot_correlation_heatmap(df)
    plot_missing_values(df)
    plot_time_pattern(df)
    plot_product_analysis(df)
    plot_card_analysis(df)
    plot_email_analysis(df)
    plot_feature_distributions(df)
    plot_amount_boxplot(df)

    logger.info(f"  📊 All EDA plots saved to {EDA_DIR}")
    return EDA_DIR


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    df = load_raw_data(sample_frac=0.1)
    run_eda_pipeline(df)
