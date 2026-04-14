"""
config.py — Central Configuration for Credit Card Fraud Detection System
=========================================================================
Cross-platform compatible configuration with pathlib for Windows and Unix.
All paths, constants, hyperparameters, and feature definitions are stored
here so that every module in the project reads from a single source of truth.
"""

from pathlib import Path

# ──────────────────────────── Paths (Cross-Platform with pathlib) ────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "ieee-fraud-detection"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
EVAL_DIR = OUTPUT_DIR / "evaluation"
EXPLAIN_DIR = OUTPUT_DIR / "explainability"
HGNN_DIR = OUTPUT_DIR / "hgnn"
CONTRIBUTORS_DIR = PROJECT_ROOT / "individual_profiles"

# Create directories if they don't exist
for dir_path in [MODEL_DIR, EDA_DIR, EVAL_DIR, EXPLAIN_DIR, HGNN_DIR, CONTRIBUTORS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data files
TRAIN_TRANSACTION = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY = DATA_DIR / "train_identity.csv"
TEST_TRANSACTION = DATA_DIR / "test_transaction.csv"
TEST_IDENTITY = DATA_DIR / "test_identity.csv"

# Model artifact paths
DT_MODEL_PATH = MODEL_DIR / "decision_tree.pkl"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
HGNN_ATT_TD_PATH = MODEL_DIR / "hgnn_att_td.pt"  # HGNN Attention TD model
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"

# ──────────────────────────── Dataset ────────────────────────────
TARGET_COL = "isFraud"
ID_COL = "TransactionID"
RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────── Feature Groups ────────────────────────────
CATEGORICAL_COLS = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
    "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18",
    "id_19", "id_20", "id_21", "id_22", "id_23", "id_24", "id_25",
    "id_26", "id_27", "id_28", "id_29", "id_30", "id_31", "id_32",
    "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
]

DROP_COLS = [ID_COL, "TransactionDT"]
MISSING_THRESHOLD = 0.50
AMT_CAP_PERCENTILE = 99

# ──────────────────────────── Hyperparameters ────────────────────────────
DT_PARAMS = {
    "max_depth": 12,
    "min_samples_split": 50,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "criterion": "gini",
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

NN_PARAMS = {
    "hidden_dims": [256, 128, 64],
    "dropout_rates": [0.3, 0.3, 0.2],
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 2048,
    "max_epochs": 50,
    "patience": 10,
    "focal_gamma": 2.0,
    "focal_alpha": 0.75,
}

# ──────────────────────────── HGNN (Heterogeneous Graph Neural Network) ────────────────────────────
HGNN_PARAMS = {
    "hidden_dims": [128, 64],
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 512,
    "max_epochs": 15,
    "patience": 5,
    "focal_gamma": 2.0,
    "focal_alpha": 0.75,
    "early_stopping": True,
}

# Device and Memory Settings (Local machine friendly)
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
DENSE_HGNN_MAX_NODES_CUDA = 12000
DENSE_HGNN_MAX_NODES_CPU = 3000

SMOTE_PARAMS = {
    "sampling_strategy": 0.5,
    "k_neighbors": 5,
    "random_state": RANDOM_STATE,
}

PCA_COMPONENTS = 50
N_FOLDS = 5

# ──────────────────────────── Cost-Benefit ────────────────────────────
COST_FN = 500
COST_FP = 10
COST_TP = -500
COST_TN = 0

# ──────────────────────────── Visualization ────────────────────────────
COLORS = {
    "fraud": "#FF4B4B",
    "legit": "#00D26A",
    "primary": "#6C63FF",
    "secondary": "#FF6B6B",
    "accent": "#4ECDC4",
    "bg_dark": "#0E1117",
    "bg_card": "#1A1F2E",
    "text": "#FAFAFA",
    "grid": "#2D3748",
}

FIGSIZE_LARGE = (16, 10)
FIGSIZE_MEDIUM = (12, 7)
FIGSIZE_SMALL = (8, 5)
DPI = 150
