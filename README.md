# Credit Card Fraud Detection

A complete end-to-end binary classification pipeline for the **IEEE-CIS Fraud Detection** dataset with three model options: Decision Tree, XGBoost, and a Heterogeneous Graph Neural Network (HGNN).

This project compares three model architectures on a highly imbalanced dataset (~3.5% fraud rate):

| Model | Type | Imbalance Strategy |
|---|---|---|
| Decision Tree | Interpretable baseline | `class_weight='balanced'` + SMOTE |
| XGBoost | Gradient boosting | `scale_pos_weight` + SMOTE |
| **HGNN-ATT-TD** ⭐ | Heterogeneous Graph Neural Network | FocalLoss + temporal decay |

The **HGNN-ATT-TD** (Heterogeneous Graph Neural Network with Attention and Temporal Decay) leverages the **relational graph structure** between transactions, payment cards, and devices — information that flat tabular models cannot exploit.

---

## Features

- **Three model options**: Users can select Decision Tree, XGBoost, HGNN, or run all three.
- **Streamlit UI**: Interactive CSV upload, model selection, and real-time fraud predictions.
- **CLI predictor**: Batch prediction via `predict_models_options.py` for automation.
- **Auto-expansion**: ID-only CSVs (e.g., sample submissions) auto-merge with full test features.
- **Crash resilience**: Subprocess isolation ensures one model failure doesn't crash the app.
- **Responsive rendering**: Large results are downloadable; UI shows previews only.

---

## Project Structure

```
fraud-detection/
├── README.md
├── .gitignore
├── requirements.txt
├── app.py                        # Streamlit web interface
├── predict_models_options.py     # CLI batch prediction script
├── src/                          # Core Python modules
│   ├── config.py                 # Paths, hyperparameters, constants
│   ├── utils.py                  # Logging, timers, plot helpers
│   ├── data_loader.py            # Load & merge IEEE CSVs
│   ├── feature_engineering.py    # Feature creation, PCA, MI selection
│   ├── preprocessing.py          # Cleaning, encoding, splitting
│   ├── models.py                 # DT, XGBoost, FraudHGNN definitions
│   ├── training.py               # Training loops
│   ├── evaluation.py             # Metrics, ROC/PR curves
│   └── hgnn_utils.py             # Graph construction
├── notebooks/                    # Training & exploration notebooks
│   ├── preprocessing.ipynb       # EDA & preprocessing
│   ├── 02_train_decision_tree.ipynb
│   ├── 03_train_xgboost.ipynb
│   ├── 04_train_hgnn.ipynb
│   └── 05_evaluation.ipynb
├── data/                         # Dataset location (see setup)
├── models/                       # Trained model artifacts (generated)
└── outputs/                      # Visualizations (generated)
```

---

## Setup

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection

python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 2. Download the Dataset

Download the **IEEE-CIS Fraud Detection** dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection):

```bash
# Place dataset files in relative ./data directory:
# - train_transaction.csv
# - train_identity.csv
# - test_transaction.csv
# - test_identity.csv

# Or set environment variable to custom location:
export IEEE_FRAUD_DATA_DIR="/path/to/ieee-fraud-detection"
```

### 3. Install PyTorch Geometric (for HGNN)

```bash
pip install torch_geometric torch_scatter torch_sparse
```

---

## Quick Start

### Streamlit Web UI (Interactive)

```bash
streamlit run app.py
```

Then:
1. Upload a CSV (transaction data or id-only like sample_submission.csv)
2. Select which model(s) to run
3. Click "Run Fraud Check"
4. Download full results as CSV

### CLI Prediction (Batch)

```bash
python predict_models_options.py \
  --input sample_submission.csv \
  --model all \
  --max-rows 0 \
  --output predictions.csv
```

**Options:**
- `--model`: `decision_tree`, `xgboost`, `hgnn`, or `all`
- `--max-rows`: Row limit (0 = all rows)
- `--output`: Output CSV path

---

## Model Training

Run individual training notebooks:

```bash
jupyter notebook notebooks/02_train_decision_tree.ipynb
jupyter notebook notebooks/03_train_xgboost.ipynb
jupyter notebook notebooks/04_train_hgnn.ipynb
jupyter notebook notebooks/05_evaluation.ipynb
```

Or use the preprocessing & training scripts from `src/`.

---

## HGNN Architecture

The flagship model uses **Heterogeneous Graph Transformer (HGT)** convolutions:

```
Transactions ──[used]──► Cards
     │
     └──[on]──► Devices

Per transaction:
  1. Linear projection → hidden dim (64-128)
  2. Temporal decay: weight = exp(-age / age_95th_percentile)
  3. HGTConv × 2 layers (4 attention heads)
  4. Residual + GELU + Dropout
  5. Linear classifier → sigmoid → fraud probability
```

**Loss Function:** FocalLoss (γ=2.0, α=0.75)

---

## Expected Performance (on 30% sample)

| Model | ROC-AUC | AUPRC | F1 |
|---|---|---|---|
| Decision Tree | ~0.85 | ~0.45 | ~0.50 |
| XGBoost | ~0.92 | ~0.62 | ~0.60 |
| HGNN-ATT-TD | ~0.94+ | ~0.68+ | ~0.63+ |

---

## Configuration

All configuration is centralized in `src/config.py`:

- **Hyperparameters**: DT max_depth, XGBoost n_estimators, HGNN hidden dims
- **Data paths**: Configured via `IEEE_FRAUD_DATA_DIR` environment variable
- **Feature constants**: MISSING_THRESHOLD, AMT_CAP_PERCENTILE, etc.

Edit `src/config.py` to adjust model hyperparameters or preprocessing logic.

---

## Environment Variables

```bash
# Required: Path to IEEE-CIS fraud dataset
export IEEE_FRAUD_DATA_DIR="/path/to/ieee-fraud-detection"

# Optional: Native library threading (macOS/Linux stability)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

---

## Known Issues

- **macOS Segfault (exit 139)**: Occurs with certain XGBoost + Streamlit configurations.
  - **Solution**: Subprocess isolation is built-in; predictions run in isolated child processes.
- **Large CSV unresponsiveness**: Tables/charts limited to first 2,000−5,000 rows for UI responsiveness.
  - **Solution**: Download full CSV for complete dataset analysis.

---

## Troubleshooting

**Import errors?**  
Ensure all packages are installed:  
```bash
pip install --upgrade -r requirements.txt
```

**HGNN crashes on startup?**  
HGNN loads only when selected. Use Decision Tree or XGBoost if issues persist.

**Dataset not found?**  
Check `IEEE_FRAUD_DATA_DIR` environment variable or place files in `./data/`.

---

## References

- [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection)
- [Heterogeneous Graph Transformer (HGT)](https://arxiv.org/abs/2003.01332) — Hu et al., 2020
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) — Lin et al., 2017
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813) — Chawla et al., 2002

---

## License

MIT
