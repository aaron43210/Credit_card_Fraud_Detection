# PROJECT Fraud Detection Package

Professional, cleaned project package with only required assets and model scope.

## Model Scope
- Decision Tree
- XGBoost
- HGNN (including DenseHGNN ATT-TD artifact)

Only the three approved models are used in this package.

## Folder Layout

- app.py: Streamlit application for batch prediction and artifact status
- src/: Core training, preprocessing, evaluation, and utilities
- notebooks/local.ipynb: Main notebook workflow
- models/: Required model and preprocessing artifacts
- FINAL/: Shared final team deliverables
- requirements.txt: Python dependencies

## Required Model Artifacts

- models/decision_tree.pkl
- models/xgboost_model.pkl
- models/hgnn_model.pt
- models/hgnn_att_td.pt
- models/scaler.pkl
- models/feature_names.pkl
- models/label_encoders.pkl

## Run

From inside PROJECT:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## HGNN Model Details

Two HGNN model checkpoints are included:

### hgnn_att_td.pt (Recommended) ⭐
- **Size**: 328 KB (optimized and compact)
- **Features**: 432 engineered features
- **Architecture**: DenseHGNN with Attention & Time Decay
- **Key Advantage**: More efficient (23x smaller), sophisticated temporal modeling
- **Status**: Active in app.py

### hgnn_model.pt
- **Size**: 7.6 MB
- **Features**: 217 engineered features  
- **Architecture**: FraudHGNN (standard heterogeneous graph transformer)
- **Note**: Legacy model, larger footprint

**Recommendation**: Use `hgnn_att_td.pt` for production - it's more efficient while using richer features (432 vs 217) and includes advanced temporal decay modeling for fraud patterns.

## Notes

- app.py supports model selection: Decision Tree, XGBoost, or HGNN Attention TD.
- Batch output shows per-transaction prediction details for every uploaded row.
- HGNN supports unlabeled batch inference and preserves uploaded row order in output.
- Models require specific feature engineering during preprocessing (run full pipeline for inference).
