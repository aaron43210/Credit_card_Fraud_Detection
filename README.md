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

## Notes

- Live batch scoring in app.py uses Decision Tree and XGBoost probabilities.
- HGNN is retained as the official neural model artifact for project training/evaluation workflows.
