"""
Test all three models (Decision Tree, XGBoost, HGNN_ATT_TD) on real IEEE-CIS dataset.
Uses test_transaction.csv and test_identity.csv from ieee-fraud-detection directory.
"""

import os
import sys
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import signal

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_DIR, DT_MODEL_PATH, XGB_MODEL_PATH, HGNN_ATT_TD_PATH,
    SCALER_PATH, LABEL_ENCODERS_PATH, FEATURE_NAMES_PATH
)
from src.utils import get_logger
from src.preprocessing import handle_missing_values, encode_categoricals
from src.feature_engineering import create_time_features, create_amount_features

logger = get_logger("RealDataTest")

# Data paths
DATA_DIR = Path("/Users/aaronr/Desktop/untitled folder/ieee-fraud-detection")
TEST_TX_PATH = DATA_DIR / "test_transaction.csv"
TEST_ID_PATH = DATA_DIR / "test_identity.csv"

print(f"\n{'='*80}")
print(f"🧪 Testing Models on Real IEEE-CIS Dataset")
print(f"{'='*80}\n")

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

print("📊 Loading datasets...")

if not TEST_TX_PATH.exists():
    print(f"❌ test_transaction.csv not found at {TEST_TX_PATH}")
    sys.exit(1)

print(f"  Loading test_transaction.csv ({TEST_TX_PATH.stat().st_size / (1024**2):.1f} MB)...")
df_test_tx = pd.read_csv(TEST_TX_PATH)
print(f"    ✅ Loaded: {df_test_tx.shape[0]:,} transactions × {df_test_tx.shape[1]} columns")

if TEST_ID_PATH.exists():
    print(f"  Loading test_identity.csv ({TEST_ID_PATH.stat().st_size / (1024**2):.1f} MB)...")
    df_test_id = pd.read_csv(TEST_ID_PATH)
    print(f"    ✅ Loaded: {df_test_id.shape[0]:,} identities × {df_test_id.shape[1]} columns")
    
    # Merge identity with transaction data
    print(f"  Merging identity and transaction data...")
    df_test = df_test_tx.merge(df_test_id, on="TransactionID", how="left")
    print(f"    ✅ Merged: {df_test.shape[0]:,} rows × {df_test.shape[1]} columns\n")
else:
    df_test = df_test_tx.copy()
    print(f"    ⚠️  test_identity.csv not found - using transaction data only\n")

# ═══════════════════════════════════════════════════════════════════════════
# 2. LOAD PREPROCESSING ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════

print("📁 Loading preprocessing artifacts...")

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"  ✅ Scaler loaded")
except Exception as e:
    logger.warning(f"  ⚠️  Scaler not found: {e}")
    scaler = None

try:
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    print(f"  ✅ Label encoders loaded ({len(label_encoders)} categorical features)")
except Exception as e:
    logger.warning(f"  ⚠️  Label encoders not found: {e}")
    label_encoders = None

try:
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = joblib.load(f)
    print(f"  ✅ Feature names loaded ({len(feature_names)} features expected)\n")
except Exception as e:
    logger.warning(f"  ⚠️  Feature names not found: {e}")
    feature_names = None

# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

print("🔧 Applying feature engineering...")

# Make a copy for preprocessing
df_processed = df_test.copy()

# Apply feature engineering
df_processed = create_time_features(df_processed)
df_processed = create_amount_features(df_processed)

print(f"  ✅ Time and amount features created")

# Handle missing values
n_cols_before = df_processed.shape[1]
df_processed = handle_missing_values(df_processed)
n_cols_after = df_processed.shape[1]
print(f"  ✅ Missing values handled ({n_cols_after - n_cols_before:+d} columns)")

# Encode categorical features if encoders loaded
if label_encoders:
    print(f"  Encoding categorical features using saved encoders...")
    for col, encoder in label_encoders.items():
        if col in df_processed.columns:
            try:
                df_processed[col] = df_processed[col].astype(str)
                df_processed[col] = encoder.transform(df_processed[col])
            except Exception as e:
                print(f"    ⚠️  Could not encode {col}: {e}")
    print(f"  ✅ Categorical features encoded ({len(label_encoders)} encoders applied)")
else:
    # Use encode_categoricals if no saved encoders
    print(f"  Encoding categorical features...")
    df_processed, new_encoders = encode_categoricals(df_processed)
    label_encoders = new_encoders
    print(f"  ✅ Categorical features encoded ({len(new_encoders)} features)")

# Select feature columns (exclude ID and target)
exclude_cols = ['TransactionID', 'isFraud', 'id']
feature_cols = [c for c in df_processed.columns if c not in exclude_cols and c.startswith(('TransactionDT', 'TransactionAmt', 'V', 'id', 'card', 'addr', 'dist', 'DeviceType', 'DeviceInfo'))]

# Get features tensor
if feature_names:
    # Use exact feature names from training
    feature_cols = [f for f in feature_names if f in df_processed.columns]
    print(f"  Using {len(feature_cols)} features from training set")
else:
    print(f"  Using {len(feature_cols)} automatically detected features")

X_test = df_processed[feature_cols].fillna(0).values.astype(np.float32)
print(f"  ✅ Features prepared: {X_test.shape[0]:,} samples × {X_test.shape[1]} features\n")

# Scale features if scaler available
if scaler:
    print("  Scaling features...")
    try:
        X_test_scaled = scaler.transform(X_test)
        X_test = X_test_scaled
        print(f"  ✅ Features scaled")
    except Exception as e:
        print(f"  ⚠️  Scaling failed: {e} - using unscaled features")

# ═══════════════════════════════════════════════════════════════════════════
# 4. LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════

print("\n🤖 Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}\n")

results = {}

# Load Decision Tree
print("  Loading Decision Tree...")
try:
    dt_model = joblib.load(DT_MODEL_PATH)
    print(f"    ✅ Decision Tree loaded")
    results["Decision Tree"] = {"model": dt_model, "type": "sklearn"}
except Exception as e:
    print(f"    ❌ Error: {e}")
    results["Decision Tree"] = {"model": None, "type": "sklearn"}

# Load XGBoost
print("  Loading XGBoost...")
try:
    xgb_model = joblib.load(XGB_MODEL_PATH)
    print(f"    ✅ XGBoost loaded")
    results["XGBoost"] = {"model": xgb_model, "type": "sklearn"}
except Exception as e:
    print(f"    ❌ Cannot load XGBoost: {type(e).__name__}")
    print(f"    ⚠️  Skipping XGBoost (known segfault with certain configurations)")
    results["XGBoost"] = {"model": None, "type": "sklearn"}

# Load HGNN_ATT_TD
print("  Loading HGNN_ATT_TD...")
try:
    checkpoint_hgnn = torch.load(HGNN_ATT_TD_PATH, map_location=device, weights_only=False)
    print(f"    ✅ Checkpoint loaded (input_dim={checkpoint_hgnn.get('input_dim')})")
    print(f"    ⚠️  Model requires {checkpoint_hgnn.get('input_dim')} features, have {X_test.shape[1]}")
    results["HGNN_ATT_TD"] = {"checkpoint": checkpoint_hgnn, "type": "hgnn"}
except Exception as e:
    print(f"    ❌ Error: {e}")
    results["HGNN_ATT_TD"] = {"checkpoint": None, "type": "hgnn"}

# ═══════════════════════════════════════════════════════════════════════════
# 5. INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("🎯 Running Inference")
print(f"{'='*80}\n")

predictions = {}
n_samples = X_test.shape[0]

# Decision Tree Inference
if results["Decision Tree"]["model"] is not None:
    print(f"Decision Tree inference on {n_samples:,} samples...")
    try:
        dt_probs = results["Decision Tree"]["model"].predict_proba(X_test)[:, 1]
        dt_preds = (dt_probs >= 0.5).astype(int)
        
        predictions["Decision Tree"] = {
            "probabilities": dt_probs,
            "predictions": dt_preds,
            "mean_prob": np.mean(dt_probs),
            "std_prob": np.std(dt_probs),
            "fraud_count": np.sum(dt_preds),
            "fraud_rate": np.sum(dt_preds) / len(dt_preds)
        }
        
        print(f"  ✅ Inference successful!")
        print(f"     Mean fraud probability: {predictions['Decision Tree']['mean_prob']:.4f}")
        print(f"     Fraud rate: {predictions['Decision Tree']['fraud_rate']:.2%}")
        print(f"     Frauds detected: {predictions['Decision Tree']['fraud_count']:,}\n")
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
        predictions["Decision Tree"] = None

# XGBoost Inference
if results["XGBoost"]["model"] is not None:
    print(f"XGBoost inference on {n_samples:,} samples...")
    try:
        xgb_probs = results["XGBoost"]["model"].predict_proba(X_test)[:, 1]
        xgb_preds = (xgb_probs >= 0.5).astype(int)
        
        predictions["XGBoost"] = {
            "probabilities": xgb_probs,
            "predictions": xgb_preds,
            "mean_prob": np.mean(xgb_probs),
            "std_prob": np.std(xgb_probs),
            "fraud_count": np.sum(xgb_preds),
            "fraud_rate": np.sum(xgb_preds) / len(xgb_preds)
        }
        
        print(f"  ✅ Inference successful!")
        print(f"     Mean fraud probability: {predictions['XGBoost']['mean_prob']:.4f}")
        print(f"     Fraud rate: {predictions['XGBoost']['fraud_rate']:.2%}")
        print(f"     Frauds detected: {predictions['XGBoost']['fraud_count']:,}\n")
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
        predictions["XGBoost"] = None

# HGNN_ATT_TD Inference (if features match)
if results["HGNN_ATT_TD"]["checkpoint"] is not None:
    print(f"HGNN_ATT_TD inference...")
    try:
        input_dim_required = results["HGNN_ATT_TD"]["checkpoint"].get("input_dim", 432)
        
        if X_test.shape[1] != input_dim_required:
            print(f"  ❌ Feature dimension mismatch!")
            print(f"     Required: {input_dim_required} features")
            print(f"     Available: {X_test.shape[1]} features")
            print(f"  ⚠️  Cannot run HGNN_ATT_TD inference - need proper feature engineering\n")
            predictions["HGNN_ATT_TD"] = None
        else:
            from src.models import DenseHGNN_ATT_TD
            
            # Reconstruct model
            model = DenseHGNN_ATT_TD(
                n_feat=input_dim_required,
                n_hid=results["HGNN_ATT_TD"]["checkpoint"].get("hidden_dim", 64),
                n_class=1,
                num_relations=results["HGNN_ATT_TD"]["checkpoint"].get("num_relations", 3),
                decay_init=0.1
            )
            model.load_state_dict(results["HGNN_ATT_TD"]["checkpoint"].get("model_state_dict"))
            model.to(device)
            model.eval()
            
            # Run inference (simplified version without dense adjacency matrices)
            # For production, would need properly constructed relation graphs
            print(f"  ⚠️  Simplified inference mode (requires full graph construction)")
            print(f"  Need actual relation matrices from dataset preprocessing\n")
            predictions["HGNN_ATT_TD"] = None
            
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
        predictions["HGNN_ATT_TD"] = None

# ═══════════════════════════════════════════════════════════════════════════
# 6. COMPARISON & RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("📊 Model Comparison Results")
print(f"{'='*80}\n")

# Summary table
print("Model Performance Summary:")
print(f"{'Model':<20} {'Mean Prob':<15} {'Std Dev':<15} {'Fraud Rate':<15} {'Frauds Found':<15}")
print("-" * 80)

successful_models = [m for m, p in predictions.items() if p is not None]

if successful_models:
    for model_name in successful_models:
        p = predictions[model_name]
        print(f"{model_name:<20} {p['mean_prob']:<15.4f} {p['std_prob']:<15.4f} {p['fraud_rate']:<15.2%} {p['fraud_count']:<15,}")
    
    # Find best model
    if len(successful_models) > 1:
        print("\n" + "="*80)
        print("📈 Analysis:")
        
        # Compare fraud detection rates
        fraud_rates = {m: predictions[m]['fraud_rate'] for m in successful_models}
        print(f"\nFraud Detection Rates:")
        for model, rate in sorted(fraud_rates.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {rate:.2%}")
        
        # Compare mean probabilities
        mean_probs = {m: predictions[m]['mean_prob'] for m in successful_models}
        print(f"\nMean Fraud Probabilities:")
        for model, prob in sorted(mean_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {prob:.4f}")
        
        # Find most conservative and most aggressive
        most_conservative = min(fraud_rates, key=fraud_rates.get)
        most_aggressive = max(fraud_rates, key=fraud_rates.get)
        print(f"\n🏆 Most Conservative: {most_conservative} ({fraud_rates[most_conservative]:.2%} fraud rate)")
        print(f"⚠️  Most Aggressive: {most_aggressive} ({fraud_rates[most_aggressive]:.2%} fraud rate)")

else:
    print("❌ No successful model inferences - check features and model setup")

# ═══════════════════════════════════════════════════════════════════════════
# 7. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("💾 Saving Results")
print(f"{'='*80}\n")

# Create output dataframe
output_df = pd.DataFrame({
    'TransactionID': df_test['TransactionID'].values[:len(X_test)]
})

for model_name in successful_models:
    p = predictions[model_name]
    output_df[f'{model_name}_prob'] = p['probabilities']
    output_df[f'{model_name}_pred'] = p['predictions']

# Save predictions
output_path = PROJECT_ROOT / "outputs" / "real_data_predictions.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to {output_path}")
print(f"   Rows: {len(output_df):,}")
print(f"   Columns: {len(output_df.columns)}")

# Save summary
summary_path = PROJECT_ROOT / "outputs" / "real_data_summary.txt"
with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("🧪 Real Data Test Results\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset: IEEE-CIS Fraud Detection\n")
    f.write(f"Test Samples: {n_samples:,}\n")
    f.write(f"Features: {X_test.shape[1]}\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("Model Results:\n")
    f.write("-" * 80 + "\n")
    for model_name in successful_models:
        p = predictions[model_name]
        f.write(f"\n{model_name}:\n")
        f.write(f"  Mean Fraud Probability: {p['mean_prob']:.6f}\n")
        f.write(f"  Std Dev: {p['std_prob']:.6f}\n")
        f.write(f"  Fraud Rate: {p['fraud_rate']:.2%}\n")
        f.write(f"  Frauds Detected: {p['fraud_count']:,} / {n_samples:,}\n")

print(f"✅ Summary saved to {summary_path}")

print(f"\n{'='*80}\n")
