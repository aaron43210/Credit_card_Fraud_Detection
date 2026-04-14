"""
Test script to compare HGNN_Model (hgnn_model.pt) vs HGNN_ATT_TD (hgnn_att_td.pt)
Tests both models on sample_submission.csv data
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import HeteroData

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HGNN_ATT_TD_PATH, MODEL_DIR
from src.utils import get_logger

logger = get_logger("ModelComparison")

# Define model paths
HGNN_MODEL_PATH = os.path.join(MODEL_DIR, "hgnn_model.pt")
HGNN_ATT_TD_ACTUAL_PATH = str(HGNN_ATT_TD_PATH)

print(f"\n{'='*80}")
print(f"🔬 HGNN Model Comparison Test")
print(f"{'='*80}\n")

# ═══════════════════════════════════════════════════════════════════════════
# 1. CHECK MODEL FILES
# ═══════════════════════════════════════════════════════════════════════════

print("📁 Checking model files...")
print(f"  HGNN Model:        {HGNN_MODEL_PATH}")
print(f"  Exists: {os.path.exists(HGNN_MODEL_PATH)}")
print(f"\n  HGNN_ATT_TD Model: {HGNN_ATT_TD_ACTUAL_PATH}")
print(f"  Exists: {os.path.exists(HGNN_ATT_TD_ACTUAL_PATH)}")

if not os.path.exists(HGNN_MODEL_PATH):
    print("\n❌ HGNN Model file not found!")
    sys.exit(1)
if not os.path.exists(HGNN_ATT_TD_ACTUAL_PATH):
    print("\n❌ HGNN_ATT_TD Model file not found!")
    sys.exit(1)

print("\n✅ Both model files found!\n")

# ═══════════════════════════════════════════════════════════════════════════
# 2. LOAD SAMPLE SUBMISSION DATA
# ═══════════════════════════════════════════════════════════════════════════

print("📊 Loading sample_submission.csv...")
submission_path = os.path.join(PROJECT_ROOT, "sample_submission.csv")
if os.path.exists(submission_path):
    df_submission = pd.read_csv(submission_path)
    print(f"  Shape: {df_submission.shape}")
    print(f"  Columns: {list(df_submission.columns)}")
    print(f"  Sample:\n{df_submission.head()}\n")
    n_samples = len(df_submission)
else:
    print(f"  ⚠️ sample_submission.csv not found at {submission_path}")
    print(f"  Using 1000 synthetic samples for testing\n")
    n_samples = 1000

# ═══════════════════════════════════════════════════════════════════════════
# 3. GENERATE SYNTHETIC TEST DATA
# ═══════════════════════════════════════════════════════════════════════════

print(f"🔧 Generating synthetic test data ({n_samples} samples)...")

# Create synthetic features matching expected model input
input_dim = 50  # Number of transaction features

# Synthetic transaction features
X_test = np.random.randn(n_samples, input_dim).astype(np.float32)

# Create heterogeneous graph structure (for HGNN models)
def create_test_graph(n_transactions, input_dim):
    """Create a synthetic heterogeneous graph for testing."""
    data = HeteroData()
    
    # Transaction nodes
    data["transaction"].x = torch.tensor(X_test, dtype=torch.float32)
    
    # Card nodes (simulate card entities)
    n_cards = max(n_transactions // 100, 10)
    data["card"].x = torch.ones((n_cards, 1), dtype=torch.float32)
    card_indices = np.random.randint(0, n_cards, n_transactions)
    tx_indices = np.arange(n_transactions)
    edge_index = torch.tensor([tx_indices, card_indices], dtype=torch.long)
    data["transaction", "used", "card"].edge_index = edge_index
    data["card", "used_by", "transaction"].edge_index = edge_index.flip([0])
    
    # Device nodes (simulate device entities)
    n_devices = max(n_transactions // 200, 5)
    data["device"].x = torch.ones((n_devices, 1), dtype=torch.float32)
    device_indices = np.random.randint(0, n_devices, n_transactions)
    edge_index_dev = torch.tensor([tx_indices, device_indices], dtype=torch.long)
    data["transaction", "on", "device"].edge_index = edge_index_dev
    data["device", "hosts", "transaction"].edge_index = edge_index_dev.flip([0])
    
    # Time decay features
    tx_time = np.arange(n_transactions, dtype=np.float32)
    age = tx_time.max() - tx_time
    age_scale = np.maximum(np.percentile(age, 95), 1.0)
    decay = np.exp(-age / age_scale).astype(np.float32)
    data["transaction"].time_decay = torch.from_numpy(decay).unsqueeze(-1)
    
    return data

test_graph = create_test_graph(n_samples, input_dim)
print(f"  ✅ Graph created with shapes:")
print(f"     - Transaction nodes: {test_graph['transaction'].x.shape}")
print(f"     - Card nodes: {test_graph['card'].x.shape}")
print(f"     - Device nodes: {test_graph['device'].x.shape}\n")

# ═══════════════════════════════════════════════════════════════════════════
# 4. LOAD BOTH MODELS
# ═══════════════════════════════════════════════════════════════════════════

print("🤖 Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Using device: {device}\n")

# Load HGNN Model
print("  Loading HGNN Model (hgnn_model.pt)...")
try:
    checkpoint_hgnn = torch.load(HGNN_MODEL_PATH, map_location=device, weights_only=False)
    print(f"    ✅ Loaded successfully")
    print(f"    Keys: {checkpoint_hgnn.keys()}")
except Exception as e:
    print(f"    ❌ Error loading: {e}")
    checkpoint_hgnn = None

# Load HGNN_ATT_TD Model
print("\n  Loading HGNN_ATT_TD Model (hgnn_att_td.pt)...")
try:
    checkpoint_att_td = torch.load(HGNN_ATT_TD_ACTUAL_PATH, map_location=device, weights_only=False)
    print(f"    ✅ Loaded successfully")
    print(f"    Keys: {checkpoint_att_td.keys()}")
except Exception as e:
    print(f"    ❌ Error loading: {e}")
    checkpoint_att_td = None

# ═══════════════════════════════════════════════════════════════════════════
# 5. RECONSTRUCT AND TEST MODELS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("🧪 Model Inference Testing")
print(f"{'='*80}\n")

results = {}

# Test HGNN Model
if checkpoint_hgnn:
    print("Testing HGNN Model (hgnn_model.pt)...")
    try:
        # Get model metadata
        input_dim_hgnn = checkpoint_hgnn.get("input_dim", 217)
        hidden_dims = checkpoint_hgnn.get("hidden_dims", [64, 32])
        dropout_rates = checkpoint_hgnn.get("dropout_rates", [0.3, 0.2])
        
        print(f"  Model config: input_dim={input_dim_hgnn}")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Dropout rates: {dropout_rates}")
        print(f"  ⚠️  Model requires {input_dim_hgnn} input features, but synthetic data has {input_dim}")
        
        print(f"\n  ❌ Feature dimension mismatch:")
        print(f"     - Model trained on: {input_dim_hgnn} features")
        print(f"     - Synthetic data: {input_dim} features")
        print(f"     - Need actual preprocessed training data to test this model\n")
        
        results["HGNN Model"] = None
        
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
        results["HGNN Model"] = None

# Test HGNN_ATT_TD Model
if checkpoint_att_td:
    print("Testing HGNN_ATT_TD Model (hgnn_att_td.pt)...")
    try:
        from src.models import DenseHGNN_ATT_TD
        
        # Get model metadata from checkpoint
        input_dim_att = checkpoint_att_td.get("input_dim", 432)
        hidden_dim = checkpoint_att_td.get("hidden_dim", 64)
        num_relations = checkpoint_att_td.get("num_relations", 3)
        
        print(f"  Model config: n_feat={input_dim_att}")
        print(f"  n_hid: {hidden_dim}")
        print(f"  num_relations: {num_relations}")
        print(f"  ⚠️  Model requires {input_dim_att} input features, but synthetic data has {input_dim}")
        
        # Note: Cannot run inference with synthetic data due to feature dimension mismatch
        # The model expects {input_dim_att} features from the actual IEEE-CIS dataset
        print(f"\n  ❌ Feature dimension mismatch:")
        print(f"     - Model trained on: {input_dim_att} features")
        print(f"     - Synthetic data: {input_dim} features")
        print(f"     - Need actual preprocessed training data to test this model\n")
        
        results["HGNN_ATT_TD"] = None
        
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
        results["HGNN_ATT_TD"] = None

# ═══════════════════════════════════════════════════════════════════════════
# 6. COMPARISON RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("📊 Model Comparison Results")
print(f"{'='*80}\n")

if all(v is not None for v in results.values()):
    print("Model Comparison Metrics:\n")
    print(f"{'Metric':<30} {'HGNN Model':<20} {'HGNN_ATT_TD':<20}")
    print("-" * 70)
    
    for metric in ["mean_prob", "std_prob", "fraud_rate"]:
        val1 = results["HGNN Model"][metric]
        val2 = results["HGNN_ATT_TD"][metric]
        
        if metric == "fraud_rate":
            print(f"{metric:<30} {val1:>18.2%}  {val2:>18.2%}")
        else:
            print(f"{metric:<30} {val1:>20.4f}  {val2:>20.4f}")
    
    print("\n" + "="*70)
    
    # Determine which model is "better"
    mean_diff = abs(results["HGNN Model"]["mean_prob"] - results["HGNN_ATT_TD"]["mean_prob"])
    if mean_diff < 0.01:
        print("✅ Both models produce similar predictions")
    elif results["HGNN Model"]["mean_prob"] > results["HGNN_ATT_TD"]["mean_prob"]:
        print("✅ HGNN Model produces higher fraud probabilities")
    else:
        print("✅ HGNN_ATT_TD produces  higher fraud probabilities")
    
    
else:
    print("⚠️ Could not compare models - feature dimension mismatch detected")
    print("\n📊 Model Feature Requirements:")
    if checkpoint_hgnn:
        print(f"   ✅ HGNN Model loaded: {checkpoint_hgnn.get('input_dim')} features required")
    if checkpoint_att_td:
        print(f"   ✅ HGNN_ATT_TD loaded: {checkpoint_att_td.get('input_dim')} features required")
    
    print("\n💡 To properly test these models, you need:")
    print("   1. Load actual IEEE-CIS transaction and identity data")
    print("   2. Apply the full feature engineering pipeline from src/feature_engineering.py")
    print("   3. Run inference with properly engineered features")
    print("   4. The models were trained on preprocessed features from the training pipeline")
    print("\n📝 What we learned:")
    if checkpoint_hgnn:
        print(f"   - HGNN Model expects {checkpoint_hgnn.get('input_dim')} features (from training data)")
    if checkpoint_att_td:
        print(f"   - HGNN_ATT_TD expects {checkpoint_att_td.get('input_dim')} features (from training data)")

print(f"\n{'='*80}\n")
