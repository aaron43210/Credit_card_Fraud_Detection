"""
training.py — Stage 6: Model Training Pipelines
===================================================
Implements training loops for all three models with SMOTE
oversampling and cost-sensitive learning strategies.
"""

import os
import copy
import sys
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import joblib
import torch
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from torch_geometric.loader import NeighborLoader
from xgboost.core import XGBoostError

from src.config import (
    SMOTE_PARAMS, NN_PARAMS, N_FOLDS, RANDOM_STATE,
    DT_MODEL_PATH, XGB_MODEL_PATH, NN_MODEL_PATH, MODEL_DIR,
)
from src.models import build_decision_tree, build_xgboost, build_neural_network, FraudNet, FocalLoss
from src.utils import get_logger, Timer, format_number

logger = get_logger("Training")

# Set to 1 to allow automatic MLP fallback on very large graphs.
# Default is full HGNN training whenever hetero_data is provided.
ALLOW_HGNN_FALLBACK = os.environ.get("ALLOW_HGNN_FALLBACK", "0") == "1"
HGNN_MAX_TRANSACTIONS = 200_000
USE_NEIGHBOR_SAMPLED_HGNN = os.environ.get("USE_NEIGHBOR_SAMPLED_HGNN", "0") == "1"


def _force_classic_dmatrix(model: Any) -> None:
    """Force sklearn XGBoost wrapper to use classic DMatrix (not QuantileDMatrix).

    This avoids runtime issues in some CUDA environments where QuantileDMatrix
    fails with array_interface.cu errors.
    """

    model._create_dmatrix = _create_classic_dmatrix


def _create_classic_dmatrix(ref=None, **kwargs):
    """Module-level DMatrix factory so patched XGBoost models remain picklable."""
    return xgb.DMatrix(**kwargs)


def _parse_neighbor_hops() -> List[int]:
    raw = os.environ.get("SAMPLED_HGNN_NEIGHBORS", "15,10")
    hops: List[int] = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            hops.append(max(1, int(value)))
        except ValueError:
            continue
    return hops or [15, 10]


def _build_neighbor_loader(
    hetero_data: Any,
    seed_nodes: torch.Tensor,
    shuffle: bool,
) -> NeighborLoader:
    num_neighbors = {edge_type: _parse_neighbor_hops() for edge_type in hetero_data.edge_types}
    batch_size = int(os.environ.get("SAMPLED_HGNN_BATCH_SIZE", "1024"))
    num_workers = int(os.environ.get("SAMPLED_HGNN_NUM_WORKERS", "0"))

    return NeighborLoader(
        hetero_data,
        input_nodes=("transaction", seed_nodes),
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
    )


def train_neighbor_sampled_hgnn(
    X_train_df: Union[pd.DataFrame, np.ndarray],
    y_train_df: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    hetero_data: Any,
    use_smote: bool = False,
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """Train FraudHGNN with neighbor-sampled mini-batches."""
    logger.info("=" * 50)
    logger.info("  🧠 Training Neural Network (Neighbor-Sampled HGNN)")
    logger.info("=" * 50)

    if use_smote:
        logger.warning("SMOTE is skipped for neighbor-sampled HGNN because it breaks graph structure.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"  Device: {device}")

    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        free_gb = free_b / (1024 ** 3)
        total_gb = total_b / (1024 ** 3)
        min_free_gb = float(os.environ.get("HGNN_MIN_FREE_GB", "20"))
        logger.info(f"  CUDA free memory: {free_gb:.2f} / {total_gb:.2f} GB")
        if free_gb < min_free_gb:
            raise RuntimeError(
                f"Not enough free GPU memory for neighbor-sampled HGNN: "
                f"{free_gb:.2f} GB free, need at least {min_free_gb:.1f} GB."
            )

    hetero_data = hetero_data.cpu()
    input_dim = int(hetero_data["transaction"].x.shape[1])

    model, criterion, optimizer = build_neural_network(input_dim, metadata=hetero_data.metadata())
    model = model.to(device)

    train_nodes = torch.where(hetero_data["transaction"].train_mask)[0]
    val_nodes = torch.where(hetero_data["transaction"].val_mask)[0]

    train_loader = _build_neighbor_loader(hetero_data, train_nodes, shuffle=True)
    val_loader = _build_neighbor_loader(hetero_data, val_nodes, shuffle=False)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=NN_PARAMS["learning_rate"] * 5,
        epochs=int(NN_PARAMS["max_epochs"]),
        steps_per_epoch=max(1, len(train_loader)),
    )

    best_val_auc = 0.0
    patience_counter = 0
    train_losses: List[float] = []
    val_aucs: List[float] = []

    with Timer("Neural Network training (Neighbor-Sampled HGNN)", logger):
        for epoch in range(int(NN_PARAMS["max_epochs"])):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)

                tx_decay = getattr(batch['transaction'], 'time_decay', None)
                logits = model(batch.x_dict, batch.edge_index_dict, tx_time_decay=tx_decay)
                seed_count = batch["transaction"].batch_size
                train_logits = logits[:seed_count]
                targets = batch["transaction"].y[:seed_count].float()

                loss = criterion(train_logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            train_losses.append(avg_loss)

            model.eval()
            val_probs: List[np.ndarray] = []
            val_targets: List[np.ndarray] = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    tx_decay = getattr(batch['transaction'], 'time_decay', None)
                    logits = model(batch.x_dict, batch.edge_index_dict, tx_time_decay=tx_decay)
                    seed_count = batch["transaction"].batch_size
                    probs = torch.sigmoid(logits[:seed_count]).squeeze().cpu().numpy()
                    targets = batch["transaction"].y[:seed_count].cpu().numpy()
                    val_probs.append(np.atleast_1d(probs))
                    val_targets.append(np.atleast_1d(targets))

            val_proba = np.concatenate(val_probs)
            y_val_np = np.concatenate(val_targets)
            val_auc = float(roc_auc_score(y_val_np, val_proba))
            val_aucs.append(val_auc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch+1:3d}/{int(NN_PARAMS['max_epochs'])} | "
                    f"Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}"
                )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                os.makedirs(MODEL_DIR, exist_ok=True)
                torch.save(
                    {
                        "model_type": "hgnn_sampled",
                        "model_state_dict": copy.deepcopy(model.state_dict()),
                        "input_dim": input_dim,
                        "hidden_dims": NN_PARAMS["hidden_dims"],
                        "dropout_rates": NN_PARAMS["dropout_rates"],
                        "best_val_auc": best_val_auc,
                        "epoch": epoch + 1,
                    },
                    NN_MODEL_PATH,
                )
            else:
                patience_counter += 1
                if patience_counter >= int(NN_PARAMS["patience"]):
                    logger.info(f"  ⏹ Early stopping at epoch {epoch+1}")
                    break

    checkpoint = torch.load(NN_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    val_probs: List[np.ndarray] = []
    val_targets: List[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            tx_decay = getattr(batch['transaction'], 'time_decay', None)
            logits = model(batch.x_dict, batch.edge_index_dict, tx_time_decay=tx_decay)
            seed_count = batch["transaction"].batch_size
            probs = torch.sigmoid(logits[:seed_count]).squeeze().cpu().numpy()
            targets = batch["transaction"].y[:seed_count].cpu().numpy()
            val_probs.append(np.atleast_1d(probs))
            val_targets.append(np.atleast_1d(targets))

    val_proba = np.concatenate(val_probs)
    y_val_np = np.concatenate(val_targets)
    val_preds = (val_proba >= 0.5).astype(int)
    val_f1 = float(f1_score(y_val_np, val_preds))
    val_auprc = float(average_precision_score(y_val_np, val_proba))

    logger.info(f"  Best Val AUC:  {best_val_auc:.4f}")
    logger.info(f"  Val F1-Score:  {val_f1:.4f}")
    logger.info(f"  Val AUPRC:     {val_auprc:.4f}")
    logger.info(f"  Model saved → {NN_MODEL_PATH}")

    return model, {"train_losses": train_losses, "val_aucs": val_aucs}


def apply_smote(
    X_train: Union[pd.DataFrame, np.ndarray], 
    y_train: Union[pd.Series, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to
    balance the training set.

    SMOTE generates synthetic fraud samples by interpolating between
    existing minority samples and their k-nearest neighbors. This is
    applied ONLY to the training set — never to validation/test.
    """
    n_before_pos = int(y_train.sum())
    n_before_neg = len(y_train) - n_before_pos

    smote = SMOTE(**SMOTE_PARAMS)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    n_after_pos = int(y_resampled.sum())
    n_after_neg = len(y_resampled) - n_after_pos

    logger.info(f"  SMOTE Resampling:")
    logger.info(f"    Before: {format_number(n_before_neg)} legit, {format_number(n_before_pos)} fraud (ratio 1:{n_before_neg/max(n_before_pos,1):.1f})")
    logger.info(f"    After:  {format_number(n_after_neg)} legit, {format_number(n_after_pos)} fraud (ratio 1:{n_after_neg/max(n_after_pos,1):.1f})")
    logger.info(f"    Synthetic samples created: {format_number(n_after_pos - n_before_pos)}")

    return X_resampled, y_resampled


def train_decision_tree(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    use_smote: bool = True
) -> Any:
    """
    Train a Decision Tree classifier.

    Uses cost-sensitive learning (class_weight='balanced') AND
    optionally SMOTE for double protection against class imbalance.
    """
    logger.info("=" * 50)
    logger.info("  🌳 Training Decision Tree")
    logger.info("=" * 50)

    if use_smote:
        X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    else:
        X_train_sm, y_train_sm = X_train, y_train

    with Timer("Decision Tree training", logger):
        model = build_decision_tree()
        model.fit(X_train_sm, y_train_sm)

    # Validation metrics
    val_score = model.score(X_val, y_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    val_preds = model.predict(X_val)
    val_f1 = f1_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_proba)

    logger.info(f"  Val Accuracy: {val_score:.4f}")
    logger.info(f"  Val F1-Score: {val_f1:.4f}")
    logger.info(f"  Val ROC-AUC:  {val_auc:.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, DT_MODEL_PATH)
    logger.info(f"  Model saved → {DT_MODEL_PATH}")

    return model


def train_xgboost(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    use_smote: bool = True
) -> Any:
    """
    Train an XGBoost classifier with early stopping.

    Uses scale_pos_weight for cost-sensitive learning AND
    optionally SMOTE. Early stopping prevents overfitting by
    monitoring AUPRC on the validation set.
    """
    logger.info("=" * 50)
    logger.info("  🚀 Training XGBoost (CPU Mode)")
    logger.info("=" * 50)
    logger.info("  ℹ️ Using CPU backend for stability (GPU CUDA backend can be unstable)")

    # macOS + OpenMP + XGBoost can segfault under high thread counts.
    # Keep local training conservative and deterministic unless user overrides.
    if sys.platform == "darwin":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("XGB_N_JOBS", "1")

    def _to_numpy_2d(x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        arr = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return np.ascontiguousarray(arr)

    def _to_numpy_1d(y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
        return np.ascontiguousarray(arr.astype(np.float32))

    # Calculate class imbalance ratio for scale_pos_weight
    y_train_arr = _to_numpy_1d(y_train)
    y_val_arr = _to_numpy_1d(y_val)
    X_val_arr = _to_numpy_2d(X_val)

    n_pos = int(y_train_arr.sum())
    n_neg = len(y_train_arr) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    if use_smote:
        X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    else:
        X_train_sm, y_train_sm = X_train, y_train

    X_train_sm_arr = _to_numpy_2d(X_train_sm)
    y_train_sm_arr = _to_numpy_1d(y_train_sm)

    with Timer("XGBoost training", logger):
        model = build_xgboost(scale_pos_weight=scale_pos_weight)
        # Use CPU for XGBoost to avoid CUDA backend instability
        # XGBoost is still very fast on CPU (hist tree method with CPU predictor)
        xgb_n_jobs = int(os.environ.get("XGB_N_JOBS", "1" if sys.platform == "darwin" else "-1"))
        model.set_params(tree_method="hist", n_jobs=xgb_n_jobs)
        logger.info(f"  XGBoost threads (n_jobs): {xgb_n_jobs}")
        # Remove GPU parameters to force CPU mode
        model.gpu_id = None
        # Force classic DMatrix (not QuantileDMatrix) to avoid CUDA initialization
        _force_classic_dmatrix(model)

        try:
            model.fit(
                X_train_sm_arr, y_train_sm_arr,
                eval_set=[(X_val_arr, y_val_arr)],
                verbose=False,
            )
        except XGBoostError as exc:
            raise RuntimeError(
                f"XGBoost training failed: {exc}"
            ) from exc

    # Validation metrics
    val_proba = model.predict_proba(X_val_arr)[:, 1]
    val_preds = model.predict(X_val_arr)
    val_f1 = f1_score(y_val_arr, val_preds)
    val_auc = roc_auc_score(y_val_arr, val_proba)
    val_auprc = average_precision_score(y_val_arr, val_proba)

    logger.info(f"  Val F1-Score: {val_f1:.4f}")
    logger.info(f"  Val ROC-AUC:  {val_auc:.4f}")
    logger.info(f"  Val AUPRC:    {val_auprc:.4f}")
    logger.info(f"  Best iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, XGB_MODEL_PATH)
    logger.info(f"  Model saved → {XGB_MODEL_PATH}")

    return model


def train_neural_network(
    X_train_df: Union[pd.DataFrame, np.ndarray],
    y_train_df: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    hetero_data: Optional[Any] = None,
    use_smote: bool = False
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train the FraudHGNN neural network with Focal Loss and PyTorch Geometric.

    Training loop includes:
    - Focal Loss for class imbalance handling
    - OneCycleLR learning rate scheduler
    - Early stopping with patience=10
    - Best model checkpointing

    The neural network is trained on the full heterogenous graph locally.
    """
    logger.info("=" * 50)
    logger.info("  🧠 Training Neural Network (FraudNet)")
    logger.info("=" * 50)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"  Device: {device}")

    use_hgnn = hetero_data is not None
    if use_hgnn and USE_NEIGHBOR_SAMPLED_HGNN:
        return train_neighbor_sampled_hgnn(
            X_train_df,
            y_train_df,
            X_val,
            y_val,
            hetero_data=hetero_data,
            use_smote=use_smote,
        )

    if use_hgnn and ALLOW_HGNN_FALLBACK:
        n_transactions = int(hetero_data['transaction'].x.shape[0])
        if n_transactions > HGNN_MAX_TRANSACTIONS:
            use_hgnn = False
            logger.warning(
                f"  Hetero graph has {format_number(n_transactions)} transactions; "
                f"switching to tabular MLP to avoid long full-graph training."
            )

        # Avoid immediate OOM when another job already occupies GPU memory.
        if use_hgnn and device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            min_free_gb = float(os.environ.get("HGNN_MIN_FREE_GB", "10"))
            logger.info(f"  CUDA free memory: {free_gb:.2f} / {total_gb:.2f} GB")
            if free_gb < min_free_gb:
                use_hgnn = False
                logger.warning(
                    f"  Free CUDA memory ({free_gb:.2f} GB) is below HGNN_MIN_FREE_GB={min_free_gb:.1f}. "
                    "Switching to tabular MLP fallback to prevent OOM."
                )

    # -----------------------------------------------------------------
    # Path A: Full-graph HGNN (for smaller graphs)
    # -----------------------------------------------------------------
    if use_hgnn:
        if use_smote:
            logger.warning("SMOTE is generally incompatible with pure GNN topology. Relying on FocalLoss instead.")

        hetero_data = hetero_data.to(device)
        input_dim: int = hetero_data['transaction'].x.shape[1]

        model: torch.nn.Module
        criterion: torch.nn.Module
        optimizer: torch.optim.Optimizer
        model, criterion, optimizer = build_neural_network(input_dim, metadata=hetero_data.metadata())
        model = model.to(device)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=NN_PARAMS["learning_rate"] * 10,
            epochs=NN_PARAMS["max_epochs"],
            steps_per_epoch=1,
        )

        best_val_auc: float = 0.0
        patience_counter: int = 0
        train_losses: List[float] = []
        val_aucs: List[float] = []

        y_train_t: torch.Tensor = hetero_data['transaction'].y[hetero_data['transaction'].train_mask]
        y_val_t: torch.Tensor = hetero_data['transaction'].y[hetero_data['transaction'].val_mask]
        tx_decay_full = getattr(hetero_data['transaction'], 'time_decay', None)

        with Timer("Neural Network training (HGNN Full Batch)", logger):
            for epoch in range(NN_PARAMS["max_epochs"]):
                model.train()

                optimizer.zero_grad()
                logits: torch.Tensor = model(
                    hetero_data.x_dict,
                    hetero_data.edge_index_dict,
                    tx_time_decay=tx_decay_full,
                )
                train_logits: torch.Tensor = logits[hetero_data['transaction'].train_mask]

                loss: torch.Tensor = criterion(train_logits, y_train_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                avg_loss = float(loss.item())
                train_losses.append(avg_loss)

                model.eval()
                with torch.no_grad():
                    val_logits_full: torch.Tensor = model(
                        hetero_data.x_dict,
                        hetero_data.edge_index_dict,
                        tx_time_decay=tx_decay_full,
                    )
                    val_logits = val_logits_full[hetero_data['transaction'].val_mask]
                    val_proba: np.ndarray = torch.sigmoid(val_logits).squeeze().cpu().numpy()

                val_auc = float(roc_auc_score(y_val_t.cpu().numpy(), val_proba))
                val_aucs.append(val_auc)

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(
                        f"  Epoch {epoch+1:3d}/{NN_PARAMS['max_epochs']} | "
                        f"Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}"
                    )

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    torch.save({
                        "model_type": "hgnn",
                        "model_state_dict": copy.deepcopy(model.state_dict()),
                        "input_dim": input_dim,
                        "hidden_dims": NN_PARAMS["hidden_dims"],
                        "dropout_rates": NN_PARAMS["dropout_rates"],
                        "best_val_auc": best_val_auc,
                        "epoch": epoch + 1,
                    }, NN_MODEL_PATH)
                else:
                    patience_counter += 1
                    if patience_counter >= NN_PARAMS["patience"]:
                        logger.info(f"  ⏹ Early stopping at epoch {epoch+1} (patience={NN_PARAMS['patience']})")
                        break

        checkpoint = torch.load(NN_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        with torch.no_grad():
            final_logits = model(
                hetero_data.x_dict,
                hetero_data.edge_index_dict,
                tx_time_decay=tx_decay_full,
            )
            val_logits = final_logits[hetero_data['transaction'].val_mask]
            val_proba = torch.sigmoid(val_logits).squeeze().cpu().numpy()

        val_preds: np.ndarray = (val_proba >= 0.5).astype(int)
        val_f1 = float(f1_score(y_val_t.cpu().numpy(), val_preds))
        val_auprc = float(average_precision_score(y_val_t.cpu().numpy(), val_proba))

        logger.info(f"  Best Val AUC:  {best_val_auc:.4f}")
        logger.info(f"  Val F1-Score:  {val_f1:.4f}")
        logger.info(f"  Val AUPRC:     {val_auprc:.4f}")
        logger.info(f"  Model saved → {NN_MODEL_PATH}")

        return model, {"train_losses": train_losses, "val_aucs": val_aucs}

    # -----------------------------------------------------------------
    # Path B: Tabular MLP fallback (fast on large datasets)
    # -----------------------------------------------------------------
    logger.info("  Using tabular FraudNet MLP training path")

    X_train_arr = X_train_df.values if isinstance(X_train_df, pd.DataFrame) else np.asarray(X_train_df)
    y_train_arr = y_train_df.values if isinstance(y_train_df, pd.Series) else np.asarray(y_train_df)
    X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else np.asarray(X_val)
    y_val_arr = y_val.values if isinstance(y_val, pd.Series) else np.asarray(y_val)

    if use_smote:
        X_train_arr, y_train_arr = apply_smote(X_train_arr, y_train_arr)

    X_train_t = torch.tensor(X_train_arr, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_arr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_arr, dtype=torch.float32, device=device)
    y_val_np = np.asarray(y_val_arr).astype(int)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_ds,
        batch_size=NN_PARAMS["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    input_dim = int(X_train_arr.shape[1])
    model = FraudNet(
        input_dim=input_dim,
        hidden_dims=NN_PARAMS["hidden_dims"],
        dropout_rates=NN_PARAMS["dropout_rates"],
    ).to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=NN_PARAMS["learning_rate"],
        weight_decay=NN_PARAMS["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=NN_PARAMS["learning_rate"] * 5,
        epochs=NN_PARAMS["max_epochs"],
        steps_per_epoch=max(1, len(train_loader)),
    )

    best_val_auc = 0.0
    patience_counter = 0
    train_losses: List[float] = []
    val_aucs: List[float] = []

    with Timer("Neural Network training (Tabular MLP)", logger):
        for epoch in range(NN_PARAMS["max_epochs"]):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            train_losses.append(avg_loss)

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_proba = torch.sigmoid(val_logits).squeeze().cpu().numpy()

            val_auc = float(roc_auc_score(y_val_np, val_proba))
            val_aucs.append(val_auc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch+1:3d}/{NN_PARAMS['max_epochs']} | "
                    f"Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}"
                )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                os.makedirs(MODEL_DIR, exist_ok=True)
                torch.save({
                    "model_type": "mlp",
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "input_dim": input_dim,
                    "hidden_dims": NN_PARAMS["hidden_dims"],
                    "dropout_rates": NN_PARAMS["dropout_rates"],
                    "best_val_auc": best_val_auc,
                    "epoch": epoch + 1,
                }, NN_MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= NN_PARAMS["patience"]:
                    logger.info(f"  ⏹ Early stopping at epoch {epoch+1} (patience={NN_PARAMS['patience']})")
                    break

    checkpoint = torch.load(NN_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_proba = torch.sigmoid(val_logits).squeeze().cpu().numpy()

    val_preds: np.ndarray = (val_proba >= 0.5).astype(int)
    val_f1 = float(f1_score(y_val_np, val_preds))
    val_auprc = float(average_precision_score(y_val_np, val_proba))

    logger.info(f"  Best Val AUC:  {best_val_auc:.4f}")
    logger.info(f"  Val F1-Score:  {val_f1:.4f}")
    logger.info(f"  Val AUPRC:     {val_auprc:.4f}")
    logger.info(f"  Model saved → {NN_MODEL_PATH}")

    return model, {"train_losses": train_losses, "val_aucs": val_aucs}


def cross_validate_models(
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray], 
    n_folds: int = N_FOLDS
) -> Dict[str, List[Dict[str, float]]]:
    """
    Perform stratified K-fold cross-validation for all models.

    SMOTE is applied inside each fold (only to training portion)
    to prevent data leakage.
    """
    logger.info("=" * 60)
    logger.info(f"     {n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    logger.info("=" * 60)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    results: Dict[str, List[Dict[str, float]]] = {"Decision Tree": [], "XGBoost": [], "Neural Network": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n  ── Fold {fold+1}/{n_folds} ──")

        X_fold_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        X_fold_val = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
        y_fold_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
        y_fold_val = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]

        # Apply SMOTE to training fold
        smote = SMOTE(**SMOTE_PARAMS)
        X_sm, y_sm = smote.fit_resample(X_fold_train, y_fold_train)

        # Decision Tree
        dt = build_decision_tree()
        dt.fit(X_sm, y_sm)
        dt_proba = dt.predict_proba(X_fold_val)[:, 1]
        dt_preds = dt.predict(X_fold_val)

        # XGBoost
        n_neg = (y_fold_train == 0).sum()
        n_pos = (y_fold_train == 1).sum()
        xgb = build_xgboost(scale_pos_weight=n_neg / max(n_pos, 1))
        # Use CPU mode for XGBoost (avoids CUDA backend instability)
        xgb.set_params(tree_method="hist")
        xgb.gpu_id = None
        # Force classic DMatrix (not QuantileDMatrix) to avoid CUDA initialization
        _force_classic_dmatrix(xgb)
        xgb.fit(X_sm, y_sm, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
        xgb_proba = xgb.predict_proba(X_fold_val)[:, 1]
        xgb_preds = xgb.predict(X_fold_val)

        for name, preds, proba in [
            ("Decision Tree", dt_preds, dt_proba),
            ("XGBoost", xgb_preds, xgb_proba),
        ]:
            fold_metrics = {
                "fold": float(fold + 1),
                "f1": float(f1_score(y_fold_val, preds)),
                "auc": float(roc_auc_score(y_fold_val, proba)),
                "auprc": float(average_precision_score(y_fold_val, proba)),
            }
            results[name].append(fold_metrics)
            logger.info(f"    {name:20s} | F1: {fold_metrics['f1']:.4f} | AUC: {fold_metrics['auc']:.4f} | AUPRC: {fold_metrics['auprc']:.4f}")

    return results


def train_dense_hgnn_att_td(
    df: pd.DataFrame,
    y: Union[np.ndarray, pd.Series],
    X_features: Union[np.ndarray, pd.DataFrame],
    relations: List[str],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    hidden_dim: int = 64,
    max_nodes: Optional[int] = None,
    batch_size: int = 2000,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
) -> Tuple[Any, Dict[str, List[float]]]:
    """
    Train DenseHGNN_ATT_TD with staged batch processing for DGX compatibility.
    
    Architecture:
    - Multiple relation-specific views (card1, addr1, P_emaildomain)
    - Time decay applied to temporal transaction distances
    - Multi-view attention fusion
    - Gradual accumulation for memory efficiency
    
    Parameters
    ----------
    df : pd.DataFrame
        Full transaction dataframe (must contain TransactionDT and relation columns)
    y : array-like
        Binary labels
    X_features : array-like
        Node feature matrix (num_nodes, num_features)
    relations : List[str]
        Relation column names for graph construction
    train_indices : np.ndarray
        Training set node indices
    val_indices : np.ndarray
        Validation set node indices
    hidden_dim : int
        Hidden embedding dimension (default: 64)
    max_nodes : int, optional
        Max nodes to use (default: None = all)
    batch_size : int
        Nodes per gradient accumulation batch (default: 2000)
    num_epochs : int
        Training epochs (default: 20)
    learning_rate : float
        Learning rate (default: 1e-3)
        
    Returns
    -------
    tuple
        (model, history_dict)
    """
    from src.models import DenseHGNN_ATT_TD, build_dense_hgnn_att_td
    from src.hgnn_utils import (
        build_dense_graphs,
        focal_loss_weight,
        prepare_training_data,
        create_node_batches,
        log_graph_info,
        estimate_graph_memory,
    )
    
    logger.info("=" * 60)
    logger.info("  🧠 Training DenseHGNN-ATT-TD (Heterogeneous Graph NN)")
    logger.info("=" * 60)
    
    # Device selection: dense N x N graph training is most stable on CUDA or CPU.
    # MPS can be fragile for very large dense tensors, so default to CPU when CUDA is unavailable.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.warning("  MPS detected, but using CPU for DenseHGNN stability.")
        device = torch.device("cpu")
    
    logger.info(f"  📍 Device: {device}")
    
    y_arr = np.asarray(y)

    # Safety defaults by backend if caller doesn't provide max_nodes.
    # Dense adjacency scales as O(N^2), so conservative caps avoid kernel crashes.
    if max_nodes is None:
        if device.type == "cuda":
            max_nodes = int(os.environ.get("DENSE_HGNN_MAX_NODES_CUDA", "12000"))
        else:
            max_nodes = int(os.environ.get("DENSE_HGNN_MAX_NODES_CPU", "3000"))

    # Enforce a hard cap and keep X/y/index arrays aligned via stratified sub-sampling.
    if len(df) > max_nodes:
        logger.info(f"  📉 Sampling down to {max_nodes} from {len(df)} nodes (memory safety)")
        from sklearn.model_selection import train_test_split

        sampled_idx, _ = train_test_split(
            np.arange(len(df)),
            train_size=max_nodes,
            random_state=42,
            stratify=y_arr,
        )
        sampled_idx = np.sort(sampled_idx)

        df = df.iloc[sampled_idx].reset_index(drop=True)
        if isinstance(X_features, pd.DataFrame):
            X_features = X_features.iloc[sampled_idx].reset_index(drop=True)
        else:
            X_features = np.asarray(X_features)[sampled_idx]
        y_arr = y_arr[sampled_idx]

        # Recreate train/val split on sampled subset to avoid index mismatch.
        val_ratio = len(val_indices) / max(1, (len(train_indices) + len(val_indices)))
        train_indices, val_indices = train_test_split(
            np.arange(len(y_arr)),
            test_size=val_ratio,
            random_state=RANDOM_STATE,
            stratify=y_arr,
        )

    est_mem_gb = estimate_graph_memory(len(df), len(relations))
    logger.info(f"  📐 Estimated dense graph memory: ~{est_mem_gb:.2f} GB")
    
    num_nodes = len(df)
    
    # Build dense heterogeneous graphs
    logger.info(f"\n  🔗 Building {len(relations)} relation views...")
    with Timer("Graph construction", logger):
        adj_list, dts_list = build_dense_graphs(df, relations, max_nodes=None, device="cpu")
    
    # Prepare features and labels
    x_tensor = torch.tensor(
        X_features if isinstance(X_features, np.ndarray) else X_features.values,
        dtype=torch.float32,
        device="cpu",
    )
    
    y_tensor, train_mask, val_mask = prepare_training_data(y_arr, train_indices, val_indices)
    
    log_graph_info(x_tensor.shape, adj_list, dts_list)
    
    # Initialize model and optimizer
    input_dim = x_tensor.shape[1]
    model, optimizer = build_dense_hgnn_att_td(input_dim, hidden_dim, len(relations))
    model = model.to(device)
    
    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)
    adj_list = [adj.to(device) for adj in adj_list]
    dts_list = [dt.to(device) for dt in dts_list]
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-5
    )
    
    train_losses = []
    val_aucs = []
    best_val_auc = 0.0
    patience = 5
    patience_counter = 0
    
    logger.info(f"\n  ⏳ Training for {num_epochs} epochs (batch_size={batch_size})...")
    
    with Timer(f"DenseHGNN-ATT-TD training ({num_epochs} epochs)", logger):
        for epoch in range(num_epochs):
            model.train()
            
            # Create node batches for staged training
            node_batches = create_node_batches(num_nodes, batch_size)
            epoch_loss = 0.0
            
            for batch_start, batch_end in node_batches:
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(x_tensor, adj_list, dts_list)
                batch_logits = logits[batch_start:batch_end]
                batch_targets = y_tensor[batch_start:batch_end]
                batch_train_mask = train_mask[batch_start:batch_end]
                
                # Only compute loss for training nodes in this batch
                if batch_train_mask.sum() > 0:
                    train_logits = batch_logits[batch_train_mask]
                    train_targets = batch_targets[batch_train_mask]
                    
                    loss = focal_loss_weight(train_logits, train_targets, gamma=2.0, alpha=0.75)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += float(loss.item()) * batch_train_mask.sum().item()
            
            # Compute training metrics
            model.eval()
            with torch.no_grad():
                all_logits = model(x_tensor, adj_list, dts_list)
                train_proba = torch.sigmoid(all_logits[train_mask]).cpu().numpy()
                train_preds = (train_proba >= 0.5).astype(int).flatten()
                train_targets = y_tensor[train_mask].cpu().numpy()
                
                train_auc = float(roc_auc_score(train_targets, train_proba.flatten()))
                train_loss_avg = epoch_loss / max(train_mask.sum().item(), 1)
                train_losses.append(train_loss_avg)
            
            # Validation
            with torch.no_grad():
                val_proba = torch.sigmoid(all_logits[val_mask]).cpu().numpy()
                val_targets = y_tensor[val_mask].cpu().numpy()
                val_auc = float(roc_auc_score(val_targets, val_proba.flatten()))
                val_aucs.append(val_auc)
            
            scheduler.step()
            
            if (epoch + 1) % 2 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch+1:2d}/{num_epochs} | Loss: {train_loss_avg:.4f} | "
                    f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}"
                )
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  ⏹ Early stopping at epoch {epoch + 1}")
                    model.load_state_dict(best_state)
                    break
    
    # Final validation
    model.eval()
    with torch.no_grad():
        all_logits = model(x_tensor, adj_list, dts_list)
        val_proba = torch.sigmoid(all_logits[val_mask]).cpu().numpy().flatten()
        val_targets = y_tensor[val_mask].cpu().numpy()
        val_preds = (val_proba >= 0.5).astype(int)
        
        val_f1 = float(f1_score(val_targets, val_preds))
        val_auprc = float(average_precision_score(val_targets, val_proba))
        best_val_auc = float(roc_auc_score(val_targets, val_proba))
    
    logger.info(f"\n  📊 Final Validation Metrics:")
    logger.info(f"     Best Val AUC:  {best_val_auc:.4f}")
    logger.info(f"     Val F1-Score:  {val_f1:.4f}")
    logger.info(f"     Val AUPRC:     {val_auprc:.4f}")
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    hgnn_model_path = os.path.join(MODEL_DIR, "hgnn_att_td.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_relations": len(relations),
        "best_val_auc": best_val_auc,
    }, hgnn_model_path)
    logger.info(f"  Model saved → {hgnn_model_path}")
    
    return model, {"train_losses": train_losses, "val_aucs": val_aucs}


def train_all_models(
    X_train: Union[pd.DataFrame, np.ndarray], 
    y_train: Union[pd.Series, np.ndarray], 
    X_val: Union[pd.DataFrame, np.ndarray], 
    y_val: Union[pd.Series, np.ndarray], 
    hetero_data: Optional[Any] = None
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """
    Train all three models and return them as a dictionary.
    """
    logger.info("=" * 60)
    logger.info("     TRAINING ALL MODELS")
    logger.info("=" * 60)

    models: Dict[str, Any] = {}

    models["Decision Tree"] = train_decision_tree(X_train, y_train, X_val, y_val)
    models["XGBoost"] = train_xgboost(X_train, y_train, X_val, y_val)
    nn_model, nn_history = train_neural_network(X_train, y_train, X_val, y_val, hetero_data=hetero_data)
    models["Neural Network"] = nn_model

    logger.info("\n  ✅ All 3 models trained and saved successfully!")
    return models, nn_history
