"""
hgnn_utils.py — Heterogeneous Graph Construction & Training Utilities
=========================================================================
Functions for building multi-view heterogeneous graphs with temporal decay,
memory-efficient batch processing, and staged training for large datasets.
"""

import gc
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Optional
from src.utils import get_logger, Timer

logger = get_logger("HGNN_Utils")


# ══════════════════════════════════════════════════════════════
# 1. DENSE GRAPH CONSTRUCTION (Memory-Optimized)
# ══════════════════════════════════════════════════════════════

def build_dense_graphs(
    df: pd.DataFrame,
    relations: List[str],
    max_nodes: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Build dense heterogeneous adjacency matrices with temporal decay.
    
    For each relation, creates:
    - Adjacency matrix: connects nodes that share the relation value
    - Time distance matrix: temporal gaps between transactions
    
    Memory Optimization:
    - Limits to max_nodes to stay within GPU memory (recommended: 30k for 80GB GPU)
    - Normalizes adjacency after construction
    - Uses CPU tensors first, transfers to device after
    
    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame with 'TransactionDT' and relation columns
    relations : List[str]
        Relation columns (e.g., ['card1', 'addr1', 'P_emaildomain'])
    max_nodes : int, optional
        Max nodes to use. If exceeded, random sample is taken (default: None = all)
    device : str
        Target device ('cpu', 'cuda', 'mps')
        
    Returns
    -------
    tuple
        (adj_list, dts_list) where each is a list of torch tensors on CPU
    """
    num_original = len(df)
    
    # Sample if needed
    if max_nodes is not None and num_original > max_nodes:
        logger.info(f"  📉 Sampling {max_nodes} nodes from {num_original} (memory optimization)")
        df = df.sample(max_nodes, random_state=42).reset_index(drop=True)
    
    num_nodes = len(df)
    adj_list = []
    dts_list = []
    
    for rel in relations:
        logger.info(f"  🔗 Building {rel} adjacency matrix ({num_nodes} nodes)...")
        
        # Initialize on CPU, then move to device if needed
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        dt_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        
        # Group indices by relation value
        groups = df.groupby(rel).indices
        
        for group_val, g_idx in groups.items():
            if len(g_idx) > 1:
                g_idx_tensor = torch.tensor(g_idx, dtype=torch.long)
                
                # All-to-all connections within group
                rows = g_idx_tensor.repeat_interleave(len(g_idx))
                cols = g_idx_tensor.repeat(len(g_idx))
                
                adj[rows, cols] = 1.0
                
                # Temporal distance: |t_i - t_j|
                times = torch.tensor(
                    df.iloc[g_idx]['TransactionDT'].values,
                    dtype=torch.float32
                )
                temporal_gaps = torch.abs(
                    times.repeat_interleave(len(g_idx)) - times.repeat(len(g_idx))
                )
                dt_matrix[rows, cols] = temporal_gaps
        
        # Normalize adjacency matrix (row-wise)
        row_sums = adj.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-6)  # Avoid division by zero
        adj = adj / row_sums
        
        # Move to device
        if device != "cpu":
            adj = adj.to(device)
            dt_matrix = dt_matrix.to(device)
        
        adj_list.append(adj)
        dts_list.append(dt_matrix)
        
        # Memory cleanup
        del adj, dt_matrix
        gc.collect()
    
    logger.info(f"  ✅ Built {len(relations)} relational views ({num_nodes} nodes)")
    return adj_list, dts_list


# ══════════════════════════════════════════════════════════════
# 2. STAGED TRAINING (Batch Processing for Large Graphs)
# ══════════════════════════════════════════════════════════════

def create_node_batches(
    num_nodes: int,
    batch_size: int = 2000,
) -> List[Tuple[int, int]]:
    """
    Create batches of node indices for staged training.
    
    Parameters
    ----------
    num_nodes : int
        Total number of nodes
    batch_size : int
        Nodes per batch (default: 2000)
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (start_idx, end_idx) tuples for each batch
    """
    batches = []
    for start_idx in range(0, num_nodes, batch_size):
        end_idx = min(start_idx + batch_size, num_nodes)
        batches.append((start_idx, end_idx))
    return batches


def get_batch_indices_for_gradient_accumulation(
    num_nodes: int,
    accumulation_steps: int = 4,
) -> List[List[int]]:
    """
    Create batches for gradient accumulation during forward/backward passes.
    
    Parameters
    ----------
    num_nodes : int
        Total number of nodes
    accumulation_steps : int
        Number of gradient accumulation steps (default: 4)
        
    Returns
    -------
    List[List[int]]
        List of node indices for each accumulation step
    """
    step_size = max(1, num_nodes // accumulation_steps)
    batches = []
    for i in range(accumulation_steps):
        start = i * step_size
        end = min((i + 1) * step_size, num_nodes) if i < accumulation_steps - 1 else num_nodes
        batches.append(list(range(start, end)))
    return batches


# ══════════════════════════════════════════════════════════════
# 3. MEMORY PROFILING & OPTIMIZATION
# ══════════════════════════════════════════════════════════════

def estimate_graph_memory(num_nodes: int, num_relations: int) -> float:
    """
    Estimate GPU memory needed for dense graphs (in GB).
    
    Each adjacency/time matrix: num_nodes^2 * float32 = N^2 * 4 bytes
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes
    num_relations : int
        Number of relation views
        
    Returns
    -------
    float
        Estimated memory in GB
    """
    bytes_per_matrix = (num_nodes ** 2) * 4
    total_bytes = bytes_per_matrix * num_relations * 2  # adj + time distance
    gb = total_bytes / (1024 ** 3)
    return gb


def get_safe_max_nodes_for_dgx(gpu_memory_gb: int = 80, safety_margin: float = 0.7) -> int:
    """
    Calculate safe max nodes for DGX with given GPU memory.
    
    Uses empirical formula: N^2 * num_relations * 2 matrices * 4 bytes <= safety_margin * GPU_GB
    
    Parameters
    ----------
    gpu_memory_gb : int
        Available GPU memory in GB (default: 80 for A100)
    safety_margin : float
        Safety margin (0.7 = use 70% of available) (default: 0.7)
        
    Returns
    -------
    int
        Safe maximum number of nodes
    """
    # Assume 3 relations
    num_relations = 3
    available_bytes = gpu_memory_gb * (1024 ** 3) * safety_margin
    # N^2 * 3 * 2 * 4 = available_bytes
    # N^2 = available_bytes / (3 * 2 * 4) = available_bytes / 24
    max_n_squared = available_bytes / (num_relations * 2 * 4)
    max_nodes = int(np.sqrt(max_n_squared))
    return max_nodes


def log_graph_info(
    x_shape: tuple,
    adj_list: List[torch.Tensor],
    dts_list: List[torch.Tensor],
):
    """Log graph construction statistics."""
    num_nodes, num_features = x_shape
    num_relations = len(adj_list)
    
    logger.info(f"\n  📊 Graph Construction Summary:")
    logger.info(f"     Nodes: {num_nodes:,}")
    logger.info(f"     Features: {num_features}")
    logger.info(f"     Relations: {num_relations}")
    
    total_edges = 0
    for i, adj in enumerate(adj_list):
        num_edges = int((adj > 0).sum().item())
        total_edges += num_edges
        logger.info(f"     Relation {i}: {num_edges:,} edges")
    
    estimated_mem = estimate_graph_memory(num_nodes, num_relations)
    logger.info(f"     Est. Memory: {estimated_mem:.2f} GB")


# ══════════════════════════════════════════════════════════════
# 4. TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════

def prepare_training_data(
    y: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare labels and masks for training.
    
    Parameters
    ----------
    y : np.ndarray
        Binary labels (0/1)
    train_indices : np.ndarray
        Training set indices
    val_indices : np.ndarray
        Validation set indices
        
    Returns
    -------
    tuple
        (y_tensor, train_mask, val_mask, test_mask)
    """
    n = len(y)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    
    logger.info(f"  ✅ Training data prepared: {len(train_indices)} train, {len(val_indices)} val")
    
    return y_tensor, train_mask, val_mask


def focal_loss_weight(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    """
    Apply focal loss weighting to imbalanced classification.
    
    Parameters
    ----------
    logits : torch.Tensor
        Model predictions
    targets : torch.Tensor
        Ground truth labels
    gamma : float
        Focusing parameter
    alpha : float
        Balance parameter
        
    Returns
    -------
    torch.Tensor
        Weighted losses
    """
    probs = torch.sigmoid(logits).squeeze()
    targets_float = targets.float()
    
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(), targets_float, reduction="none"
    )
    
    p_t = probs * targets_float + (1 - probs) * (1 - targets_float)
    focal_weight = (1 - p_t) ** gamma
    
    alpha_weight = alpha * targets_float + (1 - alpha) * (1 - targets_float)
    
    weighted_loss = alpha_weight * focal_weight * bce
    return weighted_loss.mean()
