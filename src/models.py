"""
models.py — Stage 6: Model Definitions
==========================================
Defines the three model architectures required by the project:
1. Decision Tree (sklearn)
2. XGBoost (gradient boosting)
3. FraudNet — Deep Neural Network (PyTorch)

All models implement a unified interface for training and prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from src.config import DT_PARAMS, XGB_PARAMS, NN_PARAMS, RANDOM_STATE
from src.utils import get_logger

logger = get_logger("Models")


# ══════════════════════════════════════════════════════════════
# 1. DECISION TREE — Interpretable Baseline
# ══════════════════════════════════════════════════════════════

def build_decision_tree():
    """
    Build a Decision Tree classifier with cost-sensitive weighting.

    class_weight='balanced' automatically adjusts weights inversely
    proportional to class frequencies, penalizing misclassification
    of the minority (fraud) class more heavily.

    Returns
    -------
    DecisionTreeClassifier
        Configured but unfitted decision tree.
    """
    model = DecisionTreeClassifier(**DT_PARAMS)
    logger.info(f"  🌳 Decision Tree created (max_depth={DT_PARAMS['max_depth']}, class_weight=balanced)")
    return model


# ══════════════════════════════════════════════════════════════
# 2. XGBOOST — Gradient Boosted Trees
# ══════════════════════════════════════════════════════════════

def build_xgboost(scale_pos_weight: float = None):
    """
    Build an XGBoost classifier with cost-sensitive learning.

    scale_pos_weight = n_negative / n_positive handles class imbalance
    by increasing the cost of misclassifying positive (fraud) samples.

    Parameters
    ----------
    scale_pos_weight : float
        Ratio of negative to positive samples. If None, computed from data.

    Returns
    -------
    XGBClassifier
        Configured but unfitted XGBoost model.
    """
    params = XGB_PARAMS.copy()
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    model = XGBClassifier(**params)
    logger.info(f"  🚀 XGBoost created (n_estimators={params['n_estimators']}, "
                f"max_depth={params['max_depth']}, scale_pos_weight={scale_pos_weight:.1f})")
    return model


# ══════════════════════════════════════════════════════════════
# 3. NEURAL NETWORK — Deep Learning with Focal Loss
# ══════════════════════════════════════════════════════════════

class FraudHGNN(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) for fraud detection.

    Architecture:
        1. Node-specific Linear projections to unify hidden dims.
        2. HGTConv layers for attention-based message passing across distinct edge types.
        3. Linear classifier over 'transaction' nodes → Sigmoid

    Uses transformer-style attention naturally separated by node/edge types.
    """

    def __init__(self, metadata, input_dim: int,
                 hidden_dims: list = None,
                 dropout_rates: list = None):
        super().__init__()
        # Use first hidden dim to unify projection
        hidden_dim = hidden_dims[0] if hidden_dims else NN_PARAMS["hidden_dims"][0]
        self.dropout_rate = dropout_rates[0] if dropout_rates else NN_PARAMS["dropout_rates"][0]

        # 1. Linear projection mapping for each node type
        self.node_lin = nn.ModuleDict()
        for node_type in metadata[0]:
            in_d = input_dim if node_type == 'transaction' else 1
            self.node_lin[node_type] = nn.Linear(in_d, hidden_dim)

        self.convs = nn.ModuleList()
        # 2 layers of HGT usually suffice
        for _ in range(2):
            self.convs.append(pyg_nn.HGTConv(in_channels=hidden_dim, 
                                             out_channels=hidden_dim, 
                                             metadata=metadata, 
                                             heads=4))

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x_dict, edge_index_dict, tx_time_decay=None):
        # Initial projection phase
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = F.gelu(self.node_lin[node_type](x))

        # Optional temporal decay (from TransactionDT recency) applied to
        # transaction embeddings before relation attention/message passing.
        if tx_time_decay is not None:
            decay = tx_time_decay
            if decay.dim() == 1:
                decay = decay.unsqueeze(-1)
            h_dict['transaction'] = h_dict['transaction'] * decay

        # HGT Message Passing
        for conv in self.convs:
            out_dict = conv(h_dict, edge_index_dict)
            # Add Residual and Dropout
            for node_type, h in out_dict.items():
                h_dict[node_type] = F.dropout(F.gelu(h), p=self.dropout_rate, training=self.training) + h_dict[node_type]

        logits = self.classifier(h_dict['transaction'])
        return logits

    def predict_proba(self, x_dict, edge_index_dict, tx_time_decay=None):
        """Get probability predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_dict, edge_index_dict, tx_time_decay=tx_time_decay)
            probs = torch.sigmoid(logits).squeeze()
        return probs


class FraudNet(nn.Module):
    """
    Standard Multi-Layer Perceptron (MLP) for tabular fraud detection.
    """
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], dropout_rates: list = [0.3, 0.3, 0.2]):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim, drop in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits).squeeze()
        return probs


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in neural networks.

    Focal Loss down-weights well-classified examples and focuses
    training on hard, misclassified examples. This is superior to
    simple class weighting for extreme imbalance ratios like ~1:28.

    Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter (default: 2.0). Higher gamma = more focus on hard examples.
    alpha : float
        Balancing factor for positive class (default: 0.75).
    """

    def __init__(self, gamma: float = None, alpha: float = None):
        super().__init__()
        self.gamma = gamma or NN_PARAMS["focal_gamma"]
        self.alpha = alpha or NN_PARAMS["focal_alpha"]

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).squeeze()
        targets = targets.float()

        # Binary cross-entropy component
        bce = F.binary_cross_entropy_with_logits(logits.squeeze(), targets, reduction="none")

        # Focal modulation: (1 - p_t) ^ gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_weight * focal_weight * bce
        return loss.mean()


def build_neural_network(input_dim: int, metadata=None):
    """
    Build the FraudHGNN neural network with HGT.

    Returns
    -------
    tuple of (FraudHGNN, FocalLoss, torch.optim.AdamW)
    """
    if metadata is None:
        raise ValueError("metadata (node types, edge types) is required for Heterogeneous Models.")
        
    model = FraudHGNN(metadata, input_dim)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=NN_PARAMS["learning_rate"],
        weight_decay=NN_PARAMS["weight_decay"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  🧠 FraudHGNN created ({n_params:,} parameters)")
    logger.info(f"     Architecture: HGTConv Layers")
    logger.info(f"     Loss: FocalLoss(gamma={NN_PARAMS['focal_gamma']}, alpha={NN_PARAMS['focal_alpha']})")
    return model, criterion, optimizer


# ══════════════════════════════════════════════════════════════
# 4. DENSE HGNN-ATT-TD — Heterogeneous Graph NN with Time Decay
# ══════════════════════════════════════════════════════════════

class DenseHGNN_ATT_TD(nn.Module):
    """
    Dense Heterogeneous Graph Neural Network with Attention and Time Decay.
    
    Architecture:
        1. Multiple relation-specific embeddings (card, addr, email, etc.)
        2. Time decay applied to each relational adjacency matrix
        3. Multi-view attention fusion combining all relation views
        4. Binary classification head with sigmoid
    
    Parameters
    ----------
    n_feat : int
        Input feature dimension
    n_hid : int
        Hidden embedding dimension
    n_class : int
        Output dimension (1 for binary classification)
    num_relations : int
        Number of heterogeneous relations (views)
    decay_init : float
        Initial decay rate (default: 0.1)
    """
    
    def __init__(self, n_feat: int, n_hid: int, n_class: int, num_relations: int, decay_init: float = 0.1):
        super().__init__()
        self.n_feat = n_feat
        self.n_hid = n_hid
        self.num_relations = num_relations
        
        # Relation-specific transformation matrices
        self.rel_weights = nn.ParameterList([
            nn.Parameter(torch.randn(n_feat, n_hid) * 0.01)
            for _ in range(num_relations)
        ])
        
        # Learnable temporal decay rate (0.001 to 1.0)
        self.decay_rate = nn.Parameter(torch.tensor(decay_init, dtype=torch.float32))
        
        # Multi-view attention: (num_relations, n_hid) -> (num_relations, 1)
        self.att_query = nn.Parameter(torch.randn(n_hid, 1) * 0.01)
        
        # Output classification layer
        self.out = nn.Linear(n_hid, n_class)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier uniform initialization for stable training."""
        for weight in self.rel_weights:
            nn.init.xavier_uniform_(weight)
        nn.init.xavier_uniform_(self.att_query)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    
    def forward(self, x: torch.Tensor, adjs: list, dts: list) -> torch.Tensor:
        """
        Forward pass with multi-view heterogeneous graph aggregation.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features (num_nodes, n_feat)
        adjs : list[torch.Tensor]
            List of normalized adjacency matrices
        dts : list[torch.Tensor]
            List of temporal distance matrices
            
        Returns
        -------
        torch.Tensor
            Binary classification logits (num_nodes, 1)
        """
        view_embs = []
        
        # Compute temporal decay for each relation view
        for i, (adj, dt) in enumerate(zip(adjs, dts)):
            # Exponential time decay: exp(-lambda * t)
            # clamp decay_rate to prevent numerical instability
            decay_lambda = torch.clamp(self.decay_rate, min=0.001, max=1.0)
            time_weights = torch.exp(-decay_lambda * dt)
            
            # Apply temporal weighting to adjacency
            weighted_adj = adj * time_weights
            
            # Graph convolution with relation-specific weights
            # h_view = (weighted_adj @ x) @ W_i
            h_view = torch.mm(weighted_adj, torch.mm(x, self.rel_weights[i]))
            view_embs.append(h_view)
        
        # Stack views: (num_nodes, num_relations, n_hid)
        stacked = torch.stack(view_embs, dim=1)
        
        # Attention mechanism over relations
        # alpha: (num_nodes, num_relations)
        alpha = torch.matmul(stacked, self.att_query).squeeze(-1)
        weights = F.softmax(alpha, dim=1).unsqueeze(-1)  # (num_nodes, num_relations, 1)
        
        # Fused embedding: weighted sum over relation views
        fused_embedding = torch.sum(stacked * weights, dim=1)
        
        # Classification
        logits = self.out(fused_embedding)
        return logits
    
    def predict_proba(self, x: torch.Tensor, adjs: list, dts: list) -> torch.Tensor:
        """Get probability predictions [0, 1]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, adjs, dts)
            probs = torch.sigmoid(logits).squeeze()
        return probs


def build_dense_hgnn_att_td(input_dim: int, hidden_dim: int = 64, num_relations: int = 3):
    """
    Build the DenseHGNN_ATT_TD model for fraud detection.
    
    Parameters
    ----------
    input_dim : int
        Number of input features per node
    hidden_dim : int
        Hidden embedding dimension (default: 64)
    num_relations : int
        Number of heterogeneous relations (default: 3)
        
    Returns
    -------
    tuple of (DenseHGNN_ATT_TD, torch.optim.AdamW)
    """
    model = DenseHGNN_ATT_TD(
        n_feat=input_dim,
        n_hid=hidden_dim,
        n_class=1,
        num_relations=num_relations,
        decay_init=0.1
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=NN_PARAMS.get("learning_rate", 1e-3),
        weight_decay=NN_PARAMS.get("weight_decay", 1e-5),
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  🧠 DenseHGNN-ATT-TD created ({n_params:,} parameters)")
    logger.info(f"     Architecture: {num_relations} relation views with temporal attention")
    logger.info(f"     Loss: Binary Cross-Entropy with Focal weighting")
    return model, optimizer
