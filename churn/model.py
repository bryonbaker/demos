"""
model.py
────────
Shared model and dataset definitions for the churn embedding model.

This module is imported by:
  - train.py (standalone training script)
  - evaluate.py (standalone evaluation script)
  - churn_pipeline.py (Kubeflow pipeline components)

Keeping the model definition in one place ensures consistency across
all pipeline steps and makes the code DRY.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ── Dataset ───────────────────────────────────────────────────────────────────

class ChurnDataset(Dataset):
    """
    PyTorch Dataset for churn prediction.

    Expects a DataFrame with columns:
      - plan_type (categorical): basic, pro, enterprise
      - region (categorical): north, south, east, west
      - tenure_months (numerical): 1-60
      - monthly_spend (numerical): $10-$500
      - support_tickets (numerical): 0-8
      - usage_seq (sequential): list of 6 monthly usage scores (0-100)
      - churned (target): 0 or 1
    """
    PLAN_MAP   = {"basic": 0, "pro": 1, "enterprise": 2}
    REGION_MAP = {"north": 0, "south": 1, "east": 2, "west": 3}

    def __init__(self, df: pd.DataFrame):
        # Categorical features → integer tensors
        self.cat = torch.tensor(
            np.stack([
                df["plan_type"].map(self.PLAN_MAP).values,
                df["region"].map(self.REGION_MAP).values,
            ], axis=1),
            dtype=torch.long
        )

        # Numerical features → float tensors
        self.num = torch.tensor(
            df[["tenure_months", "monthly_spend", "support_tickets"]].values,
            dtype=torch.float32
        )

        # Sequential feature → (N, 6, 1) tensor
        self.seq = torch.tensor(
            np.stack(df["usage_seq"].values),
            dtype=torch.float32
        ).unsqueeze(-1)

        # Target labels
        self.labels = torch.tensor(df["churned"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cat[idx], self.num[idx], self.seq[idx], self.labels[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class ChurnEmbeddingModel(nn.Module):
    """
    Churn prediction model with embedding layers.

    Architecture:
      - Categorical inputs  → Embedding tables (learned)
      - Numerical inputs    → Pass-through
      - Sequential inputs   → LSTM encoder
      - All features fused  → MLP → churn probability

    Args:
        embed_dim: Hidden dimension for the MLP head (default: 64)
    """
    # Embedding dimensions: (num_categories, embedding_dim)
    CAT_DIMS = [
        (3, 4),   # plan_type:  3 categories → 4-dim embedding
        (4, 4),   # region:     4 categories → 4-dim embedding
    ]

    def __init__(self, embed_dim: int = 64):
        super().__init__()

        # Categorical embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim)
            for num_cat, emb_dim in self.CAT_DIMS
        ])

        # Sequential encoder (LSTM)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Feature fusion + prediction head
        total_cat_dim = sum(e for _, e in self.CAT_DIMS)  # 8
        num_dim       = 3   # tenure, spend, tickets
        seq_dim       = 32  # LSTM hidden size

        self.head = nn.Sequential(
            nn.Linear(total_cat_dim + num_dim + seq_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, cat, num, seq):
        """
        Forward pass.

        Args:
            cat: (batch_size, 2) categorical features
            num: (batch_size, 3) numerical features
            seq: (batch_size, 6, 1) sequential features

        Returns:
            (batch_size,) churn probabilities
        """
        # Embed categorical features
        cat_vecs = [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_vec  = torch.cat(cat_vecs, dim=1)

        # Encode sequential features
        _, (hidden, _) = self.lstm(seq)
        seq_vec = hidden[-1]  # Take last hidden state

        # Fuse all features
        fused = torch.cat([cat_vec, num, seq_vec], dim=1)

        # Predict churn probability
        return self.head(fused).squeeze(1)
