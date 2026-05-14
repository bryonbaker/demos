"""
1-train.py
─────────────────
Churn embedding model training with PyTorch.

Steps:
  1. Loads training data from S3
  2. Trains the model
  3. Saves the trained model to S3
"""

import sys
from pathlib import Path

# Add script directory to path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from s3_shakeout import load_config, make_s3_client
from model import ChurnDataset, ChurnEmbeddingModel


# ── S3 helpers ─────────────────────────────────────────────────────────────────

def read_parquet_from_s3(key: str, *, bucket: str, client) -> pd.DataFrame:
    obj = client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def save_model_to_s3(model: nn.Module, key: str, *, bucket: str, client) -> None:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    client.put_object(Bucket=bucket, Key=key, Body=buffer)
    print(f"Model saved to s3://{bucket}/{key}")


# ── Training function ──────────────────────────────────────────────────────────

def train_model(
    s3_data_key: str = "churn/train.parquet",
    s3_model_key: str = "models/churn_embedding_model.pt",
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """Train the churn model and save to S3."""

    # Setup S3
    s3_cfg = load_config()
    s3 = make_s3_client(s3_cfg)
    bucket = s3_cfg["bucket"]

    print(f"Loading data from s3://{bucket}/{s3_data_key}")
    df = read_parquet_from_s3(s3_data_key, bucket=bucket, client=s3)

    # Create dataset and dataloader
    dataset = ChurnDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ChurnEmbeddingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for cat, num, seq, labels in dataloader:
            cat = cat.to(device)
            num = num.to(device)
            seq = seq.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(cat, num, seq)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

    # Save model to S3
    print(f"\nSaving model to s3://{bucket}/{s3_model_key}")
    save_model_to_s3(model, s3_model_key, bucket=bucket, client=s3)
    print("Training complete!")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train churn embedding model")
    parser.add_argument("--s3-data-key", default="churn/train.parquet",
                        help="S3 key for training data")
    parser.add_argument("--s3-model-key", default="models/churn_embedding_model.pt",
                        help="S3 key for output model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")

    args = parser.parse_args()

    train_model(
        s3_data_key=args.s3_data_key,
        s3_model_key=args.s3_model_key,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
