"""
training/train.py
─────────────────
Churn embedding model training via Ray Train.
Each worker:
  1. Pulls its data shard from S3
  2. Trains with PyTorch DDP (via Ray Train)
  3. Reports metrics back to the Ray head
  4. Master worker saves the final model to S3
"""

import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import ray
import ray.train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, CheckpointConfig, RunConfig

from s3_shakeout import load_config, make_s3_client
from model import ChurnDataset, ChurnEmbeddingModel


# ── S3 helpers (same env + client as generate_data.py / s3_shakeout.py) ───────

def read_parquet_from_s3(key: str, *, bucket: str, client) -> pd.DataFrame:
    obj = client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def save_model_to_s3(model: nn.Module, key: str, *, bucket: str, client) -> None:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    client.put_object(Bucket=bucket, Key=key, Body=buffer)
    print(f"Model saved to s3://{bucket}/{key}")


# ── Per-worker training function ───────────────────────────────────────────────

def train_func(config: dict):
    """
    This function runs on EVERY Ray worker.
    Ray Train handles DDP setup automatically.
    """
    epochs      = config["epochs"]
    batch_size  = config["batch_size"]
    lr          = config["lr"]
    s3_key      = config["s3_data_key"]
    world_size  = ray.train.get_context().get_world_size()
    world_rank  = ray.train.get_context().get_world_rank()

    s3_cfg   = load_config()
    s3       = make_s3_client(s3_cfg)
    bucket   = s3_cfg["bucket"]

    print(f"[Worker {world_rank}/{world_size}] Starting — loading data from s3://{bucket}/{s3_key}")

    # Each worker loads the full dataset; Ray Train's DistributedSampler
    # automatically shards it so each worker sees a disjoint subset.
    df      = read_parquet_from_s3(s3_key, bucket=bucket, client=s3)
    dataset = ChurnDataset(df)

    sampler    = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=world_rank,
        shuffle=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Ray Train wraps the model in DDP and moves it to the correct GPU
    model     = ray.train.torch.prepare_model(ChurnEmbeddingModel())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for cat, num, seq, labels in dataloader:
            optimizer.zero_grad()
            preds = model(cat, num, seq)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # All workers report; Ray Train aggregates
        ray.train.report({"epoch": epoch + 1, "loss": avg_loss})
        print(f"[Worker {world_rank}] Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

    # Only master worker saves the model
    if world_rank == 0:
        # Unwrap DDP to get the raw module
        raw_model = model.module if hasattr(model, "module") else model
        save_model_to_s3(
            raw_model,
            "models/churn_embedding_model.pt",
            bucket=bucket,
            client=s3,
        )
        print("[Master] Model saved to S3.")


# ── Entry point ────────────────────────────────────────────────────────────────

def run_training(
    ray_address:  str  = "ray://churn-ray-cluster-head-svc:10001",
    s3_data_key:  str  = "churn/train.parquet",
    epochs:       int  = 10,
    batch_size:   int  = 256,
    lr:           float = 1e-3,
    num_workers:  int  = 2,
):
    ray.init(address=ray_address, ignore_reinit_error=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "epochs":      epochs,
            "batch_size":  batch_size,
            "lr":          lr,
            "s3_data_key": s3_data_key,
        },
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 4},
        ),
        run_config=RunConfig(
            name="churn-embedding-training",
            checkpoint_config=CheckpointConfig(num_to_keep=2),
        ),
    )

    result = trainer.fit()
    print("Training complete.")
    print(f"Final metrics: {result.metrics}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train churn embedding model with Ray")
    parser.add_argument("--ray-address", default="ray://churn-ray-cluster-head-svc:10001",
                        help="Ray cluster address")
    parser.add_argument("--s3-data-key", default="churn/train.parquet",
                        help="S3 key for training data")
    parser.add_argument("--s3-model-key", default="models/churn_embedding_model.pt",
                        help="S3 key for output model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size per worker")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of Ray workers")

    args = parser.parse_args()

    run_training(
        ray_address=args.ray_address,
        s3_data_key=args.s3_data_key,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
    )
