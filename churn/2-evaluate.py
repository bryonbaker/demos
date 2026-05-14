"""
2-evaluate.py
───────────
Evaluates the trained churn embedding model on held-out test data.

Loads model and data from S3, computes metrics (AUC-ROC, F1, Accuracy),
and writes results to stdout (JSON format for easy parsing by pipeline).

Usage:
  python 2-evaluate.py --s3-data-key churn/train.parquet \
                     --s3-model-key models/churn_embedding_model.pt
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

# Add script directory to path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import boto3
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from s3_shakeout import load_config, make_s3_client
from model import ChurnEmbeddingModel


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate_model(s3_data_key: str, s3_model_key: str, s3_bucket: str = None):
    """
    Load model and data from S3, evaluate on held-out test set.

    Returns:
        dict with keys: auc_roc, f1_score, accuracy
    """
    cfg = load_config()
    s3 = make_s3_client(cfg)
    bucket = s3_bucket if s3_bucket else cfg["bucket"]

    print(f"Loading model from s3://{bucket}/{s3_model_key}", file=sys.stderr)

    # Load model
    obj = s3.get_object(Bucket=bucket, Key=s3_model_key)
    state = torch.load(io.BytesIO(obj["Body"].read()), map_location="cpu")
    model = ChurnEmbeddingModel()
    model.load_state_dict(state)
    model.eval()

    print(f"Loading data from s3://{bucket}/{s3_data_key}", file=sys.stderr)

    # Load data (use last 20% as held-out test)
    obj = s3.get_object(Bucket=bucket, Key=s3_data_key)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    df = df.iloc[int(len(df) * 0.8):]  # held-out split

    print(f"Evaluating on {len(df)} held-out samples", file=sys.stderr)

    PLAN_MAP = {"basic": 0, "pro": 1, "enterprise": 2}
    REGION_MAP = {"north": 0, "south": 1, "east": 2, "west": 3}

    cat = torch.tensor(
        np.stack([df["plan_type"].map(PLAN_MAP).values,
                  df["region"].map(REGION_MAP).values], 1),
        dtype=torch.long)
    num = torch.tensor(
        df[["tenure_months", "monthly_spend", "support_tickets"]].values,
        dtype=torch.float32)
    seq = torch.tensor(
        np.stack(df["usage_seq"].values),
        dtype=torch.float32).unsqueeze(-1)
    labels = df["churned"].values

    with torch.no_grad():
        preds = model(cat, num, seq).numpy()

    auc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, (preds > 0.5).astype(int))
    acc = accuracy_score(labels, (preds > 0.5).astype(int))

    metrics = {
        "auc_roc": round(float(auc), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
    }

    print(f"AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}", file=sys.stderr)

    return metrics


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate churn embedding model")
    parser.add_argument("--s3-bucket", default=None,
                        help="S3 bucket (default: from AWS_S3_BUCKET env)")
    parser.add_argument("--s3-data-key", default="churn/train.parquet",
                        help="S3 key for training data")
    parser.add_argument("--s3-model-key", default="models/churn_embedding_model.pt",
                        help="S3 key for trained model")
    parser.add_argument("--output", default=None,
                        help="Output JSON file for metrics (default: stdout)")

    args = parser.parse_args()

    metrics = evaluate_model(
        s3_data_key=args.s3_data_key,
        s3_model_key=args.s3_model_key,
        s3_bucket=args.s3_bucket,
    )

    # Write metrics as JSON
    output = json.dumps(metrics, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Metrics written to {args.output}", file=sys.stderr)
    else:
        print(output)
