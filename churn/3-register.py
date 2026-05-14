"""
3-register.py
───────────
Registers the trained model if it meets quality thresholds.

Reads evaluation metrics, applies quality gate (AUC threshold),
and writes model metadata to S3 if the model passes.

Usage:
  python 3-register.py --metrics-file metrics.json \
                     --s3-model-key models/churn_embedding_model.pt \
                     --auc-threshold 0.70
"""

import argparse
import json
import sys
from pathlib import Path

# Add script directory to path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import boto3

from s3_shakeout import load_config, make_s3_client

# ── S3 Bucket Configuration ────────────────────────────────────────────────────
DATA_BUCKET = "data"
MODELS_BUCKET = "models"


def register_model(
    metrics: dict,
    s3_model_key: str,
    auc_threshold: float = 0.70,
):
    """
    Register model if it meets quality threshold.

    Args:
        metrics: Dictionary with evaluation metrics (must contain 'auc_roc')
        s3_model_key: S3 key where model is stored
        auc_threshold: Minimum AUC-ROC to register

    Raises:
        ValueError: If AUC is below threshold
    """
    cfg = load_config()
    s3 = make_s3_client(cfg)

    auc = metrics.get("auc_roc", 0.0)
    print(f"Model AUC: {auc}  Threshold: {auc_threshold}", file=sys.stderr)

    if auc < auc_threshold:
        raise ValueError(
            f"AUC {auc:.4f} below threshold {auc_threshold} — NOT registered."
        )

    # Write metadata file alongside the model
    metadata = {
        "model_s3_path": f"s3://{MODELS_BUCKET}/{s3_model_key}",
        "auc_roc": auc,
        "f1_score": metrics.get("f1_score"),
        "accuracy": metrics.get("accuracy"),
        "status": "registered",
    }

    metadata_key = s3_model_key.replace(".pt", "_metadata.json")

    s3.put_object(
        Bucket=MODELS_BUCKET,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode(),
    )

    print(f"✓ Model registered to s3://{MODELS_BUCKET}/{metadata_key}", file=sys.stderr)
    print(json.dumps(metadata, indent=2), file=sys.stderr)

    return metadata


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register churn embedding model")
    parser.add_argument("--metrics-file", required=True,
                        help="Path to JSON file with evaluation metrics")
    parser.add_argument("--s3-model-key", default="models/churn_embedding_model.pt",
                        help="S3 key for trained model")
    parser.add_argument("--auc-threshold", type=float, default=0.70,
                        help="Minimum AUC-ROC to register model")

    args = parser.parse_args()

    # Load metrics from file
    with open(args.metrics_file, "r") as f:
        metrics = json.load(f)

    register_model(
        metrics=metrics,
        s3_model_key=args.s3_model_key,
        auc_threshold=args.auc_threshold,
    )

    print("Registration complete.")
