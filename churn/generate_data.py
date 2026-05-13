"""
generate_data.py
────────────────
Generates synthetic churn dataset, writes a Parquet file locally, then uploads to S3.

S3 connectivity matches ``s3_shakeout.py``: the same environment variables are required
for upload (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_DEFAULT_REGION``,
``AWS_S3_BUCKET``, ``AWS_S3_ENDPOINT``). Optional ``AWS_SESSION_TOKEN``.

  python generate_data.py --n 2000 --key churn/train.parquet

``--bucket`` overrides ``AWS_S3_BUCKET`` for the upload destination only.

When run from Jupyter (%run), extra ipykernel argv is ignored (``parse_known_args``).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from s3_shakeout import load_config, make_s3_client


def generate_churn_dataset(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    plan_type = rng.choice(["basic", "pro", "enterprise"], n, p=[0.5, 0.35, 0.15])
    region    = rng.choice(["north", "south", "east", "west"], n)

    tenure_months   = rng.integers(1, 61, n)
    monthly_spend   = np.round(rng.uniform(10, 500, n), 2)
    support_tickets = rng.poisson(lam=2, size=n)

    # Churners show a downward usage trend over 6 months
    sequences = []
    for i in range(n):
        base  = rng.uniform(40, 90)
        trend = rng.uniform(-5, 1) if plan_type[i] == "basic" else rng.uniform(-2, 3)
        seq   = np.clip(
            base + trend * np.arange(6) + rng.normal(0, 5, 6), 0, 100
        )
        sequences.append(seq.tolist())

    # Rule-based churn label (learnable signal)
    churn_score = (
        (plan_type == "basic").astype(float) * 0.30 +
        (tenure_months < 12).astype(float)   * 0.20 +
        (support_tickets > 4).astype(float)  * 0.25 +
        (monthly_spend < 50).astype(float)   * 0.15 +
        rng.uniform(0, 0.10, n)
    )
    churned = (churn_score > 0.40).astype(int)

    df = pd.DataFrame({
        "plan_type":       plan_type,
        "region":          region,
        "tenure_months":   tenure_months,
        "monthly_spend":   monthly_spend,
        "support_tickets": support_tickets,
        "usage_seq":       sequences,
        "churned":         churned,
    })

    print(f"Generated {n} rows  |  churn rate: {churned.mean():.1%}")
    return df


def write_parquet_local(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Wrote {path.resolve()}")


def upload_file_to_s3(local_path: Path, bucket: str, key: str, *, client) -> None:
    client.upload_file(str(local_path), bucket, key)
    print(f"Uploaded to s3://{bucket}/{key}")


def upload_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    """Write ``df`` to a temp Parquet file and upload (same S3 client / env as ``s3_shakeout``)."""
    cfg = load_config()
    client = make_s3_client(cfg)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp = Path(f.name)
    try:
        df.to_parquet(tmp, index=False)
        upload_file_to_s3(tmp, bucket=bucket, key=key, client=client)
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="S3 bucket (default: AWS_S3_BUCKET from environment)",
    )
    parser.add_argument("--key", type=str, default="churn/train.parquet")
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help="Local Parquet path (default: same relative path as --key)",
    )
    args, _ = parser.parse_known_args()

    local_path = Path(args.local) if args.local else Path(args.key)

    cfg = load_config()
    client = make_s3_client(cfg)
    bucket = args.bucket if args.bucket is not None else cfg["bucket"]

    df = generate_churn_dataset(n=args.n)
    write_parquet_local(df, local_path)
    upload_file_to_s3(local_path, bucket=bucket, key=args.key, client=client)
