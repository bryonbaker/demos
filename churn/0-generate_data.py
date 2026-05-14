"""
0-generate_data.py
────────────────
Generates synthetic churn dataset, writes a Parquet file locally, then uploads to S3.

S3 connectivity matches ``s3_shakeout.py``: the same environment variables are required
for upload (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_DEFAULT_REGION``,
``AWS_S3_BUCKET``, ``AWS_S3_ENDPOINT``). Optional ``AWS_SESSION_TOKEN``.

  python 0-generate_data.py --n 2000 --key churn/train.parquet

Data is uploaded to the DATA_BUCKET constant defined in this script.

When run from Jupyter (%run), extra ipykernel argv is ignored (``parse_known_args``).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Add script directory to path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import numpy as np
import pandas as pd

from s3_shakeout import load_config, make_s3_client

# ── S3 Bucket Configuration ────────────────────────────────────────────────────
DATA_BUCKET = "data"


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

    df = generate_churn_dataset(n=args.n)
    write_parquet_local(df, local_path)
    upload_file_to_s3(local_path, bucket=DATA_BUCKET, key=args.key, client=client)
