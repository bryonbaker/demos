"""
churn_pipeline.py
──────────────────────────
Kubeflow Pipeline (KFP v2) that orchestrates:
  1. train_model      – trains the churn embedding model
  2. evaluate_model   – evaluates on held-out set
  3. register_model   – gates on AUC threshold before marking ready

Prerequisites:
  - Training data must exist in S3 (run 0-generate_data.py manually during demo prep)
  - S3 credentials configured in Kubernetes secret (default: 'aws-connection-churn')

Usage:
  # Run pipeline programmatically (from within OpenShift AI workbench)
  python churn_pipeline.py

  # Or compile to YAML for manual import
  python churn_pipeline.py --compile
"""

import kfp
import kfp.dsl as dsl
from kfp.dsl import component, pipeline, Input, Output, Metrics
from kfp import kubernetes


# ── Configuration ──────────────────────────────────────────────────────────────

BASE_IMAGE = "python:3.11-slim"

# S3 Bucket Configuration (must match standalone scripts)
DATA_BUCKET = "data"
MODELS_BUCKET = "models"

# Default S3 keys
DEFAULT_DATA_KEY = "churn/train.parquet"
DEFAULT_MODEL_KEY = "models/churn_embedding_model.pt"

# Kubernetes secret containing S3 credentials
S3_SECRET_NAME = "aws-connection-churn"


# ── Component 1: Train Model ──────────────────────────────────────────────────

@component(
    base_image=BASE_IMAGE,
    packages_to_install=["torch", "boto3", "pandas", "numpy", "pyarrow"],
)
def train_model(
    s3_data_key: str,
    s3_model_key: str,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """
    Trains the churn embedding model using PyTorch.
    Matches standalone 1-train.py logic.
    """
    import io
    import os
    import boto3
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    # ── S3 Bucket Configuration ───────────────────────────────────────────────
    DATA_BUCKET = "data"
    MODELS_BUCKET = "models"

    # ── S3 Client Setup ───────────────────────────────────────────────────────
    REQUIRED_ENV = (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "AWS_S3_ENDPOINT",
    )
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env: {', '.join(missing)}")

    s3_kwargs = {
        "endpoint_url": os.environ["AWS_S3_ENDPOINT"],
        "region_name": os.environ["AWS_DEFAULT_REGION"],
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    }
    if os.environ.get("AWS_SESSION_TOKEN"):
        s3_kwargs["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]

    s3 = boto3.client("s3", **s3_kwargs)

    # ── Model & Dataset (inline for KFP serialization) ───────────────────────

    class ChurnDataset(Dataset):
        PLAN_MAP = {"basic": 0, "pro": 1, "enterprise": 2}
        REGION_MAP = {"north": 0, "south": 1, "east": 2, "west": 3}

        def __init__(self, df):
            self.cat = torch.tensor(
                np.stack([df["plan_type"].map(self.PLAN_MAP).values,
                          df["region"].map(self.REGION_MAP).values], axis=1),
                dtype=torch.long)
            self.num = torch.tensor(
                df[["tenure_months", "monthly_spend", "support_tickets"]].values,
                dtype=torch.float32)
            self.seq = torch.tensor(
                np.stack(df["usage_seq"].values),
                dtype=torch.float32).unsqueeze(-1)
            self.labels = torch.tensor(df["churned"].values, dtype=torch.float32)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return self.cat[i], self.num[i], self.seq[i], self.labels[i]

    class ChurnEmbeddingModel(nn.Module):
        CAT_DIMS = [(3, 4), (4, 4)]

        def __init__(self, embed_dim=64):
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(n, d) for n, d in self.CAT_DIMS])
            self.lstm = nn.LSTM(1, 32, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(
                nn.Linear(8 + 3 + 32, embed_dim), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
                nn.Linear(embed_dim // 2, 1), nn.Sigmoid())

        def forward(self, cat, num, seq):
            cv = torch.cat([e(cat[:, i]) for i, e in enumerate(self.embeddings)], 1)
            _, (h, _) = self.lstm(seq)
            return self.head(torch.cat([cv, num, h[-1]], 1)).squeeze(1)

    # ── Load Data ─────────────────────────────────────────────────────────────
    print(f"Loading data from s3://{DATA_BUCKET}/{s3_data_key}")
    obj = s3.get_object(Bucket=DATA_BUCKET, Key=s3_data_key)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))

    # ── Training ──────────────────────────────────────────────────────────────
    dataset = ChurnDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ChurnEmbeddingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print(f"Starting training for {epochs} epochs...")
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

    # ── Save Model ────────────────────────────────────────────────────────────
    print(f"Saving model to s3://{MODELS_BUCKET}/{s3_model_key}")
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    s3.put_object(Bucket=MODELS_BUCKET, Key=s3_model_key, Body=buffer)
    print("Training complete!")


# ── Component 2: Evaluate Model ───────────────────────────────────────────────

@component(
    base_image=BASE_IMAGE,
    packages_to_install=["boto3", "torch", "scikit-learn", "pandas", "numpy", "pyarrow"],
)
def evaluate_model(
    s3_data_key: str,
    s3_model_key: str,
    metrics: Output[Metrics],
):
    """
    Evaluates the trained model.
    Matches standalone 2-evaluate.py logic.
    """
    import io
    import os
    import boto3
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    # ── S3 Bucket Configuration ───────────────────────────────────────────────
    DATA_BUCKET = "data"
    MODELS_BUCKET = "models"

    # ── S3 Client Setup ───────────────────────────────────────────────────────
    REQUIRED_ENV = (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "AWS_S3_ENDPOINT",
    )
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env: {', '.join(missing)}")

    s3_kwargs = {
        "endpoint_url": os.environ["AWS_S3_ENDPOINT"],
        "region_name": os.environ["AWS_DEFAULT_REGION"],
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    }
    if os.environ.get("AWS_SESSION_TOKEN"):
        s3_kwargs["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]

    s3 = boto3.client("s3", **s3_kwargs)

    # ── Model Definition (must match training) ────────────────────────────────
    class ChurnEmbeddingModel(nn.Module):
        CAT_DIMS = [(3, 4), (4, 4)]

        def __init__(self, embed_dim=64):
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(n, d) for n, d in self.CAT_DIMS])
            self.lstm = nn.LSTM(1, 32, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(
                nn.Linear(8 + 3 + 32, embed_dim), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
                nn.Linear(embed_dim // 2, 1), nn.Sigmoid())

        def forward(self, cat, num, seq):
            cv = torch.cat([e(cat[:, i]) for i, e in enumerate(self.embeddings)], 1)
            _, (h, _) = self.lstm(seq)
            return self.head(torch.cat([cv, num, h[-1]], 1)).squeeze(1)

    # ── Load Model ────────────────────────────────────────────────────────────
    print(f"Loading model from s3://{MODELS_BUCKET}/{s3_model_key}")
    obj = s3.get_object(Bucket=MODELS_BUCKET, Key=s3_model_key)
    state = torch.load(io.BytesIO(obj["Body"].read()), map_location="cpu")
    model = ChurnEmbeddingModel()
    model.load_state_dict(state)
    model.eval()

    # ── Load Data (use last 20% as held-out test) ────────────────────────────
    print(f"Loading data from s3://{DATA_BUCKET}/{s3_data_key}")
    obj = s3.get_object(Bucket=DATA_BUCKET, Key=s3_data_key)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    df = df.iloc[int(len(df) * 0.8):]  # held-out split

    print(f"Evaluating on {len(df)} held-out samples")

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

    # ── Evaluate ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        preds = model(cat, num, seq).numpy()

    auc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, (preds > 0.5).astype(int))
    acc = accuracy_score(labels, (preds > 0.5).astype(int))

    # ── Log Metrics ───────────────────────────────────────────────────────────
    metrics.log_metric("auc_roc", round(float(auc), 4))
    metrics.log_metric("f1_score", round(float(f1), 4))
    metrics.log_metric("accuracy", round(float(acc), 4))

    print(f"AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")


# ── Component 3: Register Model ───────────────────────────────────────────────

@component(
    base_image=BASE_IMAGE,
    packages_to_install=["boto3"],
)
def register_model(
    s3_model_key: str,
    metrics: Input[Metrics],
    auc_threshold: float = 0.70,
):
    """
    Registers the model if it meets quality threshold.
    Matches standalone 3-register.py logic.
    """
    import os
    import json
    import boto3

    # ── S3 Bucket Configuration ───────────────────────────────────────────────
    MODELS_BUCKET = "models"

    # ── S3 Client Setup ───────────────────────────────────────────────────────
    REQUIRED_ENV = (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "AWS_S3_ENDPOINT",
    )
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env: {', '.join(missing)}")

    s3_kwargs = {
        "endpoint_url": os.environ["AWS_S3_ENDPOINT"],
        "region_name": os.environ["AWS_DEFAULT_REGION"],
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    }
    if os.environ.get("AWS_SESSION_TOKEN"):
        s3_kwargs["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]

    s3 = boto3.client("s3", **s3_kwargs)

    # ── Check Quality Gate ────────────────────────────────────────────────────
    auc = metrics.metadata.get("auc_roc", 0.0)
    f1 = metrics.metadata.get("f1_score", 0.0)
    acc = metrics.metadata.get("accuracy", 0.0)

    print(f"Model AUC: {auc}  Threshold: {auc_threshold}")

    if auc < auc_threshold:
        raise ValueError(
            f"AUC {auc:.4f} below threshold {auc_threshold} — NOT registered."
        )

    # ── Write Metadata ────────────────────────────────────────────────────────
    metadata = {
        "model_s3_path": f"s3://{MODELS_BUCKET}/{s3_model_key}",
        "auc_roc": auc,
        "f1_score": f1,
        "accuracy": acc,
        "status": "registered",
    }

    metadata_key = s3_model_key.replace(".pt", "_metadata.json")

    s3.put_object(
        Bucket=MODELS_BUCKET,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode(),
    )

    print(f"✓ Model registered to s3://{MODELS_BUCKET}/{metadata_key}")
    print(json.dumps(metadata, indent=2))


# ── Pipeline Definition ───────────────────────────────────────────────────────

@dsl.pipeline(
    name="churn-embedding-pipeline",
    description="Churn embedding training pipeline (assumes data in S3)",
)
def churn_pipeline(
    s3_data_key: str = DEFAULT_DATA_KEY,
    s3_model_key: str = DEFAULT_MODEL_KEY,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    auc_threshold: float = 0.70,
):
    """
    Churn model training pipeline.

    Parameters:
        s3_data_key: S3 key for training data (in DATA_BUCKET)
        s3_model_key: S3 key for output model (in MODELS_BUCKET)
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        auc_threshold: Minimum AUC to register model
    """
    # Step 1 — Train
    train_task = train_model(
        s3_data_key=s3_data_key,
        s3_model_key=s3_model_key,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    # Inject S3 credentials from secret
    kubernetes.use_secret_as_env(
        train_task,
        secret_name=S3_SECRET_NAME,
        secret_key_to_env={
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        },
    )

    # Step 2 — Evaluate
    eval_task = evaluate_model(
        s3_data_key=s3_data_key,
        s3_model_key=s3_model_key,
    )
    eval_task.after(train_task)
    # Inject S3 credentials from secret
    kubernetes.use_secret_as_env(
        eval_task,
        secret_name=S3_SECRET_NAME,
        secret_key_to_env={
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        },
    )

    # Step 3 — Register
    register_task = register_model(
        s3_model_key=s3_model_key,
        metrics=eval_task.outputs["metrics"],
        auc_threshold=auc_threshold,
    )
    # Inject S3 credentials from secret
    kubernetes.use_secret_as_env(
        register_task,
        secret_name=S3_SECRET_NAME,
        secret_key_to_env={
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        },
    )


# ── Main: Programmatic Execution ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Check if user wants to compile to YAML instead of running
    if "--compile" in sys.argv:
        kfp.compiler.Compiler().compile(
            pipeline_func=churn_pipeline,
            package_path="churn_pipeline.yaml",
        )
        print("✓ Pipeline compiled → churn_pipeline.yaml")
        print("\nTo upload to OpenShift AI:")
        print("  1. Go to Data Science Pipelines")
        print("  2. Click 'Import pipeline'")
        print("  3. Upload churn_pipeline.yaml")
        print("  4. Create a run with your parameters")
        sys.exit(0)

    # Programmatic execution (default)
    print("Running pipeline programmatically...")

    # Pipeline parameters
    pipeline_args = {
        "s3_data_key": "churn/train.parquet",
        "s3_model_key": "models/churn_embedding_model.pt",
        "epochs": 10,
        "batch_size": 256,
        "lr": 0.001,
        "auc_threshold": 0.70,
    }

    # Connect to Data Science Pipelines
    # Read namespace from service account
    namespace_file_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    with open(namespace_file_path, "r") as namespace_file:
        namespace = namespace_file.read()

    kubeflow_endpoint = f"https://ds-pipeline-dspa.{namespace}.svc:8443"

    # Read service account token
    sa_token_file_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    with open(sa_token_file_path, "r") as token_file:
        bearer_token = token_file.read()

    ssl_ca_cert = "/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"

    print(f"Connecting to Data Science Pipelines: {kubeflow_endpoint}")
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=ssl_ca_cert,
    )

    # Submit pipeline run
    print("Submitting pipeline run...")
    client.create_run_from_pipeline_func(
        churn_pipeline,
        arguments=pipeline_args,
        experiment_name="churn-embedding-experiment",
        enable_caching=True,
    )

    print("✓ Pipeline run submitted successfully!")
    print(f"  Check the Data Science Pipelines UI in namespace '{namespace}'")
