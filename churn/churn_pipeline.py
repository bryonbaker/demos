"""
churn_pipeline.py
──────────────────────────
Kubeflow Pipeline (KFP v2) that orchestrates:
  1. train_model      – trains the churn embedding model
  2. evaluate_model   – evaluates on held-out set
  3. register_model   – gates on AUC threshold before marking ready

Prerequisites:
  - Training data must exist in S3 (run 0-generate_data.py manually during demo prep)

Compile:
  python churn_pipeline.py          # writes churn_pipeline.yaml

Submit from Jupyter:
  import kfp
  client = kfp.Client(host="http://<kfp-endpoint>")
  client.create_run_from_pipeline_package("churn_pipeline.yaml")
"""

import os

import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics


BASE_IMAGE      = "python:3.10-slim"
# Object keys (defaults match 0-generate_data.py --key / --local layout)
S3_TRAIN_KEY    = "churn/train.parquet"
S3_MODEL_KEY    = "models/churn_embedding_model.pt"
# Default empty: pipeline steps use AWS_S3_BUCKET from the environment (see s3_shakeout.py).
DEFAULT_S3_BUCKET_PARAM = os.environ.get("AWS_S3_BUCKET", "")


# ── Component 1: Train Model ──────────────────────────────────────────────────

@component(
    base_image=BASE_IMAGE,
    packages_to_install=["torch", "boto3", "pandas", "numpy", "pyarrow"],
)
def train_model(
    s3_data_key:  str,
    s3_model_key: str,
    epochs:       int   = 10,
    batch_size:   int   = 256,
    lr:           float = 1e-3,
):
    """
    Trains the churn embedding model using PyTorch.
    Matches standalone 1-train.py logic.
    """
    import io, os, boto3, numpy as np, pandas as pd
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    # ── S3 config from environment ────────────────────────────────────────────
    _req = (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "AWS_S3_BUCKET",
        "AWS_S3_ENDPOINT",
    )
    _missing = [k for k in _req if not os.environ.get(k)]
    if _missing:
        raise RuntimeError("Missing required env: " + ", ".join(_missing))
    _cfg = {
        "access_key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "region": os.environ["AWS_DEFAULT_REGION"],
        "bucket": os.environ["AWS_S3_BUCKET"],
        "endpoint": os.environ["AWS_S3_ENDPOINT"],
    }
    _kw = dict(
        endpoint_url=_cfg["endpoint"],
        region_name=_cfg["region"],
        aws_access_key_id=_cfg["access_key"],
        aws_secret_access_key=_cfg["secret_key"],
    )
    _tok = os.environ.get("AWS_SESSION_TOKEN")
    if _tok:
        _kw["aws_session_token"] = _tok
    s3 = boto3.client("s3", **_kw)
    bucket = _cfg["bucket"]

    # ── Model & Dataset (inline for KFP serialization) ────────────────────────

    class ChurnDataset(Dataset):
        PLAN_MAP   = {"basic": 0, "pro": 1, "enterprise": 2}
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

        def __len__(self):  return len(self.labels)
        def __getitem__(self, i): return self.cat[i], self.num[i], self.seq[i], self.labels[i]

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

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading data from s3://{bucket}/{s3_data_key}")
    obj = s3.get_object(Bucket=bucket, Key=s3_data_key)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))

    # ── Training ───────────────────────────────────────────────────────────────
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

    # ── Save model ─────────────────────────────────────────────────────────────
    print(f"Saving model to s3://{bucket}/{s3_model_key}")
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=s3_model_key, Body=buffer)
    print("Training complete!")


# ── Component 2: Evaluate ──────────────────────────────────────────────────────

@component(
    base_image=BASE_IMAGE,
    packages_to_install=["boto3", "torch", "scikit-learn",
                         "pandas", "numpy", "pyarrow"],
)
def evaluate_model(
    s3_bucket:    str,
    s3_data_key:  str,
    s3_model_key: str,
    metrics:      Output[Metrics],
):
    """
    Evaluates the trained model (same logic as standalone 2-evaluate.py).

    Loads model and data from S3, computes metrics on held-out test set.
    """
    import io, os, boto3, numpy as np, pandas as pd
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    # S3 config from environment
    _req = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
            "AWS_S3_BUCKET", "AWS_S3_ENDPOINT")
    _missing = [k for k in _req if not os.environ.get(k)]
    if _missing:
        raise RuntimeError("Missing required env: " + ", ".join(_missing))
    _cfg = {
        "access_key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "region": os.environ["AWS_DEFAULT_REGION"],
        "bucket": os.environ["AWS_S3_BUCKET"],
        "endpoint": os.environ["AWS_S3_ENDPOINT"],
    }
    _kw = dict(endpoint_url=_cfg["endpoint"], region_name=_cfg["region"],
               aws_access_key_id=_cfg["access_key"],
               aws_secret_access_key=_cfg["secret_key"])
    _tok = os.environ.get("AWS_SESSION_TOKEN")
    if _tok:
        _kw["aws_session_token"] = _tok
    s3 = boto3.client("s3", **_kw)
    bucket = (s3_bucket or "").strip() or _cfg["bucket"]

    # Model definition (must match model.py)
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

    # Load model
    print(f"Loading model from s3://{bucket}/{s3_model_key}")
    obj   = s3.get_object(Bucket=bucket, Key=s3_model_key)
    state = torch.load(io.BytesIO(obj["Body"].read()), map_location="cpu")
    model = ChurnEmbeddingModel()
    model.load_state_dict(state)
    model.eval()

    # Load data (use last 20% as held-out test)
    print(f"Loading data from s3://{bucket}/{s3_data_key}")
    obj  = s3.get_object(Bucket=bucket, Key=s3_data_key)
    df   = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    df   = df.iloc[int(len(df) * 0.8):]  # held-out split
    print(f"Evaluating on {len(df)} held-out samples")

    PLAN_MAP   = {"basic": 0, "pro": 1, "enterprise": 2}
    REGION_MAP = {"north": 0, "south": 1, "east": 2, "west": 3}

    cat    = torch.tensor(np.stack([df["plan_type"].map(PLAN_MAP).values,
                                     df["region"].map(REGION_MAP).values], 1), dtype=torch.long)
    num    = torch.tensor(df[["tenure_months","monthly_spend","support_tickets"]].values, dtype=torch.float32)
    seq    = torch.tensor(np.stack(df["usage_seq"].values), dtype=torch.float32).unsqueeze(-1)
    labels = df["churned"].values

    with torch.no_grad():
        preds = model(cat, num, seq).numpy()

    auc = roc_auc_score(labels, preds)
    f1  = f1_score(labels, (preds > 0.5).astype(int))
    acc = accuracy_score(labels, (preds > 0.5).astype(int))

    metrics.log_metric("auc_roc",  round(float(auc), 4))
    metrics.log_metric("f1_score", round(float(f1),  4))
    metrics.log_metric("accuracy", round(float(acc), 4))
    print(f"AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")


# ── Component 3: Register ──────────────────────────────────────────────────────

@component(
    base_image=BASE_IMAGE,
    packages_to_install=["boto3"],
)
def register_model(
    s3_bucket:     str,
    s3_model_key:  str,
    metrics:       Input[Metrics],
    auc_threshold: float = 0.70,
):
    """
    Registers the model if it meets quality threshold (same logic as standalone 3-register.py).

    Writes model metadata to S3 if AUC >= threshold, otherwise raises ValueError.
    """
    import os, boto3, json

    # S3 config from environment
    _req = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
            "AWS_S3_BUCKET", "AWS_S3_ENDPOINT")
    _missing = [k for k in _req if not os.environ.get(k)]
    if _missing:
        raise RuntimeError("Missing required env: " + ", ".join(_missing))
    _cfg = {
        "access_key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "region": os.environ["AWS_DEFAULT_REGION"],
        "bucket": os.environ["AWS_S3_BUCKET"],
        "endpoint": os.environ["AWS_S3_ENDPOINT"],
    }
    _kw = dict(endpoint_url=_cfg["endpoint"], region_name=_cfg["region"],
               aws_access_key_id=_cfg["access_key"],
               aws_secret_access_key=_cfg["secret_key"])
    _tok = os.environ.get("AWS_SESSION_TOKEN")
    if _tok:
        _kw["aws_session_token"] = _tok
    s3 = boto3.client("s3", **_kw)
    bucket = (s3_bucket or "").strip() or _cfg["bucket"]

    auc = metrics.metadata.get("auc_roc", 0.0)
    f1  = metrics.metadata.get("f1_score", 0.0)
    acc = metrics.metadata.get("accuracy", 0.0)

    print(f"Model AUC: {auc}  Threshold: {auc_threshold}")

    if auc < auc_threshold:
        raise ValueError(f"AUC {auc:.4f} below threshold {auc_threshold} — NOT registered.")

    # Write metadata file alongside the model
    metadata = {
        "model_s3_path": f"s3://{bucket}/{s3_model_key}",
        "auc_roc":       auc,
        "f1_score":      f1,
        "accuracy":      acc,
        "status":        "registered",
    }

    metadata_key = s3_model_key.replace(".pt", "_metadata.json")
    s3.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode(),
    )

    print(f"✓ Model registered to s3://{bucket}/{metadata_key}")
    print(json.dumps(metadata, indent=2))


# ── Pipeline definition ────────────────────────────────────────────────────────

@pipeline(
    name="churn-embedding-pipeline",
    description="Churn embedding training pipeline (assumes data in S3)",
)
def churn_pipeline(
    s3_bucket:     str   = DEFAULT_S3_BUCKET_PARAM,
    s3_data_key:   str   = S3_TRAIN_KEY,
    s3_model_key:  str   = S3_MODEL_KEY,
    epochs:        int   = 10,
    batch_size:    int   = 256,
    lr:            float = 1e-3,
    auc_threshold: float = 0.70,
):
    # Step 1 — training
    train_task = train_model(
        s3_data_key=s3_data_key,
        s3_model_key=s3_model_key,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # Step 2 — evaluate
    eval_task = evaluate_model(
        s3_bucket=s3_bucket,
        s3_data_key=s3_data_key,
        s3_model_key=s3_model_key,
    )
    eval_task.after(train_task)

    # Step 3 — register
    register_model(
        s3_bucket=s3_bucket,
        s3_model_key=s3_model_key,
        metrics=eval_task.outputs["metrics"],
        auc_threshold=auc_threshold,
    )


# ── Compile ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=churn_pipeline,
        package_path="churn_pipeline.yaml",
    )
    print("Pipeline compiled → churn_pipeline.yaml")
