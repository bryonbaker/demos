#!/usr/bin/env python3
"""
S3 bucket shakeout: verify connectivity to a custom S3 endpoint by creating a
local file, uploading it, downloading it, comparing bytes, then deleting the object.

Required environment variables:

  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_DEFAULT_REGION
  AWS_S3_BUCKET
  AWS_S3_ENDPOINT

Optional:

  AWS_SESSION_TOKEN   — temporary credentials

  python s3_shakeout.py
  python s3_shakeout.py --prefix my-shakeout/

When run from a Jupyter notebook (%run), ipykernel adds extra argv (e.g. ``-f`` …); those are ignored.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REQUIRED_ENV = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "AWS_S3_BUCKET",
    "AWS_S3_ENDPOINT",
)


def load_config() -> dict:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print("Missing required environment variables:", ", ".join(missing), file=sys.stderr)
        sys.exit(1)
    return {
        "access_key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "region": os.environ["AWS_DEFAULT_REGION"],
        "bucket": os.environ["AWS_S3_BUCKET"],
        "endpoint": os.environ["AWS_S3_ENDPOINT"],
    }


def make_s3_client(cfg: dict):
    kwargs = {
        "endpoint_url": cfg["endpoint"],
        "region_name": cfg["region"],
        "aws_access_key_id": cfg["access_key"],
        "aws_secret_access_key": cfg["secret_key"],
    }
    token = os.environ.get("AWS_SESSION_TOKEN")
    if token:
        kwargs["aws_session_token"] = token
    return boto3.client("s3", **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="S3 endpoint + bucket upload/download shakeout")
    parser.add_argument(
        "--prefix",
        default="shakeout/",
        help="Object key prefix (default: shakeout/)",
    )
    # parse_known_args: Jupyter/ipykernel passes ``-f /path/to/kernel-….json`` on sys.argv
    args, _ = parser.parse_known_args()

    prefix = args.prefix
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    cfg = load_config()
    client = make_s3_client(cfg)
    run_id = str(uuid.uuid4())
    payload = f"s3-shakeout {run_id}\n".encode("utf-8")
    key = f"{prefix}{run_id}.txt"

    local_upload: Path | None = None
    local_download: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".upload.txt") as f:
            f.write(payload)
            local_upload = Path(f.name)

        print(f"Local file: {local_upload} ({len(payload)} bytes)")
        client.upload_file(str(local_upload), cfg["bucket"], key)
        print(f"Uploaded: s3://{cfg['bucket']}/{key}")

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".download.txt") as f:
            pass
        local_download = Path(f.name)

        client.download_file(cfg["bucket"], key, str(local_download))
        got = local_download.read_bytes()
        if got != payload:
            print("ERROR: downloaded bytes do not match upload.", file=sys.stderr)
            sys.exit(1)
        print("Download OK: content matches.")

        client.delete_object(Bucket=cfg["bucket"], Key=key)
        print(f"Deleted object: s3://{cfg['bucket']}/{key}")
        print("Shakeout succeeded.")
    except ClientError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if local_upload is not None:
            local_upload.unlink(missing_ok=True)
        if local_download is not None:
            local_download.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
