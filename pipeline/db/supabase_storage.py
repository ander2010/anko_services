from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)


def _storage_settings() -> Dict[str, str]:
    settings = {
        "endpoint_url": os.getenv("SUPABASE_S3_ENDPOINT", ""),
        "region": os.getenv("SUPABASE_S3_REGION", ""),
        "bucket": os.getenv("SUPABASE_S3_BUCKET", ""),
        "access_key": os.getenv("SUPABASE_S3_ACCESS_KEY", ""),
        "secret_key": os.getenv("SUPABASE_S3_SECRET_KEY", ""),
    }
    missing = [key for key, value in settings.items() if not value]
    if missing:
        raise RuntimeError(f"Missing Supabase storage settings: {', '.join(missing)}")
    return settings


def get_bucket_name() -> str:
    return _storage_settings()["bucket"]


@lru_cache(maxsize=1)
def _client():
    cfg = _storage_settings()
    return boto3.client(
        "s3",
        endpoint_url=cfg["endpoint_url"],
        region_name=cfg["region"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
        config=Config(signature_version="s3v4"),
    )


def object_exists(key: str) -> bool:
    """Check if an object exists in the configured Supabase bucket."""
    cfg = _storage_settings()
    client = _client()
    try:
        client.head_object(Bucket=cfg["bucket"], Key=key)
        return True
    except ClientError as exc:  # pragma: no cover - requires network
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 404:
            return False
        logger.exception("Unexpected error checking object '%s' in bucket '%s'", key, cfg["bucket"])
        raise


def get_object_metadata(key: str) -> Dict[str, Optional[Union[str, int, float]]]:
    """Return metadata for a Supabase object (e.g., ContentLength, ContentType)."""
    cfg = _storage_settings()
    client = _client()
    try:
        response = client.head_object(Bucket=cfg["bucket"], Key=key)
        # Normalize a few commonly used fields; keep raw response for flexibility.
        return {
            "ContentLength": response.get("ContentLength"),
            "ContentType": response.get("ContentType"),
            "ETag": response.get("ETag"),
            "LastModified": response.get("LastModified"),
            "Raw": response,
        }
    except ClientError as exc:  # pragma: no cover - requires network
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 404:
            raise FileNotFoundError(f"Object not found: {key}") from exc
        logger.exception("Unexpected error fetching metadata for '%s' in bucket '%s'", key, cfg["bucket"])
        raise


def download_object(key: str, destination: Path) -> Path:
    """Download an object into the given destination path."""
    cfg = _storage_settings()
    client = _client()
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from bucket %s to %s", key, cfg["bucket"], destination)
    client.download_file(cfg["bucket"], key, str(destination))  # pragma: no cover - requires network
    return destination


def download_object_bytes(key: str, *, max_bytes: Optional[int] = None) -> bytes:
    """Download an object into memory and return raw bytes.

    If max_bytes is provided, enforce it using both reported ContentLength and the actual read size.
    """
    cfg = _storage_settings()
    client = _client()
    try:
        response = client.get_object(Bucket=cfg["bucket"], Key=key)
    except ClientError as exc:  # pragma: no cover - requires network
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 404:
            raise FileNotFoundError(f"Object not found: {key}") from exc
        logger.exception("Unexpected error downloading '%s' from bucket '%s'", key, cfg["bucket"])
        raise

    content_length = response.get("ContentLength")
    if max_bytes is not None and content_length and content_length > max_bytes:
        raise ValueError(f"Object '{key}' is {content_length} bytes which exceeds limit of {max_bytes} bytes")

    body = response.get("Body")
    if body is None:
        raise RuntimeError(f"Missing response body for '{key}'")

    data: bytes = body.read()  # pragma: no cover - requires network
    if max_bytes is not None and len(data) > max_bytes:
        raise ValueError(f"Downloaded data for '{key}' exceeds limit of {max_bytes} bytes")
    return data


def upload_object(source: Path, key: str) -> str:
    """Upload a local file to Supabase storage at the given key."""
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")

    cfg = _storage_settings()
    client = _client()
    try:
        client.upload_file(str(source), cfg["bucket"], key)  # pragma: no cover - requires network
    except ClientError as exc:  # pragma: no cover - requires network
        logger.exception("Failed to upload '%s' to bucket '%s' as '%s'", source, cfg["bucket"], key)
        raise
    return key


def list_objects(prefix: str | None = None) -> list[str]:
    """List all object keys under an optional prefix."""
    cfg = _storage_settings()
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=cfg["bucket"], Prefix=prefix or ""):  # pragma: no cover - requires network
        for item in page.get("Contents", []):
            if "Key" in item:
                keys.append(item["Key"])
    return keys
