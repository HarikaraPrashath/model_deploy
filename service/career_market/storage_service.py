from __future__ import annotations

import mimetypes
import json
import os
from pathlib import Path
from typing import Any, Optional

import requests


def storage_enabled() -> bool:
    return bool(
        os.environ.get("SUPABASE_URL", "").strip()
        and os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        and os.environ.get("SUPABASE_STORAGE_BUCKET", "").strip()
    )


def storage_object_url(remote_path: str) -> str:
    url = os.environ.get("SUPABASE_URL", "").strip()
    bucket = os.environ.get("SUPABASE_STORAGE_BUCKET", "").strip()
    return f"{url}/storage/v1/object/{bucket}/{remote_path}"


def storage_public_url(remote_path: str) -> str:
    url = os.environ.get("SUPABASE_URL", "").strip()
    bucket = os.environ.get("SUPABASE_STORAGE_BUCKET", "").strip()
    return f"{url}/storage/v1/object/public/{bucket}/{remote_path}"


def storage_object_exists(remote_path: str) -> bool:
    if not storage_enabled():
        return False
    url = storage_public_url(remote_path)
    try:
        resp = requests.head(url, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def upload_bytes_to_storage(content: bytes, remote_path: str, content_type: str) -> Optional[str]:
    if not storage_enabled():
        return None
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    try:
        resp = requests.post(storage_object_url(remote_path), data=content, headers=headers, timeout=30)
        if resp.status_code in (200, 201):
            return storage_public_url(remote_path)
        return None
    except Exception:
        return None


def upload_file_to_storage(local_path: Path, remote_path: str) -> Optional[str]:
    if not storage_enabled() or not local_path.exists():
        return None
    if storage_object_exists(remote_path):
        return storage_public_url(remote_path)
    content_type, _ = mimetypes.guess_type(local_path.name)
    content_type = content_type or "application/octet-stream"
    try:
        return upload_bytes_to_storage(local_path.read_bytes(), remote_path, content_type)
    except Exception:
        return None


def upload_json_to_storage(data: Any, remote_path: str) -> Optional[str]:
    try:
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    except Exception:
        return None
    return upload_bytes_to_storage(payload, remote_path, "application/json")


def download_json_from_storage(remote_path: str) -> Any | None:
    if not storage_enabled():
        return None
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
    }
    try:
        resp = requests.get(storage_object_url(remote_path), headers=headers, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None
