from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

import requests

from service.career_market.utils.config import (
    JOB_STORAGE_PREFIX,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_STORAGE_BUCKET,
    SUPABASE_URL,
)


def _storage_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and SUPABASE_STORAGE_BUCKET)


def _storage_public_url(path: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_STORAGE_BUCKET}/{path}"


def _storage_object_exists(remote_path: str) -> bool:
    if not _storage_enabled():
        return False
    url = _storage_public_url(remote_path)
    try:
        resp = requests.head(url, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def _upload_to_storage(local_path: Path, remote_path: str) -> str | None:
    if not _storage_enabled():
        return None
    if not local_path.exists():
        return None
    if _storage_object_exists(remote_path):
        return _storage_public_url(remote_path)
    content_type, _ = mimetypes.guess_type(local_path.name)
    content_type = content_type or "application/octet-stream"
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_STORAGE_BUCKET}/{remote_path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    try:
        with local_path.open("rb") as f:
            resp = requests.post(url, data=f, headers=headers, timeout=30)
        if resp.status_code in (200, 201):
            return _storage_public_url(remote_path)
        return None
    except Exception:
        return None


def _storage_object_url(remote_path: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_STORAGE_BUCKET}/{remote_path}"


def _upload_bytes_to_storage(content: bytes, remote_path: str, content_type: str) -> str | None:
    if not _storage_enabled():
        return None
    url = _storage_object_url(remote_path)
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    try:
        resp = requests.post(url, data=content, headers=headers, timeout=30)
        if resp.status_code in (200, 201):
            return _storage_public_url(remote_path)
        return None
    except Exception:
        return None


def _job_storage_path(name: str) -> str:
    return f"{JOB_STORAGE_PREFIX}/{name}"


def _upload_job_json(name: str, data: Any) -> str | None:
    try:
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    except Exception:
        return None
    return _upload_bytes_to_storage(payload, _job_storage_path(name), "application/json")


def _download_job_json(name: str) -> Any | None:
    if not _storage_enabled():
        return None
    url = _storage_object_url(_job_storage_path(name))
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None
