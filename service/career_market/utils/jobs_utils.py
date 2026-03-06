from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select, text as sql_text

from lib.database.db import SessionLocal
from lib.database.models import JobMetadata, LastQuery, RankedJob
from service.career_market.utils.config import CV_EXTRACTOR_DIR, SCR_OUTPUT_DIR
from service.career_market.utils.io_utils import _read_json
from service.career_market.utils.storage_utils import _download_job_json, _storage_enabled, _upload_to_storage


def _safe_filename(name: str) -> str:
    return Path(name).name


def _python_run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Run a python command and stream output for visibility."""
    process = subprocess.Popen(
        cmd,
        cwd=str(CV_EXTRACTOR_DIR),
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    if not process.stdout:
        raise RuntimeError("Failed to start process")
    for line in process.stdout:
        line = line.rstrip()
        if line:
            print(f"[jobs/refresh] {line}")
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"Command failed with exit code {returncode}")


def _load_jobs_payload() -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    if _storage_enabled():
        metadata = _download_job_json("metadata.json")
        ranked = _download_job_json("ranked_jobs.json")
        if isinstance(metadata, list) and isinstance(ranked, list):
            return metadata, ranked
        return None
    metadata_path = SCR_OUTPUT_DIR / "metadata.json"
    ranked_path = SCR_OUTPUT_DIR / "ranked_jobs.json"
    if not metadata_path.exists() or not ranked_path.exists():
        return None
    metadata = _read_json(metadata_path, [])
    ranked = _read_json(ranked_path, [])
    if not isinstance(metadata, list) or not isinstance(ranked, list):
        return None
    return metadata, ranked


def _cleanup_scr_output_dir() -> None:
    if not SCR_OUTPUT_DIR.exists():
        return
    for item in SCR_OUTPUT_DIR.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink()
        except Exception:
            continue


def _sync_jobs_from_files(
    metadata: list[dict[str, Any]] | None = None,
    ranked: list[dict[str, Any]] | None = None,
) -> None:
    if metadata is None or ranked is None:
        payload = _load_jobs_payload()
        if not payload:
            return
        metadata, ranked = payload
    if not isinstance(metadata, list) or not isinstance(ranked, list):
        return

    now = datetime.now(tz=timezone.utc)
    with SessionLocal() as db:
        db.execute(sql_text("SET LOCAL statement_timeout = '300s'"))
        db.execute(sql_text("TRUNCATE TABLE job_metadata"))
        db.execute(sql_text("TRUNCATE TABLE ranked_jobs"))
        db.commit()

        batch = 0
        for job in metadata:
            if not isinstance(job, dict):
                continue
            files = job.get("files") if isinstance(job.get("files"), list) else []
            text_file = next((f for f in files if str(f).lower().endswith(".txt")), None)
            image_file = next(
                (f for f in files if str(f).lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))),
                None,
            )
            image_url = None
            if image_file and _storage_enabled():
                if isinstance(image_file, str) and image_file.startswith("http"):
                    image_url = image_file
                else:
                    local_img = SCR_OUTPUT_DIR / _safe_filename(str(image_file))
                    image_url = _upload_to_storage(local_img, f"jobs/{local_img.name}")
            snippet = ""
            if text_file:
                try:
                    text = (SCR_OUTPUT_DIR / _safe_filename(text_file)).read_text(
                        encoding="utf-8", errors="ignore"
                    )
                    text = " ".join(text.split())
                    snippet = text[:300] + ("..." if len(text) > 300 else "")
                except Exception:
                    snippet = ""

            db.add(
                JobMetadata(
                    ref=job.get("ref"),
                    position=job.get("position"),
                    employer=job.get("employer"),
                    url=job.get("url"),
                    ad_type=job.get("type"),
                    files=files,
                    text_snippet=snippet,
                    image_file=image_url or image_file,
                    created_at=now,
                )
            )
            batch += 1
            if batch % 25 == 0:
                db.commit()

        if batch % 25 != 0:
            db.commit()

        batch = 0
        for job in ranked:
            if not isinstance(job, dict):
                continue
            db.add(
                RankedJob(
                    ref=job.get("ref"),
                    position=job.get("position"),
                    employer=job.get("employer"),
                    url=job.get("url"),
                    skills_found=job.get("skills_found", []) or [],
                    overlap=job.get("overlap", []) or [],
                    missing=job.get("missing", []) or [],
                    match_percent=job.get("match_percent"),
                    baseline_match_percent=job.get("baseline_match_percent"),
                    job_skill_count=job.get("job_skill_count"),
                    user_skill_count=job.get("user_skill_count"),
                    text_excerpt=job.get("text_excerpt"),
                    text_full=job.get("text_full"),
                    must_have_skills=job.get("must_have_skills", []) or [],
                    nice_to_have_skills=job.get("nice_to_have_skills", []) or [],
                    core_skills=job.get("core_skills", []) or [],
                    matched_must_have=job.get("matched_must_have", []) or [],
                    missing_must_have=job.get("missing_must_have", []) or [],
                    must_have_gate_pass=job.get("must_have_gate_pass"),
                    matched_nice_to_have=job.get("matched_nice_to_have", []) or [],
                    weighted_components=job.get("weighted_components", {}) or {},
                    explanations=job.get("explanations", []) or [],
                    created_at=now,
                )
            )
            batch += 1
            if batch % 25 == 0:
                db.commit()

        if batch % 25 != 0:
            db.commit()


def _has_jobs_in_db() -> bool:
    with SessionLocal() as db:
        meta_row = db.execute(select(JobMetadata.id)).scalars().first()
        ranked_row = db.execute(select(RankedJob.id)).scalars().first()
    return bool(meta_row and ranked_row)


def _ensure_jobs_cached() -> bool:
    if _has_jobs_in_db():
        return True
    payload = _load_jobs_payload()
    if not payload:
        return False
    metadata, ranked = payload
    _sync_jobs_from_files(metadata, ranked)
    return True


def _should_refresh(keyword: str, force: bool) -> bool:
    if force:
        return True
    if not _ensure_jobs_cached():
        return True
    last_keyword = ""
    last_run = datetime.fromtimestamp(0, tz=timezone.utc)
    with SessionLocal() as db:
        row = db.execute(select(LastQuery).where(LastQuery.id == 1)).scalar_one_or_none()
        if row:
            last_keyword = str(row.keyword or "").strip().lower()
            if isinstance(row.ran_at, datetime):
                last_run = row.ran_at
                if last_run.tzinfo is None:
                    last_run = last_run.replace(tzinfo=timezone.utc)

    age = datetime.now(tz=timezone.utc) - last_run
    if last_keyword != keyword.strip().lower():
        return True
    if age > timedelta(hours=3):
        return True
    return False
