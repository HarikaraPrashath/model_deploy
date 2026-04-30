from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select, text as sql_text

from lib.database.db import SessionLocal
from lib.database.models import JobMetadata, LastQuery, RankedJob
from service.career_market.utils.config import CV_EXTRACTOR_DIR, SCR_OUTPUT_DIR
from service.career_market.utils.io_utils import _read_json
from service.career_market.utils.role_match_utils import infer_role_tags
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
    source_keyword: str = "",
) -> None:
    if metadata is None or ranked is None:
        payload = _load_jobs_payload()
        if not payload:
            return
        metadata, ranked = payload
    if not isinstance(metadata, list) or not isinstance(ranked, list):
        return

    ranked_by_ref: dict[str, dict[str, Any]] = {}
    ranked_by_url: dict[str, dict[str, Any]] = {}
    ranked_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for item in ranked:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("ref") or "").strip()
        url = str(item.get("url") or "").strip()
        position = str(item.get("position") or "").strip().lower()
        employer = str(item.get("employer") or "").strip().lower()
        if ref:
            ranked_by_ref[ref] = item
        if url:
            ranked_by_url[url] = item
        if position or employer:
            ranked_by_identity[(position, employer)] = item

    now = datetime.now(tz=timezone.utc)
    with SessionLocal() as db:
        db.execute(sql_text("SET LOCAL statement_timeout = '300s'"))
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
                    text = ""
            else:
                text = str(job.get("raw_text") or "").strip()

            ref = str(job.get("ref") or "").strip()
            url = str(job.get("url") or "").strip()
            position = str(job.get("position") or "").strip()
            employer = str(job.get("employer") or "").strip()
            ranked_job = (
                ranked_by_ref.get(ref)
                or ranked_by_url.get(url)
                or ranked_by_identity.get((position.lower(), employer.lower()))
                or {}
            )

            text_full = str(ranked_job.get("text_full") or text or "").strip()
            if not snippet and text_full:
                snippet = text_full[:300] + ("..." if len(text_full) > 300 else "")
            skills_found = ranked_job.get("skills_found", []) or []
            must_have_skills = ranked_job.get("must_have_skills", []) or []
            nice_to_have_skills = ranked_job.get("nice_to_have_skills", []) or []
            core_skills = ranked_job.get("core_skills", []) or []
            source_label = str(job.get("source_label") or "").strip()
            role_tags = infer_role_tags(
                position,
                text_full or snippet,
                source_keyword if not source_label else "",
            )

            existing = None
            if url:
                existing = db.execute(select(JobMetadata).where(JobMetadata.url == url)).scalar_one_or_none()
            if existing is None and ref:
                existing = db.execute(
                    select(JobMetadata).where(
                        or_(JobMetadata.ref == ref, JobMetadata.ref == f"legacy:{ref}")
                    )
                ).scalar_one_or_none()

            payload = {
                "ref": ref or job.get("ref"),
                "position": position or job.get("position"),
                "employer": employer or job.get("employer"),
                "url": url or job.get("url"),
                "ad_type": job.get("type"),
                "files": files,
                "text_snippet": snippet,
                "text_full": text_full or None,
                "skills_found": skills_found,
                "must_have_skills": must_have_skills,
                "nice_to_have_skills": nice_to_have_skills,
                "core_skills": core_skills,
                "role_tags": role_tags,
                "source_keyword": source_label or source_keyword or None,
                "scraped_at": now,
                "extraction_metadata": {
                    "skills_found": len(skills_found),
                    "must_have_skills": len(must_have_skills),
                    "nice_to_have_skills": len(nice_to_have_skills),
                    "core_skills": len(core_skills),
                },
                "image_file": image_url or image_file,
                "created_at": existing.created_at if existing else now,
            }

            if existing:
                for field, value in payload.items():
                    setattr(existing, field, value)
            else:
                db.add(JobMetadata(**payload))
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
                    ocr_text=job.get("ocr_text"),
                    regex_skills_found=job.get("regex_skills_found", []) or [],
                    llm_skills_found=job.get("llm_skills_found", []) or [],
                    llm_must_have_skills=job.get("llm_must_have_skills", []) or [],
                    llm_nice_to_have_skills=job.get("llm_nice_to_have_skills", []) or [],
                    extraction_source=job.get("extraction_source"),
                    extraction_metadata=job.get("extraction_metadata", {}) or {},
                    vision_skills_found=job.get("vision_skills_found", []) or [],
                    vision_must_have_skills=job.get("vision_must_have_skills", []) or [],
                    vision_nice_to_have_skills=job.get("vision_nice_to_have_skills", []) or [],
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
    return bool(meta_row)


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
    last_run = datetime.fromtimestamp(0, tz=timezone.utc)
    with SessionLocal() as db:
        row = db.execute(select(LastQuery).where(LastQuery.id == 1)).scalar_one_or_none()
        if row:
            if isinstance(row.ran_at, datetime):
                last_run = row.ran_at
                if last_run.tzinfo is None:
                    last_run = last_run.replace(tzinfo=timezone.utc)

    age = datetime.now(tz=timezone.utc) - last_run
    if age > timedelta(hours=12):
        return True
    return False
