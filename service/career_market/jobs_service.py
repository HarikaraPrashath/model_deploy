from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import os
import sys
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from sqlalchemy import select

from lib.database.db import SessionLocal
from lib.database.models import JobMetadata, LastQuery, TrendSnapshot
from service.career_market.utils.auth_utils import _require_user
from service.career_market.utils.config import PIPELINE_PATH, SCRAPER_PATH, SCR_OUTPUT_DIR
from service.career_market.utils.io_utils import _read_json
from service.career_market.utils.jobs_utils import (
    _cleanup_scr_output_dir,
    _python_run,
    _safe_filename,
    _should_refresh,
    _sync_jobs_from_files,
)
from service.career_market.utils.profile_utils import _load_profile_for_email
from service.career_market.utils.storage_utils import _upload_job_json
from service.career_market.utils.trends_utils import _record_trend_snapshot


def get_jobs_service() -> JSONResponse:
    print("[jobs] fetch metadata")
    with SessionLocal() as db:
        rows = db.execute(select(JobMetadata).order_by(JobMetadata.created_at.desc())).scalars().all()
    jobs = [
        {
            "ref": row.ref or "",
            "position": row.position or "",
            "employer": row.employer or "",
            "url": row.url or "",
            "type": row.ad_type,
            "files": row.files or [],
            "textSnippet": row.text_snippet or "",
            "imageFile": row.image_file,
        }
        for row in rows
    ]
    return JSONResponse({"jobs": jobs})


def get_job_file_service(name: str) -> FileResponse:
    if name.startswith("http://") or name.startswith("https://"):
        return RedirectResponse(name)
    safe_name = _safe_filename(name)
    file_path = SCR_OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path)


def refresh_jobs_service(request: Request, payload: dict[str, Any] | None = None) -> JSONResponse:
    user = _require_user(request)
    payload = payload or {}
    profile = _load_profile_for_email(str(user.get("email", "")))
    keyword = str(
        payload.get("keyword")
        or profile.get("basics", {}).get("position")
        or "software engineer"
    )
    user_skills = payload.get("userSkills")
    if not isinstance(user_skills, list):
        user_skills = profile.get("skills", [])
    user_skills = [str(skill).strip() for skill in user_skills if str(skill).strip()]
    force = bool(payload.get("force"))
    enable_ocr = bool(payload.get("enableOcr"))
    if not enable_ocr:
        enable_ocr = str(os.environ.get("ENABLE_JOB_OCR", "1")).lower() in ("1", "true", "yes")

    try:
        print(f"[jobs/refresh] keyword='{keyword}' skills={len(user_skills)} force={force} ocr={enable_ocr}")
        if not force:
            with SessionLocal() as db:
                latest = db.execute(
                    select(TrendSnapshot).order_by(TrendSnapshot.ran_at.desc())
                ).scalars().first()
            if latest and latest.ran_at:
                try:
                    local_tz = ZoneInfo("Asia/Colombo")
                except Exception:
                    local_tz = timezone.utc
                last_run = latest.ran_at
                if last_run.tzinfo is None:
                    last_run_local = last_run.replace(tzinfo=local_tz)
                else:
                    last_run_local = last_run.astimezone(local_tz)
                today = datetime.now(tz=local_tz).date()
                if last_run_local.date() == today:
                    print("[jobs/refresh] skipping scraper (already ran today)")
                    return JSONResponse(
                        {
                            "ok": True,
                            "refreshed": False,
                            "reason": "daily_cache",
                            "lastRun": last_run_local.isoformat(),
                        }
                    )
        if _should_refresh(keyword, force):
            print("[jobs/refresh] running scraper + pipeline")
            env = os.environ.copy()
            env["TOPJOBS_KEYWORD"] = keyword
            _python_run([sys.executable, str(SCRAPER_PATH)], env=env)
            _python_run(
                [
                    sys.executable,
                    str(PIPELINE_PATH),
                    "--scraped_folder",
                    str(SCR_OUTPUT_DIR),
                    "--user_skills",
                    ",".join(user_skills),
                    "--out_json",
                    "ranked_jobs.json",
                ]
                + (["--enable_ocr"] if enable_ocr else []),
                env=os.environ.copy(),
            )
            metadata_path = SCR_OUTPUT_DIR / "metadata.json"
            ranked_path = SCR_OUTPUT_DIR / "ranked_jobs.json"
            metadata = _read_json(metadata_path, []) if metadata_path.exists() else None
            ranked = _read_json(ranked_path, []) if ranked_path.exists() else None
            now = datetime.now(tz=timezone.utc)
            with SessionLocal() as db:
                row = db.execute(select(LastQuery).where(LastQuery.id == 1)).scalar_one_or_none()
                if row:
                    row.keyword = keyword
                    row.ran_at = now
                else:
                    db.add(LastQuery(id=1, keyword=keyword, ran_at=now))
                db.commit()
            _sync_jobs_from_files(
                metadata if isinstance(metadata, list) else None,
                ranked if isinstance(ranked, list) else None,
            )
            if isinstance(metadata, list):
                _upload_job_json("metadata.json", metadata)
            if isinstance(ranked, list):
                _upload_job_json("ranked_jobs.json", ranked)
            ranked_research_path = SCR_OUTPUT_DIR / "ranked_jobs.research.json"
            if ranked_research_path.exists():
                ranked_research = _read_json(ranked_research_path, [])
                if isinstance(ranked_research, list):
                    _upload_job_json("ranked_jobs.research.json", ranked_research)
            _cleanup_scr_output_dir()
            try:
                _record_trend_snapshot(
                    keyword,
                    metadata if isinstance(metadata, list) else None,
                    ranked if isinstance(ranked, list) else None,
                )
            except Exception:
                pass
            print("[jobs/refresh] refresh complete")
            return JSONResponse({"ok": True, "refreshed": True})

        print("[jobs/refresh] using cached results")
        return JSONResponse({"ok": True, "refreshed": False})
    except Exception as exc:
        print(f"[jobs/refresh] failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Job refresh failed: {exc}") from exc
