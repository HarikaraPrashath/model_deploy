from __future__ import annotations

from typing import Any

import asyncio
import json

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from datetime import datetime, timezone

from service.career_market import storage_service
from service.career_market.utils.auth_utils import _require_user
from service.career_market.utils.config import ANALYSIS_OUTPUT_DIR, CAREER_MARKET_DIR, SCR_OUTPUT_DIR
from service.career_market.utils.profile_utils import _build_student_profile, _load_profile_for_email
from service.career_market.utils.storage_utils import _storage_enabled
from service.career_market.utils.storage_utils import _download_job_json
from service.career_market.utils.io_utils import _read_json


async def analyse_service(request: Request, payload: dict[str, Any] | None = None) -> JSONResponse:
    user = _require_user(request)
    payload = payload or {}
    keyword = str(payload.get("keyword", "")).strip()

    try:
        from service.career_market.job_analyse.Job_Analysis_and_Skill_Gap import (
            run_analysis_from_metadata,
            STUDENT_PROFILE,
        )  # type: ignore
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Analysis pipeline not available. Check server dependencies.",
        ) from exc

    student_profile: dict[str, Any] = STUDENT_PROFILE
    profile_source: dict[str, Any] | None = None
    override = payload.get("profile")
    if isinstance(override, dict):
        profile_source = override
        student_profile = _build_student_profile(profile_source, defaults=STUDENT_PROFILE)
    else:
        profile_source = _load_profile_for_email(str(user.get("email", "")))
        student_profile = _build_student_profile(profile_source, defaults=STUDENT_PROFILE)

    if not keyword:
        basics = profile_source.get("basics") if isinstance(profile_source, dict) else {}
        keyword = str(basics.get("position", "")).strip()
    if not keyword:
        raise HTTPException(status_code=400, detail="Keyword is required.")

    now = datetime.now(tz=timezone.utc)
    date_key = now.strftime("%Y%m%d")
    run_folder = now.strftime("%Y%m%d_%H%M%S")
    use_storage = _storage_enabled()
    daily_prefix = f"analysis/daily/{date_key}"
    storage_prefix = daily_prefix if use_storage else None
    if use_storage:
        output_dir = "topjobs_ads"
    else:
        output_dir = ANALYSIS_OUTPUT_DIR / run_folder
        output_dir.mkdir(parents=True, exist_ok=True)

    local_base = CAREER_MARKET_DIR / "analysis" / "daily" / date_key
    local_base.mkdir(parents=True, exist_ok=True)
    local_cache_path = local_base / "analysis.json"

    if use_storage and not payload.get("force"):
        cache_path = f"{daily_prefix}/analysis.json"
        if storage_service.storage_object_exists(cache_path):
            cached = storage_service.download_json_from_storage(cache_path)
            if cached is not None:
                try:
                    local_cache_path.write_text(
                        json.dumps(cached, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
            warning = f"Using cached analysis for {now.strftime('%Y-%m-%d')}. Set force=true to re-scrape."
            return JSONResponse(
                {
                    "ok": True,
                    "cached": True,
                    "warning": warning,
                    "storage": {"prefix": daily_prefix},
                    "localPath": str(local_cache_path),
                    "result": cached,
                }
            )

    metadata: list[dict[str, Any]] | None = None
    write_local = not use_storage
    analysis_folder = str(output_dir)

    local_metadata = _read_json(SCR_OUTPUT_DIR / "metadata.json", None)
    if isinstance(local_metadata, list) and local_metadata:
        metadata = local_metadata
        analysis_folder = str(SCR_OUTPUT_DIR)
        write_local = True
    elif use_storage:
        metadata = _download_job_json("metadata.json")
        if isinstance(metadata, list) and metadata:
            write_local = False

    if not isinstance(metadata, list) or not metadata:
        raise HTTPException(
            status_code=404,
            detail="No cached job metadata found. Run /jobs/refresh first.",
        )

    try:
        result = await asyncio.to_thread(
            run_analysis_from_metadata,
            metadata,
            student_profile,
            analysis_folder,
            False,
            write_local,
            storage_prefix,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if use_storage:
        storage_service.upload_json_to_storage(result, f"{daily_prefix}/analysis.json")
    try:
        local_cache_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    return JSONResponse(result)
