from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy.dialects.postgresql import insert

from lib.database.db import SessionLocal
from lib.database.models import (
    User,
    Profile,
    CvFile,
    LastQuery,
    JobMetadata,
    RankedJob,
    TrendSnapshot,
)


BASE_DIR = Path(__file__).resolve().parents[2]
CAREER_MARKET_DIR = BASE_DIR / "service" / "career_market"
STORAGE_DIR = CAREER_MARKET_DIR / "storage"
SCR_OUTPUT_DIR = CAREER_MARKET_DIR / "scr_output" / "topjobs_ads"

USERS_PATH = STORAGE_DIR / "users.json"
CV_INDEX_PATH = STORAGE_DIR / "cv_index.json"
PROFILE_PATH = STORAGE_DIR / "profile.json"
PROFILES_DIR = STORAGE_DIR / "profiles"
LAST_QUERY_PATH = STORAGE_DIR / "last_query.json"
SCR_LAST_QUERY_PATH = SCR_OUTPUT_DIR / "last_query.json"
TREND_HISTORY_PATH = STORAGE_DIR / "trends_history.json"
METADATA_PATH = SCR_OUTPUT_DIR / "metadata.json"
RANKED_PATH = SCR_OUTPUT_DIR / "ranked_jobs.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _parse_iso(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            pass
    return datetime.now(tz=timezone.utc)


def _safe_filename(name: str) -> str:
    return Path(name).name


def _text_snippet(files: list[str]) -> tuple[str | None, str | None]:
    text_file = next((f for f in files if str(f).lower().endswith(".txt")), None)
    image_file = next(
        (f for f in files if str(f).lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))),
        None,
    )
    if not text_file:
        return None, image_file
    try:
        text = (SCR_OUTPUT_DIR / _safe_filename(text_file)).read_text(
            encoding="utf-8", errors="ignore"
        )
        text = " ".join(text.split())
        snippet = text[:300] + ("..." if len(text) > 300 else "")
        return snippet, image_file
    except Exception:
        return None, image_file


def _iter_profiles() -> Iterable[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for path in PROFILES_DIR.glob("*.json"):
        data = _read_json(path, {})
        if isinstance(data, dict):
            profiles.append(data)
    legacy = _read_json(PROFILE_PATH, {})
    if isinstance(legacy, dict) and legacy:
        profiles.append(legacy)
    return profiles


def _upsert_users(db, users: list[dict[str, Any]]) -> None:
    if not users:
        return
    stmt = insert(User).values(
        [
            {
                "email": str(u.get("email", "")).strip().lower(),
                "password_salt": str(u.get("passwordSalt", "")),
                "password_hash": str(u.get("passwordHash", "")),
                "token": u.get("token"),
                "created_at": _parse_iso(u.get("createdAt")),
            }
            for u in users
            if str(u.get("email", "")).strip()
        ]
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[User.email],
        set_={
            "password_salt": stmt.excluded.password_salt,
            "password_hash": stmt.excluded.password_hash,
            "token": stmt.excluded.token,
        },
    )
    db.execute(stmt)


def _upsert_profiles(db, profiles: Iterable[dict[str, Any]]) -> None:
    rows = []
    seen_emails: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        email = str(profile.get("basics", {}).get("contactEmail", "")).strip().lower()
        if not email:
            continue
        if email in seen_emails:
            continue
        seen_emails.add(email)
        rows.append(
            {
                "email": email,
                "profile_json": profile,
                "updated_at": datetime.now(tz=timezone.utc),
            }
        )
    if not rows:
        return
    stmt = insert(Profile).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Profile.email],
        set_={
            "profile_json": stmt.excluded.profile_json,
            "updated_at": stmt.excluded.updated_at,
        },
    )
    db.execute(stmt)


def _upsert_cv_files(db, entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cv_id = str(entry.get("id", "")).strip()
        if not cv_id:
            continue
        rows.append(
            {
                "id": cv_id,
                "path": str(entry.get("path", "")),
                "original_name": str(entry.get("originalName", "")),
                "size": int(entry.get("size", 0) or 0),
                "content_type": str(entry.get("contentType", "")) or "application/octet-stream",
                "uploaded_at": _parse_iso(entry.get("uploadedAt")),
            }
        )
    if not rows:
        return
    stmt = insert(CvFile).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[CvFile.id],
        set_={
            "path": stmt.excluded.path,
            "original_name": stmt.excluded.original_name,
            "size": stmt.excluded.size,
            "content_type": stmt.excluded.content_type,
            "uploaded_at": stmt.excluded.uploaded_at,
        },
    )
    db.execute(stmt)


def _upsert_last_query(db, payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or not payload.get("keyword"):
        return
    stmt = insert(LastQuery).values(
        {"id": 1, "keyword": str(payload.get("keyword", "")), "ran_at": _parse_iso(payload.get("ranAt"))}
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[LastQuery.id],
        set_={"keyword": stmt.excluded.keyword, "ran_at": stmt.excluded.ran_at},
    )
    db.execute(stmt)


def _upsert_trends(db, history: list[dict[str, Any]]) -> None:
    rows = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        ran_at = _parse_iso(entry.get("ranAt"))
        keyword = str(entry.get("keyword", "")).strip()
        if not keyword:
            continue
        rows.append(
            {
                "ran_at": ran_at,
                "keyword": keyword,
                "job_count": int(entry.get("jobCount", 0) or 0),
                "skill_counts": entry.get("skillCounts", {}) or {},
                "role_counts": entry.get("roleCounts", {}) or {},
            }
        )
    if not rows:
        return
    stmt = insert(TrendSnapshot).values(rows)
    stmt = stmt.on_conflict_do_nothing()
    db.execute(stmt)


def _upsert_jobs(db, metadata: list[dict[str, Any]]) -> None:
    rows = []
    for job in metadata:
        if not isinstance(job, dict):
            continue
        files = job.get("files") if isinstance(job.get("files"), list) else []
        snippet, image = _text_snippet(files)
        rows.append(
            {
                "ref": job.get("ref"),
                "position": job.get("position"),
                "employer": job.get("employer"),
                "url": job.get("url"),
                "ad_type": job.get("type"),
                "files": files,
                "text_snippet": snippet,
                "image_file": image,
                "created_at": datetime.now(tz=timezone.utc),
            }
        )
    if not rows:
        return
    stmt = insert(JobMetadata).values(rows)
    stmt = stmt.on_conflict_do_nothing()
    db.execute(stmt)


def _upsert_ranked(db, ranked: list[dict[str, Any]]) -> None:
    rows = []
    for job in ranked:
        if not isinstance(job, dict):
            continue
        rows.append(
            {
                "ref": job.get("ref"),
                "position": job.get("position"),
                "employer": job.get("employer"),
                "url": job.get("url"),
                "skills_found": job.get("skills_found", []) or [],
                "overlap": job.get("overlap", []) or [],
                "missing": job.get("missing", []) or [],
                "match_percent": job.get("match_percent"),
                "baseline_match_percent": job.get("baseline_match_percent"),
                "job_skill_count": job.get("job_skill_count"),
                "user_skill_count": job.get("user_skill_count"),
                "text_excerpt": job.get("text_excerpt"),
                "text_full": job.get("text_full"),
                "must_have_skills": job.get("must_have_skills", []) or [],
                "nice_to_have_skills": job.get("nice_to_have_skills", []) or [],
                "core_skills": job.get("core_skills", []) or [],
                "matched_must_have": job.get("matched_must_have", []) or [],
                "missing_must_have": job.get("missing_must_have", []) or [],
                "must_have_gate_pass": job.get("must_have_gate_pass"),
                "matched_nice_to_have": job.get("matched_nice_to_have", []) or [],
                "weighted_components": job.get("weighted_components", {}) or {},
                "explanations": job.get("explanations", []) or [],
                "created_at": datetime.now(tz=timezone.utc),
            }
        )
    if not rows:
        return
    stmt = insert(RankedJob).values(rows)
    stmt = stmt.on_conflict_do_nothing()
    db.execute(stmt)


def main() -> None:
    users = _read_json(USERS_PATH, [])
    cv_index = _read_json(CV_INDEX_PATH, [])
    profiles = list(_iter_profiles())
    last_query = _read_json(LAST_QUERY_PATH, {})
    if not last_query:
        last_query = _read_json(SCR_LAST_QUERY_PATH, {})
    trends = _read_json(TREND_HISTORY_PATH, [])
    metadata = _read_json(METADATA_PATH, [])
    ranked = _read_json(RANKED_PATH, [])

    with SessionLocal() as db:
        _upsert_users(db, users if isinstance(users, list) else [])
        _upsert_profiles(db, profiles)
        _upsert_cv_files(db, cv_index if isinstance(cv_index, list) else [])
        _upsert_last_query(db, last_query if isinstance(last_query, dict) else {})
        _upsert_trends(db, trends if isinstance(trends, list) else [])
        _upsert_jobs(db, metadata if isinstance(metadata, list) else [])
        _upsert_ranked(db, ranked if isinstance(ranked, list) else [])
        db.commit()

    print("Import complete.")


if __name__ == "__main__":
    main()

