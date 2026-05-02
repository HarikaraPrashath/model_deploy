from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select

from lib.database.db import SessionLocal
from lib.database.models import JobMetadata, RankedJob
from service.career_market.utils.auth_utils import _require_user
from service.career_market.utils.profile_utils import _load_profile_for_email
from service.career_market.utils.role_match_utils import role_matches

try:
    from service.career_market.cv_extractor.skill_config import SKILL_LEXICON
except Exception:
    SKILL_LEXICON = {}


def _build_skill_alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical, values in SKILL_LEXICON.items():
        for value in [canonical] + list(values):
            key = " ".join(str(value).strip().lower().split())
            if key and key not in aliases:
                aliases[key] = canonical
    return aliases


SKILL_ALIAS_MAP = _build_skill_alias_map()


def _canonicalize_skill(skill: str) -> str:
    key = " ".join(skill.strip().lower().split())
    if not key:
        return ""
    return SKILL_ALIAS_MAP.get(key, key)


def _unique_skills(skills: list[str] | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for skill in skills or []:
        canonical = _canonicalize_skill(str(skill))
        if canonical and canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
    return ordered


def _serialize_cached_ranked(rows: list[RankedJob]) -> list[dict[str, Any]]:
    return [
        {
            "ref": row.ref,
            "position": row.position,
            "employer": row.employer,
            "url": row.url,
            "text_excerpt": row.text_excerpt,
            "text_full": row.text_full,
            "skills_found": row.skills_found or [],
            "signals_found": [],
            "match_percent": row.match_percent,
            "baseline_match_percent": row.baseline_match_percent,
            "overlap": row.overlap or [],
            "missing": row.missing or [],
            "job_skill_count": row.job_skill_count,
            "user_skill_count": row.user_skill_count,
            "must_have_skills": row.must_have_skills or [],
            "nice_to_have_skills": row.nice_to_have_skills or [],
            "core_skills": row.core_skills or [],
            "matched_must_have": row.matched_must_have or [],
            "missing_must_have": row.missing_must_have or [],
            "must_have_gate_pass": row.must_have_gate_pass,
            "matched_nice_to_have": row.matched_nice_to_have or [],
            "weighted_components": row.weighted_components or {},
            "explanations": row.explanations or [],
        }
        for row in rows
    ]


def _load_inventory_jobs(role: str | None = None, limit: int | None = None) -> list[JobMetadata]:
    with SessionLocal() as db:
        rows = db.execute(select(JobMetadata).order_by(JobMetadata.scraped_at.desc().nullslast())).scalars().all()
    if role:
        rows = [row for row in rows if role_matches(row.role_tags or [], role)]
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


def _rank_inventory_job(row: JobMetadata, user_skills: list[str], role: str | None = None) -> dict[str, Any]:
    user_skill_list = _unique_skills(user_skills)
    user_skill_set = set(user_skill_list)
    skills_found = _unique_skills((row.skills_found or []) + (row.core_skills or []))
    must_have_skills = _unique_skills(row.must_have_skills or [])
    nice_to_have_skills = _unique_skills(row.nice_to_have_skills or [])
    core_skills = _unique_skills(row.core_skills or skills_found or must_have_skills + nice_to_have_skills)

    overlap = [skill for skill in skills_found if skill in user_skill_set]
    missing = [skill for skill in skills_found if skill not in user_skill_set]
    matched_must_have = [skill for skill in must_have_skills if skill in user_skill_set]
    missing_must_have = [skill for skill in must_have_skills if skill not in user_skill_set]
    matched_nice_to_have = [skill for skill in nice_to_have_skills if skill in user_skill_set]

    # Compute match score with "active weights" so a missing extraction never becomes 100%.
    must_ratio = len(matched_must_have) / len(must_have_skills) if must_have_skills else 0.0
    core_matches = [skill for skill in core_skills if skill in user_skill_set]
    core_ratio = len(core_matches) / len(core_skills) if core_skills else 0.0
    nice_ratio = len(matched_nice_to_have) / len(nice_to_have_skills) if nice_to_have_skills else 0.0
    role_signal = 1.0 if not role or role_matches(row.role_tags or [], role) else 0.0
    baseline_ratio = len(overlap) / len(skills_found) if skills_found else 0.0

    active_weights = {
        "must_have": 0.45 if must_have_skills else 0.0,
        "core": 0.35 if core_skills else 0.0,
        "nice_to_have": 0.15 if nice_to_have_skills else 0.0,
        "role_signal": 0.05,
    }
    total_weight = sum(active_weights.values())
    weighted = (
        (
            active_weights["must_have"] * must_ratio
            + active_weights["core"] * core_ratio
            + active_weights["nice_to_have"] * nice_ratio
            + active_weights["role_signal"] * role_signal
        )
        / total_weight
        if total_weight > 0
        else 0.0
    )

    match_percent = round(weighted * 100.0, 1)

    # Guardrails for low-signal extraction (common when skills aren't extracted yet).
    if len(skills_found) < 3 and len(overlap) < 2:
        match_percent = min(match_percent, 65.0)
    if len(skills_found) < 2:
        match_percent = min(match_percent, 55.0)
    if must_have_skills and missing_must_have:
        match_percent = round(match_percent * 0.55, 1)

    explanations: list[str] = []
    if not skills_found:
        explanations.append("No skills extracted for this job yet (run job reindex).")
    if matched_must_have:
        explanations.append(f"Matched {len(matched_must_have)} must-have skills.")
    if missing_must_have:
        explanations.append(f"Missing {len(missing_must_have)} must-have skills.")
    if matched_nice_to_have:
        explanations.append(f"Matched {len(matched_nice_to_have)} nice-to-have skills.")
    if row.role_tags:
        explanations.append(f"Tagged for roles: {', '.join(row.role_tags[:3])}.")

    return {
        "ref": row.ref,
        "position": row.position,
        "employer": row.employer,
        "url": row.url,
        "text_excerpt": row.text_snippet or (row.text_full or "")[:280],
        "text_full": row.text_full or "",
        "skills_found": skills_found,
        "signals_found": [],
        "match_percent": match_percent,
        "baseline_match_percent": round(baseline_ratio * 100, 1),
        "overlap": overlap,
        "missing": missing,
        "job_skill_count": len(skills_found),
        "user_skill_count": len(user_skill_list),
        "must_have_skills": must_have_skills,
        "nice_to_have_skills": nice_to_have_skills,
        "core_skills": core_skills,
        "matched_must_have": matched_must_have,
        "missing_must_have": missing_must_have,
        "must_have_gate_pass": not missing_must_have,
        "matched_nice_to_have": matched_nice_to_have,
        "weighted_components": {
            "must_have": round(must_ratio * 100, 1),
            "core": round(core_ratio * 100, 1),
            "nice_to_have": round(nice_ratio * 100, 1),
            "role_signal": round(role_signal * 100, 1),
        },
        "explanations": explanations,
        "role_tags": row.role_tags or [],
    }


def _rank_inventory(role: str | None, user_skills: list[str], limit: int | None = None) -> list[dict[str, Any]]:
    rows = _load_inventory_jobs(role=role, limit=limit)
    ranked = [_rank_inventory_job(row, user_skills, role=role) for row in rows]
    ranked.sort(
        key=lambda item: (
            item.get("match_percent", 0),
            len(item.get("overlap", [])),
            -len(item.get("missing_must_have", [])),
        ),
        reverse=True,
    )
    return ranked


def get_ranked_service() -> JSONResponse:
    print("[ranked] fetch ranked list")
    with SessionLocal() as db:
        rows = db.execute(select(RankedJob).order_by(RankedJob.match_percent.desc().nullslast())).scalars().all()
    return JSONResponse({"ranked": _serialize_cached_ranked(rows)})


def search_ranked_service(request: Request, payload: dict[str, Any] | None = None) -> JSONResponse:
    payload = payload or {}
    role = str(payload.get("role", "")).strip() or None
    raw_user_skills = payload.get("userSkills")
    limit_value = payload.get("limit", 200)

    if isinstance(raw_user_skills, list):
        user_skills = [str(skill).strip() for skill in raw_user_skills if str(skill).strip()]
    else:
        user_skills = []

    if not user_skills:
        try:
            user = _require_user(request)
        except HTTPException:
            user = None
        if user:
            profile = _load_profile_for_email(str(user.get("email", "")))
            if not role:
                basics = profile.get("basics", {}) if isinstance(profile.get("basics"), dict) else {}
                role = str(basics.get("position", "")).strip() or None
            skills = profile.get("skills", [])
            if isinstance(skills, list):
                user_skills = [str(skill).strip() for skill in skills if str(skill).strip()]

    try:
        limit = max(1, min(int(limit_value), 500))
    except (TypeError, ValueError):
        limit = 200

    ranked = _rank_inventory(role=role, user_skills=user_skills, limit=limit)
    return JSONResponse(
        {
            "role": role or "",
            "count": len(ranked),
            "ranked": ranked,
        }
    )


def get_ranked_summary_service() -> JSONResponse:
    with SessionLocal() as db:
        rows = db.execute(select(RankedJob)).scalars().all()
    ranked = [
        {
            "ref": row.ref,
            "position": row.position,
            "employer": row.employer,
            "url": row.url,
            "match_percent": row.match_percent or 0,
        }
        for row in rows
    ]
    if not ranked:
        return JSONResponse({"best": None, "top": []})

    sorted_jobs = sorted(ranked, key=lambda j: j.get("match_percent", 0), reverse=True)
    best = sorted_jobs[0]
    filtered = [j for j in sorted_jobs if j.get("match_percent", 0) > 0]
    top = filtered[:5]
    return JSONResponse(
        {
            "best": {
                "ref": best.get("ref", ""),
                "position": best.get("position", ""),
                "employer": best.get("employer", ""),
                "url": best.get("url", ""),
                "match_percent": round(best.get("match_percent", 0)),
            },
            "top": [
                {
                    "ref": job.get("ref", ""),
                    "position": job.get("position", ""),
                    "employer": job.get("employer", ""),
                    "url": job.get("url", ""),
                    "match_percent": round(job.get("match_percent", 0)),
                }
                for job in top
            ],
        }
    )
