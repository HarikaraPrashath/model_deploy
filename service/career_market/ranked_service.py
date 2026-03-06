from __future__ import annotations

from fastapi.responses import JSONResponse
from sqlalchemy import select

from lib.database.db import SessionLocal
from lib.database.models import RankedJob


def get_ranked_service() -> JSONResponse:
    print("[ranked] fetch ranked list")
    with SessionLocal() as db:
        rows = db.execute(select(RankedJob).order_by(RankedJob.match_percent.desc().nullslast())).scalars().all()
    ranked = [
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
    return JSONResponse({"ranked": ranked})


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
