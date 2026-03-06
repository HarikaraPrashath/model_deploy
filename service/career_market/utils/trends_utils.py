from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any

from sqlalchemy import delete, select

from lib.database.db import SessionLocal
from lib.database.models import TrendSnapshot
from service.career_market.utils.config import (
    TREND_DECLINE_THRESHOLD,
    TREND_HISTORY_DAYS,
    TREND_MIN_COUNT,
    TREND_RISE_THRESHOLD,
    TREND_WINDOW_DAYS,
)
from service.career_market.utils.jobs_utils import _load_jobs_payload
from service.career_market.utils.skills_utils import SKILLS


def _normalize_term(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _load_trend_history() -> list[dict[str, Any]]:
    with SessionLocal() as db:
        rows = db.execute(select(TrendSnapshot).order_by(TrendSnapshot.ran_at.asc())).scalars().all()
    return [
        {
            "ranAt": row.ran_at.isoformat(),
            "keyword": row.keyword,
            "jobCount": row.job_count,
            "skillCounts": row.skill_counts or {},
            "roleCounts": row.role_counts or {},
        }
        for row in rows
    ]


def _save_trend_history(entries: list[dict[str, Any]]) -> None:
    with SessionLocal() as db:
        db.execute(delete(TrendSnapshot))
        db.commit()
        for entry in entries:
            try:
                ran_at = datetime.fromisoformat(str(entry.get("ranAt")))
            except Exception:
                continue
            keyword = str(entry.get("keyword", "")).strip()
            if not keyword:
                continue
            db.add(
                TrendSnapshot(
                    ran_at=ran_at,
                    keyword=keyword,
                    job_count=int(entry.get("jobCount", 0) or 0),
                    skill_counts=entry.get("skillCounts", {}) or {},
                    role_counts=entry.get("roleCounts", {}) or {},
                )
            )
        db.commit()


def _count_skills(ranked: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in ranked:
        skills = job.get("skills_found")
        if not isinstance(skills, list):
            continue
        for skill in skills:
            key = _normalize_term(str(skill))
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
    return counts


def _count_roles(metadata: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in metadata:
        position = job.get("position")
        if not isinstance(position, str):
            continue
        key = _normalize_term(position)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _record_trend_snapshot(
    keyword: str,
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

    try:
        local_tz = ZoneInfo("Asia/Colombo")
    except Exception:
        local_tz = timezone.utc
    now = datetime.now(tz=local_tz)
    now_local_naive = now.replace(tzinfo=None)
    snapshot = {
        "ranAt": now.isoformat(),
        "keyword": keyword,
        "jobCount": len(metadata),
        "skillCounts": _count_skills(ranked),
        "roleCounts": _count_roles(metadata),
    }

    with SessionLocal() as db:
        db.add(
            TrendSnapshot(
                # Store local time as naive timestamp so DB shows Sri Lanka dates.
                ran_at=now_local_naive,
                keyword=keyword,
                job_count=len(metadata),
                skill_counts=snapshot["skillCounts"],
                role_counts=snapshot["roleCounts"],
            )
        )
        cutoff = now_local_naive - timedelta(days=TREND_HISTORY_DAYS)
        db.execute(delete(TrendSnapshot).where(TrendSnapshot.ran_at < cutoff))
        db.commit()


def _summarize_trends(history: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        local_tz = ZoneInfo("Asia/Colombo")
    except Exception:
        local_tz = timezone.utc
    now = datetime.now(tz=local_tz)
    window_cutoff = now - timedelta(days=TREND_WINDOW_DAYS)
    windowed: list[dict[str, Any]] = []
    for entry in history:
        try:
            ran_at = datetime.fromisoformat(str(entry.get("ranAt")))
        except Exception:
            continue
        if ran_at.tzinfo is None:
            # Treat stored timestamps as Sri Lanka local time.
            ran_at_local = ran_at.replace(tzinfo=local_tz)
        else:
            try:
                ran_at_local = ran_at.astimezone(local_tz)
            except Exception:
                ran_at_local = ran_at
        if ran_at_local >= window_cutoff:
            windowed.append({**entry, "_ranAt": ran_at_local})

    if not windowed:
        return {
            "windowDays": TREND_WINDOW_DAYS,
            "snapshotCount": 0,
            "latestAt": None,
            "skills": {"emerging": [], "rising": [], "declining": [], "stable": []},
            "roles": {"emerging": [], "rising": [], "declining": [], "stable": []},
        }

    latest = max(windowed, key=lambda item: item["_ranAt"])
    baseline = [entry for entry in windowed if entry is not latest]

    def build_summary(key: str) -> dict[str, list[dict[str, Any]]]:
        current_counts = latest.get(key, {})
        if not isinstance(current_counts, dict):
            current_counts = {}

        baseline_counts: dict[str, list[int]] = {}
        for entry in baseline:
            counts = entry.get(key, {})
            if not isinstance(counts, dict):
                continue
            for term, count in counts.items():
                try:
                    count_value = int(count)
                except Exception:
                    continue
                baseline_counts.setdefault(term, []).append(count_value)

        all_terms = set(current_counts.keys()) | set(baseline_counts.keys())
        emerging: list[dict[str, Any]] = []
        rising: list[dict[str, Any]] = []
        declining: list[dict[str, Any]] = []
        stable: list[dict[str, Any]] = []

        for term in all_terms:
            try:
                current = int(current_counts.get(term, 0))
            except Exception:
                current = 0
            baseline_list = baseline_counts.get(term, [])
            baseline_avg = sum(baseline_list) / len(baseline_list) if baseline_list else 0

            if current < TREND_MIN_COUNT and baseline_avg < TREND_MIN_COUNT:
                continue

            if baseline_avg == 0 and current >= TREND_MIN_COUNT:
                emerging.append(
                    {"term": term, "current": current, "baseline": 0, "changePct": None}
                )
                continue

            if baseline_avg > 0:
                change_pct = (current - baseline_avg) / baseline_avg
                entry = {
                    "term": term,
                    "current": current,
                    "baseline": round(baseline_avg, 2),
                    "changePct": round(change_pct * 100, 1),
                }
                if change_pct >= TREND_RISE_THRESHOLD:
                    rising.append(entry)
                elif change_pct <= -TREND_DECLINE_THRESHOLD:
                    declining.append(entry)
                else:
                    stable.append(entry)

        emerging.sort(key=lambda item: item["current"], reverse=True)
        rising.sort(key=lambda item: item["changePct"] or 0, reverse=True)
        declining.sort(key=lambda item: item["changePct"] or 0)
        stable.sort(key=lambda item: item["current"], reverse=True)

        return {
            "emerging": emerging[:10],
            "rising": rising[:10],
            "declining": declining[:10],
            "stable": stable[:10],
        }

    return {
        "windowDays": TREND_WINDOW_DAYS,
        "snapshotCount": len(windowed),
        "latestAt": latest["_ranAt"].isoformat(),
        "skills": build_summary("skillCounts"),
        "roles": build_summary("roleCounts"),
    }


def _seed_trend_history(days: int, replace: bool) -> list[dict[str, Any]]:
    rng = random.Random(42)
    days = max(2, min(days, 30))
    now = datetime.now(tz=timezone.utc)

    skills_pool = [s.strip() for s in SKILLS if s.strip()]
    if len(skills_pool) < 8:
        skills_pool += ["AWS", "Docker", "React", "SQL", "Python", "Java"]
    roles_pool = [
        "data scientist",
        "ml engineer",
        "backend engineer",
        "frontend engineer",
        "devops engineer",
        "data analyst",
        "product analyst",
        "ai engineer",
    ]

    tracked_skills = rng.sample(skills_pool, k=min(8, len(skills_pool)))
    tracked_roles = rng.sample(roles_pool, k=6)
    emerging_skill = rng.choice(skills_pool)
    emerging_role = rng.choice(roles_pool)

    history: list[dict[str, Any]] = []
    for i in range(days):
        ran_at = now - timedelta(days=days - 1 - i)
        drift = (i - (days / 2)) / max(1, days / 2)

        skill_counts: dict[str, int] = {}
        for skill in tracked_skills:
            base = rng.randint(2, 8)
            count = max(1, int(base + drift * rng.randint(1, 4) + rng.randint(-1, 2)))
            skill_counts[_normalize_term(skill)] = count

        role_counts: dict[str, int] = {}
        for role in tracked_roles:
            base = rng.randint(2, 9)
            count = max(1, int(base + drift * rng.randint(1, 4) + rng.randint(-2, 2)))
            role_counts[_normalize_term(role)] = count

        if i >= days // 2:
            skill_counts[_normalize_term(emerging_skill)] = rng.randint(2, 6)
            role_counts[_normalize_term(emerging_role)] = rng.randint(2, 5)

        history.append(
            {
                "ranAt": ran_at.isoformat(),
                "keyword": "seed",
                "jobCount": rng.randint(30, 80),
                "skillCounts": skill_counts,
                "roleCounts": role_counts,
            }
        )

    if replace:
        return history
    existing = _load_trend_history()
    return existing + history
