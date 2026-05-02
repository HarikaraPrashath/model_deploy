from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any

from sqlalchemy import select

from lib.database.db import SessionLocal
from lib.database.models import ScrTrend
from service.career_market.utils.config import (
    TREND_DECLINE_THRESHOLD,
    TREND_MIN_COUNT,
    TREND_RISE_THRESHOLD,
    TREND_WINDOW_DAYS,
)


def _load_all_trend_history() -> list[dict[str, Any]]:
    with SessionLocal() as db:
        rows = db.execute(select(ScrTrend).order_by(ScrTrend.ran_at.asc())).scalars().all()
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


def _summarize_all_trends(history: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        local_tz = ZoneInfo("Asia/Colombo")
    except Exception:
        local_tz = timezone.utc
    now = datetime.now(tz=local_tz)
    # For "All Trend", we might want a larger window, but let's stick to TREND_WINDOW_DAYS or maybe 30.
    window_cutoff = now - timedelta(days=TREND_WINDOW_DAYS * 2) # Double the window for "All Trend"
    windowed: list[dict[str, Any]] = []
    for entry in history:
        try:
            ran_at = datetime.fromisoformat(str(entry.get("ranAt")))
        except Exception:
            continue
        if ran_at.tzinfo is None:
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
            "windowDays": TREND_WINDOW_DAYS * 2,
            "snapshotCount": 0,
            "latestAt": None,
            "skills": {"emerging": [], "rising": [], "declining": [], "stable": []},
            "roles": {"emerging": [], "rising": [], "declining": [], "stable": []},
        }

    # Group by keyword to find the latest snapshot for each
    latest_by_keyword: dict[str, dict[str, Any]] = {}
    for entry in windowed:
        kw = entry.get("keyword", "unknown")
        if kw not in latest_by_keyword or entry["_ranAt"] > latest_by_keyword[kw]["_ranAt"]:
            latest_by_keyword[kw] = entry

    latest_snapshots = list(latest_by_keyword.values())
    latest_ids = {id(s) for s in latest_snapshots}
    baseline = [entry for entry in windowed if id(entry) not in latest_ids]

    def build_summary(key: str) -> dict[str, list[dict[str, Any]]]:
        # Aggregate current counts across all latest snapshots
        current_counts: dict[str, int] = {}
        for s in latest_snapshots:
            counts = s.get(key, {})
            if isinstance(counts, dict):
                for term, count in counts.items():
                    current_counts[term] = current_counts.get(term, 0) + int(count)

        # Aggregate baseline counts
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
            current = current_counts.get(term, 0)
            baseline_list = baseline_counts.get(term, [])
            baseline_avg = sum(baseline_list) / len(baseline_list) if baseline_list else 0

            # Filter out low-volume terms
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
            "emerging": emerging[:15],
            "rising": rising[:15],
            "declining": declining[:15],
            "stable": stable[:30],
        }

    latest_at = max(s["_ranAt"] for s in latest_snapshots) if latest_snapshots else None

    return {
        "windowDays": TREND_WINDOW_DAYS * 2,
        "snapshotCount": len(windowed),
        "latestAt": latest_at.isoformat() if latest_at else None,
        "skills": build_summary("skillCounts"),
        "roles": build_summary("roleCounts"),
    }
