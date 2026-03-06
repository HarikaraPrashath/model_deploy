from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from service.career_market.utils.auth_utils import _require_user
from service.career_market.utils.config import TREND_WINDOW_DAYS
from service.career_market.utils.trends_utils import (
    _load_trend_history,
    _save_trend_history,
    _seed_trend_history,
    _summarize_trends,
)


def get_trend_history_service() -> JSONResponse:
    history = _load_trend_history()
    return JSONResponse({"history": history})


def get_trends_service() -> JSONResponse:
    history = _load_trend_history()
    summary = _summarize_trends(history)
    return JSONResponse(summary)


def seed_trends_service(request: Request, payload: dict[str, Any] | None = None) -> JSONResponse:
    _require_user(request)
    payload = payload or {}
    days = payload.get("days", TREND_WINDOW_DAYS)
    replace = bool(payload.get("replace", True))
    try:
        days_int = int(days)
    except Exception:
        days_int = TREND_WINDOW_DAYS

    history = _seed_trend_history(days_int, replace)
    _save_trend_history(history)
    summary = _summarize_trends(history)
    return JSONResponse({"ok": True, "summary": summary})
