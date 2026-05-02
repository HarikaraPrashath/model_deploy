from __future__ import annotations

from fastapi.responses import JSONResponse

from service.career_market.utils.all_trends_utils import (
    _load_all_trend_history,
    _summarize_all_trends,
)


def get_all_trend_history_service() -> JSONResponse:
    history = _load_all_trend_history()
    return JSONResponse({"history": history})


def get_all_trends_service() -> JSONResponse:
    history = _load_all_trend_history()
    summary = _summarize_all_trends(history)
    return JSONResponse(summary)
