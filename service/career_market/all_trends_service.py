from __future__ import annotations

from fastapi.responses import JSONResponse

from service.career_market.utils.all_trends_utils import (
    _load_all_trend_history,
    _summarize_all_trends,
)


import os
import json

def get_all_trend_history_service() -> JSONResponse:
    history = _load_all_trend_history()
    return JSONResponse({"history": history})


def get_all_trends_service() -> JSONResponse:
    history = _load_all_trend_history()
    summary = _summarize_all_trends(history)
    
    # 🔥 Load forecasted trends
    forecasted_path = os.path.join("service", "career_market", "skill_predict", "top_trends.json")
    forecasted = {"rising": [], "declining": []}
    
    if os.path.exists(forecasted_path):
        try:
            with open(forecasted_path, "r") as f:
                forecasted = json.load(f)
        except Exception:
            pass
            
    summary["forecasted"] = forecasted
    return JSONResponse(summary)
