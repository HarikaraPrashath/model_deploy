from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from service.career_market.utils.auth_utils import _require_user
from service.career_market.utils.profile_utils import (
    _coerce_profile,
    _load_profile_for_email,
    _save_profile_for_email,
)


def get_profile_service(request: Request) -> JSONResponse:
    user = _require_user(request)
    return JSONResponse(_load_profile_for_email(str(user.get("email", ""))))


def put_profile_service(payload: dict[str, Any], request: Request) -> JSONResponse:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid profile payload.")
    user = _require_user(request)
    stored = _coerce_profile(payload)
    _save_profile_for_email(str(user.get("email", "")), stored)
    return JSONResponse({"ok": True})
