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
    try:
        user = _require_user(request)
        return JSONResponse(_load_profile_for_email(str(user.get("email", ""))))
    except HTTPException as he:
        # Re-raise or return the specific HTTP error
        return JSONResponse({"error": str(he.detail)}, status_code=he.status_code)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("⚠️ Error in get_profile_service:", str(e))
        print(tb)
        return JSONResponse({"error": str(e), "trace": tb}, status_code=500)


def put_profile_service(payload: dict[str, Any], request: Request) -> JSONResponse:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid profile payload.")
    try:
        user = _require_user(request)
        stored = _coerce_profile(payload)
        _save_profile_for_email(str(user.get("email", "")), stored)
        return JSONResponse({"ok": True})
    except HTTPException as he:
        return JSONResponse({"error": str(he.detail)}, status_code=he.status_code)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("⚠️ Error in put_profile_service:", str(e))
        print(tb)
        return JSONResponse({"error": str(e), "trace": tb}, status_code=500)
