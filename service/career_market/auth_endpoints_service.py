from __future__ import annotations

from typing import Any

from datetime import datetime, timezone

from fastapi import HTTPException
from fastapi.responses import JSONResponse
import secrets


def signup_service(payload: dict[str, Any]) -> JSONResponse:
    from service.career_market.utils.auth_utils import (
        _find_user,
        _hash_password,
        _load_users,
        _normalize_email,
        _save_users,
    )
    email = str(payload.get("email", "")).strip()
    password = str(payload.get("password", ""))
    confirm_password = str(payload.get("confirmPassword", ""))

    if not email or "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if not password or len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if confirm_password and confirm_password != password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    if _find_user(email):
        raise HTTPException(status_code=409, detail="Account already exists.")

    salt = secrets.token_bytes(16)
    token = secrets.token_urlsafe(32)
    user = {
        "email": _normalize_email(email),
        "passwordSalt": salt.hex(),
        "passwordHash": _hash_password(password, salt),
        "token": token,
        "createdAt": datetime.now(tz=timezone.utc).isoformat(),
    }
    users = _load_users()
    users.append(user)
    _save_users(users)

    return JSONResponse(
        {"ok": True, "user": {"email": user["email"]}, "token": token}
    )


def login_service(payload: dict[str, Any]) -> JSONResponse:
    from service.career_market.utils.auth_utils import (
        _hash_password,
        _load_users,
        _normalize_email,
        _save_users,
    )
    email = str(payload.get("email", "")).strip()
    password = str(payload.get("password", ""))

    if not email or "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if not password:
        raise HTTPException(status_code=400, detail="Password is required.")

    users = _load_users()
    normalized = _normalize_email(email)
    user_index = next(
        (idx for idx, item in enumerate(users) if _normalize_email(str(item.get("email", ""))) == normalized),
        None,
    )
    if user_index is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    user = users[user_index]
    salt_hex = str(user.get("passwordSalt", ""))
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        raise HTTPException(status_code=500, detail="Corrupt user record.")

    expected = str(user.get("passwordHash", ""))
    provided = _hash_password(password, salt)
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = secrets.token_urlsafe(32)
    user["token"] = token
    users[user_index] = user
    _save_users(users)

    return JSONResponse(
        {"ok": True, "user": {"email": user.get("email", "")}, "token": token}
    )


def forgot_password_service(payload: dict[str, Any]) -> JSONResponse:
    email = str(payload.get("email", "")).strip()
    if not email or "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")

    return JSONResponse({"ok": True, "message": "Check your inbox for a reset link."})
