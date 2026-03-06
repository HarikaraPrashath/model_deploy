from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException, Request
from sqlalchemy import select

from lib.database.db import SessionLocal
from lib.database.models import User


def normalize_email(email: str) -> str:
    return email.strip().lower()


def extract_bearer_token(request: Request) -> Optional[str]:
    auth_header = request.headers.get("authorization")
    if not auth_header:
        return None
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def find_user_by_token(token: str) -> Optional[dict[str, Any]]:
    with SessionLocal() as db:
        row = db.execute(select(User).where(User.token == token)).scalar_one_or_none()
    if not row:
        return None
    created_at = row.created_at
    if isinstance(created_at, datetime) and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return {
        "email": row.email,
        "passwordSalt": row.password_salt,
        "passwordHash": row.password_hash,
        "token": row.token,
        "createdAt": created_at.isoformat() if isinstance(created_at, datetime) else "",
    }


def require_user(request: Request) -> dict[str, Any]:
    token = extract_bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing authorization token.")
    user = find_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return user
