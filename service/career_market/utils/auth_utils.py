from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, Request
from sqlalchemy import select

from lib.database.db import SessionLocal
from lib.database.models import User


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return digest.hex()


def _load_users() -> list[dict[str, Any]]:
    with SessionLocal() as db:
        rows = db.execute(select(User)).scalars().all()
    return [
        {
            "email": row.email,
            "passwordSalt": row.password_salt,
            "passwordHash": row.password_hash,
            "token": row.token,
            "createdAt": row.created_at.isoformat(),
        }
        for row in rows
    ]


def _save_users(users: list[dict[str, Any]]) -> None:
    if not users:
        return
    with SessionLocal() as db:
        for user in users:
            email = _normalize_email(str(user.get("email", "")))
            if not email:
                continue
            row = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
            if row:
                row.password_salt = str(user.get("passwordSalt", row.password_salt))
                row.password_hash = str(user.get("passwordHash", row.password_hash))
                row.token = user.get("token", row.token)
            else:
                created_at = user.get("createdAt")
                try:
                    created_dt = datetime.fromisoformat(str(created_at))
                except Exception:
                    created_dt = datetime.now(tz=timezone.utc)
                db.add(
                    User(
                        email=email,
                        password_salt=str(user.get("passwordSalt", "")),
                        password_hash=str(user.get("passwordHash", "")),
                        token=user.get("token"),
                        created_at=created_dt,
                    )
                )
        db.commit()


def _find_user(email: str) -> dict[str, Any] | None:
    normalized = _normalize_email(email)
    with SessionLocal() as db:
        row = db.execute(select(User).where(User.email == normalized)).scalar_one_or_none()
    if not row:
        return None
    return {
        "email": row.email,
        "passwordSalt": row.password_salt,
        "passwordHash": row.password_hash,
        "token": row.token,
        "createdAt": row.created_at.isoformat(),
    }


def _extract_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization")
    if not auth_header:
        return None
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def _find_user_by_token(token: str) -> dict[str, Any] | None:
    with SessionLocal() as db:
        row = db.execute(select(User).where(User.token == token)).scalar_one_or_none()
    if not row:
        return None
    return {
        "email": row.email,
        "passwordSalt": row.password_salt,
        "passwordHash": row.password_hash,
        "token": row.token,
        "createdAt": row.created_at.isoformat(),
    }


def _require_user(request: Request) -> dict[str, Any]:
    token = _extract_bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing authorization token.")
    user = _find_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return user
