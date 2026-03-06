from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
# Load env from project root; allow override to avoid empty/placeholder values
load_dotenv(dotenv_path=ENV_PATH, override=True)
# Fallback: also try current working directory if different
load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)


def _normalize_db_url(url: str) -> str:
    if not url:
        return url
    if "supabase.co" in url and "sslmode=" not in url:
        joiner = "&" if "?" in url else "?"
        return f"{url}{joiner}sslmode=require"
    return url


DATABASE_URL = _normalize_db_url(os.environ.get("DATABASE_URL", "").strip())
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Add it to model_deploy/.env")


engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
