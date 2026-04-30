from __future__ import annotations

import os
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from alembic import context
from dotenv import load_dotenv

from lib.database.models import Base  # type: ignore


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

BASE_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)


def _normalize_db_url(url: str) -> str:
    if not url:
        return url
    if "supabase.co" in url and "sslmode=" not in url:
        joiner = "&" if "?" in url else "?"
        return f"{url}{joiner}sslmode=require"
    return url


db_url = _normalize_db_url(os.environ.get("DATABASE_URL", "").strip())
if not db_url:
    raise RuntimeError("DATABASE_URL is not set. Add it to model_deploy/.env")

config.set_main_option("sqlalchemy.url", db_url.replace("%", "%%"))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

