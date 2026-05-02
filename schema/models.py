from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import Column, String, DateTime
from database.database import Base


def _utcnow() -> datetime:
    return datetime.utcnow()


class User(Base):
    __tablename__ = "users"

    # uuid primary key matches other models
    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    name = Column(String, nullable=True)  # optional field added later via migration
    email = Column(String, unique=True, index=True, nullable=False)

    # the table uses a salt/hash scheme rather than plain password
    password_salt = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)

    # token is used by the career-market endpoints for bearer authentication
    token = Column(String, nullable=True, index=True)

    created_at = Column(DateTime, default=_utcnow, nullable=False)