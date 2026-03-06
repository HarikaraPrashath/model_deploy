from __future__ import annotations

from datetime import datetime
from typing import Any
import uuid

from sqlalchemy import Column, DateTime, Integer, String, Text, Float, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base


Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.utcnow()


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    email = Column(String, unique=True, nullable=False, index=True)
    password_salt = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    token = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    email = Column(String, unique=True, nullable=False, index=True)
    profile_json = Column(JSONB, nullable=False, default=dict)
    updated_at = Column(DateTime, default=_utcnow, nullable=False)


class CvFile(Base):
    __tablename__ = "cv_files"

    id = Column(String, primary_key=True)
    path = Column(Text, nullable=False)
    original_name = Column(Text, nullable=False)
    size = Column(Integer, nullable=False)
    content_type = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=_utcnow, nullable=False, index=True)


class LastQuery(Base):
    __tablename__ = "last_query"

    id = Column(Integer, primary_key=True, default=1)
    keyword = Column(String, nullable=False)
    ran_at = Column(DateTime, nullable=False)


class JobMetadata(Base):
    __tablename__ = "job_metadata"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    ref = Column(String, nullable=True, index=True)
    position = Column(Text, nullable=True)
    employer = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    ad_type = Column(String, nullable=True)
    files = Column(JSONB, nullable=False, default=list)
    text_snippet = Column(Text, nullable=True)
    image_file = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False, index=True)


class RankedJob(Base):
    __tablename__ = "ranked_jobs"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    ref = Column(String, nullable=True, index=True)
    position = Column(Text, nullable=True)
    employer = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    skills_found = Column(JSONB, nullable=False, default=list)
    overlap = Column(JSONB, nullable=False, default=list)
    missing = Column(JSONB, nullable=False, default=list)
    match_percent = Column(Float, nullable=True)
    baseline_match_percent = Column(Float, nullable=True)
    job_skill_count = Column(Integer, nullable=True)
    user_skill_count = Column(Integer, nullable=True)
    text_excerpt = Column(Text, nullable=True)
    text_full = Column(Text, nullable=True)
    must_have_skills = Column(JSONB, nullable=False, default=list)
    nice_to_have_skills = Column(JSONB, nullable=False, default=list)
    core_skills = Column(JSONB, nullable=False, default=list)
    matched_must_have = Column(JSONB, nullable=False, default=list)
    missing_must_have = Column(JSONB, nullable=False, default=list)
    must_have_gate_pass = Column(Boolean, nullable=True)
    matched_nice_to_have = Column(JSONB, nullable=False, default=list)
    weighted_components = Column(JSONB, nullable=False, default=dict)
    explanations = Column(JSONB, nullable=False, default=list)
    created_at = Column(DateTime, default=_utcnow, nullable=False, index=True)


class TrendSnapshot(Base):
    __tablename__ = "trend_history"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    ran_at = Column(DateTime, nullable=False, index=True)
    keyword = Column(String, nullable=False)
    job_count = Column(Integer, nullable=False)
    skill_counts = Column(JSONB, nullable=False, default=dict)
    role_counts = Column(JSONB, nullable=False, default=dict)
