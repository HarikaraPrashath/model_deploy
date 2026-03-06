from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from lib.database.db import SessionLocal
from lib.database.models import Profile
from service.career_market.utils.auth_utils import _normalize_email
from service.career_market.utils.config import PROFILES_DIR


def _default_profile() -> dict[str, Any]:
    return {
        "basics": {
            "firstName": "",
            "lastName": "",
            "additionalName": "",
            "headline": "",
            "position": "",
            "industry": "",
            "school": "",
            "country": "",
            "city": "",
            "contactEmail": "",
            "showCurrentCompany": True,
            "showSchool": True,
        },
        "about": "",
        "experiences": [],
        "educationItems": [],
        "skills": [],
        "projects": [],
        "certifications": [],
        "recommendations": [],
    }


def _build_student_profile(profile: dict[str, Any], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    defaults = defaults or {}
    basics = profile.get("basics") if isinstance(profile.get("basics"), dict) else {}
    first = str(basics.get("firstName", "")).strip()
    last = str(basics.get("lastName", "")).strip()
    name = " ".join(part for part in [first, last] if part).strip()
    if not name:
        name = str(defaults.get("name", "")).strip() or "Student"

    skills = profile.get("skills", [])
    skills_list = (
        [str(skill).strip() for skill in skills if str(skill).strip()]
        if isinstance(skills, list)
        else []
    )
    if not skills_list:
        fallback_skills = defaults.get("technical_skills", [])
        if isinstance(fallback_skills, list):
            skills_list = [str(skill).strip() for skill in fallback_skills if str(skill).strip()]

    projects: list[Any] = []
    raw_projects = profile.get("projects", [])
    if isinstance(raw_projects, list):
        for item in raw_projects:
            if isinstance(item, dict):
                entry: dict[str, Any] = {}
                title = item.get("title") or item.get("name")
                description = item.get("description") or item.get("summary")
                technologies = item.get("technologies") or item.get("skills")
                if title:
                    entry["title"] = str(title)
                if description:
                    entry["description"] = str(description)
                if isinstance(technologies, list):
                    entry["technologies"] = [
                        str(tech).strip() for tech in technologies if str(tech).strip()
                    ]
                if entry:
                    projects.append(entry)
            elif isinstance(item, str):
                projects.append(item)
    if not projects:
        fallback_projects = defaults.get("projects", [])
        if isinstance(fallback_projects, list):
            projects = fallback_projects

    experience = profile.get("experiences", [])
    if not isinstance(experience, list) or not experience:
        fallback_experience = defaults.get("experience", [])
        experience = fallback_experience if isinstance(fallback_experience, list) else []

    certifications = profile.get("certifications", [])
    if not isinstance(certifications, list) or not certifications:
        fallback_certs = defaults.get("certifications", [])
        certifications = fallback_certs if isinstance(fallback_certs, list) else []

    soft_skills = defaults.get("soft_skills", [])
    if isinstance(soft_skills, list):
        soft_skills = [str(skill).strip() for skill in soft_skills if str(skill).strip()]
    else:
        soft_skills = []

    return {
        "name": name,
        "technical_skills": skills_list,
        "soft_skills": soft_skills,
        "certifications": certifications if isinstance(certifications, list) else [],
        "projects": projects,
        "experience": experience if isinstance(experience, list) else [],
    }


def _coerce_profile(payload: dict[str, Any]) -> dict[str, Any]:
    base = _default_profile()
    basics = payload.get("basics") if isinstance(payload.get("basics"), dict) else {}
    base["basics"].update(
        {
            "firstName": basics.get("firstName", ""),
            "lastName": basics.get("lastName", ""),
            "additionalName": basics.get("additionalName", ""),
            "headline": basics.get("headline", ""),
            "position": basics.get("position", ""),
            "industry": basics.get("industry", ""),
            "school": basics.get("school", ""),
            "country": basics.get("country", ""),
            "city": basics.get("city", ""),
            "contactEmail": basics.get("contactEmail", ""),
            "showCurrentCompany": bool(basics.get("showCurrentCompany", True)),
            "showSchool": bool(basics.get("showSchool", True)),
        }
    )

    for key in ["about", "experiences", "educationItems", "skills", "projects", "certifications", "recommendations"]:
        value = payload.get(key, base[key])
        if isinstance(base[key], list):
            base[key] = value if isinstance(value, list) else []
        elif isinstance(base[key], str):
            base[key] = value if isinstance(value, str) else ""

    return base


def _profile_path_for_email(email: str) -> Path:
    safe_key = hashlib.sha256(email.encode("utf-8")).hexdigest()
    return PROFILES_DIR / f"{safe_key}.json"


def _load_profile_for_email(email: str) -> dict[str, Any]:
    with SessionLocal() as db:
        row = db.execute(select(Profile).where(Profile.email == _normalize_email(email))).scalar_one_or_none()
        stored = row.profile_json if row and isinstance(row.profile_json, dict) else {}
    profile = _coerce_profile(stored if isinstance(stored, dict) else {})
    if email and not profile.get("basics", {}).get("contactEmail"):
        profile["basics"]["contactEmail"] = email
    return profile


def _save_profile_for_email(email: str, payload: dict[str, Any]) -> None:
    normalized = _normalize_email(email)
    with SessionLocal() as db:
        row = db.execute(select(Profile).where(Profile.email == normalized)).scalar_one_or_none()
        if row:
            row.profile_json = payload
            row.updated_at = datetime.now(tz=timezone.utc)
        else:
            db.add(
                Profile(
                    email=normalized,
                    profile_json=payload,
                    updated_at=datetime.now(tz=timezone.utc),
                )
            )
        db.commit()
