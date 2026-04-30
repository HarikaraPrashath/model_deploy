from __future__ import annotations

import re
from typing import Iterable


ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "AI Engineer": (
        r"\bai engineer\b",
        r"\bai systems engineer\b",
        r"\bai platform\b",
        r"\bautomation\s*&\s*ai\b",
        r"\bml engineer\b",
        r"\bmachine learning\b",
        r"\bdeep learning\b",
        r"\bllm\b",
        r"\bnlp\b",
        r"\bcomputer vision\b",
        r"\bdata scientist\b",
    ),
    "Software Engineer": (
        r"\bsoftware engineer\b",
        r"\bsoftware developer\b",
        r"\bdeveloper\b",
        r"\bprogrammer\b",
        r"\bfull stack engineer\b",
        r"\bfull stack developer\b",
        r"\bapplication engineer\b",
        r"\bfull stack\b",
        r"\bbackend\b",
        r"\bfrontend\b",
        r"\bweb engineer\b",
    ),
    "Data Analyst": (
        r"\bdata analyst\b",
        r"\bbusiness intelligence\b",
        r"\bbi analyst\b",
        r"\banalytics\b",
        r"\breporting analyst\b",
    ),
    "DevOps Engineer": (
        r"\bdevops\b",
        r"\bsite reliability\b",
        r"\bsre\b",
        r"\bplatform engineer\b",
        r"\bcloud engineer\b",
        r"\binfrastructure engineer\b",
    ),
    "QA Engineer": (
        r"\bqa engineer\b",
        r"\bquality assurance\b",
        r"\btest engineer\b",
        r"\bautomation tester\b",
        r"\bsdet\b",
    ),
    "Cybersecurity Analyst": (
        r"\bcyber ?security\b",
        r"\bsecurity analyst\b",
        r"\bsecurity engineer\b",
        r"\binformation security\b",
        r"\bdevsecops\b",
    ),
    "UI/UX Designer": (
        r"\bui/?ux\b",
        r"\bproduct designer\b",
        r"\bux designer\b",
        r"\bui designer\b",
        r"\binteraction designer\b",
        r"\bgraphic designer\b",
    ),
    "Product Manager": (
        r"\bproduct manager\b",
        r"\bproduct owner\b",
        r"\btechnical product manager\b",
        r"\bproject manager\b",
    ),
    "Network Engineer": (
        r"\bnetwork engineer\b",
        r"\bnetwork administrator\b",
        r"\bsystem administrator\b",
        r"\bit support\b",
    ),
}


def normalize_role_label(value: str) -> str:
    return " ".join(value.strip().lower().split())


def infer_role_tags(position: str, text: str = "", source_keyword: str = "") -> list[str]:
    haystack = " ".join(part for part in [position, text, source_keyword] if part).lower()
    tags = [
        role
        for role, patterns in ROLE_PATTERNS.items()
        if any(re.search(pattern, haystack) for pattern in patterns)
    ]
    if not tags and position.strip():
        tags.append(position.strip())
    elif source_keyword.strip():
        keyword = source_keyword.strip()
        if keyword not in tags:
            tags.append(keyword)
    return tags


def role_matches(role_tags: Iterable[str] | None, selected_role: str | None) -> bool:
    if not selected_role:
        return True
    target = normalize_role_label(selected_role)
    tags = [normalize_role_label(tag) for tag in (role_tags or []) if str(tag).strip()]
    if target in tags:
        return True
    return any(target in tag or tag in target for tag in tags)
