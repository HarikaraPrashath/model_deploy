from __future__ import annotations


def root_service() -> dict[str, str]:
    return {"status": "Career Prediction API running"}


def health_service() -> dict[str, str]:
    return {"status": "ok"}
