from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.repositories import CvRepository, StateRepository, TrendRepository

STORAGE_DIR = BASE_DIR / "career_market" / "storage"
CV_INDEX_PATH = STORAGE_DIR / "cv_index.json"
LAST_QUERY_PATH = STORAGE_DIR / "last_query.json"
TREND_HISTORY_PATH = STORAGE_DIR / "trends_history.json"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def backfill(db: Session) -> None:
    cv_repo = CvRepository(db)
    trend_repo = TrendRepository(db)
    state_repo = StateRepository(db)

    cv_entries = _read_json(CV_INDEX_PATH, [])
    if isinstance(cv_entries, list):
        for item in cv_entries:
            try:
                cv_repo.create(
                    file_id=str(item.get("id")),
                    user_id=None,
                    original_name=str(item.get("originalName", "cv_upload")),
                    content_type=str(item.get("contentType", "application/octet-stream")),
                    size=int(item.get("size", 0)),
                    storage_path=Path(str(item.get("path", ""))),
                    uploaded_at=datetime.fromisoformat(str(item.get("uploadedAt"))),
                )
            except Exception:
                continue

    trend_history = _read_json(TREND_HISTORY_PATH, [])
    normalized = []
    if isinstance(trend_history, list):
        for item in trend_history:
            try:
                normalized.append(
                    {
                        "ran_at": datetime.fromisoformat(str(item.get("ranAt"))),
                        "keyword": str(item.get("keyword", "")),
                        "job_count": int(item.get("jobCount", 0)),
                        "skill_counts": item.get("skillCounts", {}) if isinstance(item.get("skillCounts", {}), dict) else {},
                        "role_counts": item.get("roleCounts", {}) if isinstance(item.get("roleCounts", {}), dict) else {},
                    }
                )
            except Exception:
                continue
    if normalized:
        trend_repo.replace_history(normalized)

    last_query = _read_json(LAST_QUERY_PATH, {})
    if isinstance(last_query, dict) and last_query:
        state_repo.upsert("last_query", last_query)

    db.commit()


def main() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        backfill(db)
        print("Backfill complete.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
