from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from lib.database.db import SessionLocal
from lib.database.models import CvFile


def _save_cv_index(entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    with SessionLocal() as db:
        for entry in entries:
            cv_id = str(entry.get("id", "")).strip()
            if not cv_id:
                continue
            row = db.execute(select(CvFile).where(CvFile.id == cv_id)).scalar_one_or_none()
            uploaded_at = entry.get("uploadedAt")
            try:
                uploaded_dt = datetime.fromisoformat(str(uploaded_at))
            except Exception:
                uploaded_dt = datetime.now(tz=timezone.utc)
            if row:
                row.path = str(entry.get("path", row.path))
                row.original_name = str(entry.get("originalName", row.original_name))
                row.size = int(entry.get("size", row.size) or 0)
                row.content_type = str(entry.get("contentType", row.content_type))
                row.uploaded_at = uploaded_dt
            else:
                db.add(
                    CvFile(
                        id=cv_id,
                        path=str(entry.get("path", "")),
                        original_name=str(entry.get("originalName", "")),
                        size=int(entry.get("size", 0) or 0),
                        content_type=str(entry.get("contentType", "application/octet-stream")),
                        uploaded_at=uploaded_dt,
                    )
                )
        db.commit()
