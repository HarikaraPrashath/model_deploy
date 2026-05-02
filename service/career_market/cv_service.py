from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from sqlalchemy import select

from lib.database.db import SessionLocal
from lib.database.models import CvFile
from service.career_market.auth_service import require_user
from service.career_market import storage_service


BASE_DIR = Path(__file__).resolve().parents[2]
CAREER_MARKET_DIR = BASE_DIR / "service" / "career_market"
CV_EXTRACTOR_DIR = CAREER_MARKET_DIR / "cv_extractor"
STORAGE_DIR = CAREER_MARKET_DIR / "storage"
CV_STORAGE_DIR = STORAGE_DIR / "cvs"
SKILLS_PATH = CV_EXTRACTOR_DIR / "skills.txt"


def load_skills() -> list[str]:
    if SKILLS_PATH.exists():
        lines = SKILLS_PATH.read_text(encoding="utf-8").splitlines()
        skills = [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if skills:
            return skills
    return [
        "Python",
        "SQL",
        "Machine Learning",
        "Deep Learning",
        "NLP",
        "TensorFlow",
        "PyTorch",
        "Docker",
        "Kubernetes",
        "AWS",
        "FastAPI",
        "Django",
        "Flask",
    ]


def parse_cv(
    file: UploadFile,
    max_file_size: int,
) -> dict[str, Any]:
    try:
        from service.career_market.cv_extractor.resume_pipeline import parse_resume
    except ModuleNotFoundError as exc:
        if exc.name == "paddle":
            raise HTTPException(
                status_code=500,
                detail="Missing dependency: paddle. Install paddlepaddle to enable OCR.",
            ) from exc
        raise HTTPException(
            status_code=500,
            detail=f"Resume parser dependency missing: {exc.name}. Check server logs.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to load resume parser.",
        ) from exc

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    contents = file.file.read()
    if len(contents) > max_file_size:
        raise HTTPException(status_code=400, detail="File too large. Max size is 20 MB.")

    content_type = file.content_type or ""
    if content_type and not (content_type == "application/pdf" or content_type.startswith("image/")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use a PDF or image.")

    suffix = Path(file.filename).suffix or ""
    if not suffix and content_type == "application/pdf":
        suffix = ".pdf"
    elif not suffix and content_type.startswith("image/"):
        suffix = f".{content_type.split('/')[-1]}"

    CV_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    cv_id = uuid.uuid4().hex
    stored_name = f"{cv_id}{suffix or '.bin'}"
    stored_path = CV_STORAGE_DIR / stored_name
    stored_path.write_bytes(contents)

    parsed = parse_resume(str(stored_path), skills_list=load_skills())

    stored_reference = str(stored_path)
    if storage_service.storage_enabled():
        remote_path = f"cvs/{stored_name}"
        storage_service.upload_file_to_storage(stored_path, remote_path)
        stored_reference = remote_path
        try:
            stored_path.unlink()
        except Exception:
            pass

    with SessionLocal() as db:
        db.add(
            CvFile(
                id=cv_id,
                path=stored_reference,
                original_name=file.filename,
                size=len(contents),
                content_type=content_type or "application/octet-stream",
                uploaded_at=datetime.now(tz=timezone.utc),
            )
        )
        db.commit()

    return {**parsed, "cvId": cv_id}


def load_cv_index() -> list[dict[str, Any]]:
    with SessionLocal() as db:
        rows = db.execute(select(CvFile).order_by(CvFile.uploaded_at.desc())).scalars().all()
    return [
        {
            "id": row.id,
            "path": row.path,
            "originalName": row.original_name,
            "size": row.size,
            "contentType": row.content_type,
            "uploadedAt": row.uploaded_at.isoformat(),
        }
        for row in rows
    ]


def latest_cv_entry() -> Optional[dict[str, Any]]:
    with SessionLocal() as db:
        row = db.execute(select(CvFile).order_by(CvFile.uploaded_at.desc())).scalar_one_or_none()
    if not row:
        return None
    return {
        "id": row.id,
        "path": row.path,
        "originalName": row.original_name,
        "size": row.size,
        "contentType": row.content_type,
        "uploadedAt": row.uploaded_at.isoformat(),
    }


def cv_storage_url(entry: dict[str, Any] | None) -> Optional[str]:
    if not entry or not storage_service.storage_enabled():
        return None
    path_value = str(entry.get("path", "")).strip()
    if not path_value:
        return None
    if path_value.startswith("http://") or path_value.startswith("https://"):
        return path_value
    if path_value.startswith("cvs/"):
        return storage_service.storage_public_url(path_value)
    remote_path = f"cvs/{Path(path_value).name}"
    if storage_service.storage_object_exists(remote_path):
        return storage_service.storage_public_url(remote_path)
    return None


def get_latest_cv_response(request: Request) -> JSONResponse:
    require_user(request)
    entry = latest_cv_entry()
    if not entry:
        return JSONResponse({"ok": True, "file": None})

    storage_url = cv_storage_url(entry)
    return JSONResponse(
        {
            "ok": True,
            "file": {
                "id": entry.get("id"),
                "originalName": entry.get("originalName", "cv_upload"),
                "size": entry.get("size", 0),
                "contentType": entry.get("contentType", "application/octet-stream"),
                "uploadedAt": entry.get("uploadedAt"),
                "viewUrl": f"/cv/file?id={entry.get('id')}",
                "storageUrl": storage_url,
            },
        }
    )


def get_cv_file_response(id: str, request: Request) -> FileResponse:
    require_user(request)
    entries = load_cv_index()
    entry = next((item for item in entries if item.get("id") == id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="File not found.")

    storage_url = cv_storage_url(entry)
    if storage_url:
        return RedirectResponse(storage_url)

    path = Path(entry.get("path", "")) if entry.get("path") else None
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(
        path,
        media_type=entry.get("contentType", "application/octet-stream"),
        filename=entry.get("originalName", path.name),
    )
