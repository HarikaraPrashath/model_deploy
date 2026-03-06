from __future__ import annotations

import json
import mimetypes
import os
import sys
from pathlib import Path


def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and value and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: upload_analysis_output.py <folder_name>")
        return 2

    folder_name = sys.argv[1].strip()
    if not folder_name:
        print("Missing folder name")
        return 2

    root = Path(__file__).resolve().parents[3]
    analysis_output = root / "service" / "career_market" / "storage" / "analysis_output"
    folder = analysis_output / folder_name
    report_path = analysis_output / f"_supabase_upload_report_{folder_name}.json"

    _load_env(root / ".env")

    sys.path.insert(0, str(root))
    from service.career_market import storage_service  # pylint: disable=import-error

    if not folder.exists():
        report = {
            "ok": False,
            "error": "folder not found",
            "uploaded": 0,
            "failed": 0,
            "items": [],
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
        return 1

    if not storage_service.storage_enabled():
        report = {
            "ok": False,
            "error": "Supabase storage not enabled (missing env vars)",
            "uploaded": 0,
            "failed": 0,
            "items": [],
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
        return 1

    files = [p for p in folder.rglob("*") if p.is_file()]
    items = []
    uploaded = 0
    failed = 0

    for file_path in files:
        rel = file_path.relative_to(analysis_output)
        remote_path = f"analysis_output/{rel.as_posix()}"
        ctype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        try:
            content = file_path.read_bytes()
            url = storage_service.upload_bytes_to_storage(content, remote_path, ctype)
            if url:
                uploaded += 1
                items.append({"file": str(rel), "remote": remote_path, "ok": True, "url": url})
            else:
                failed += 1
                items.append(
                    {"file": str(rel), "remote": remote_path, "ok": False, "error": "upload_failed"}
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            failed += 1
            items.append({"file": str(rel), "remote": remote_path, "ok": False, "error": str(exc)})

    report = {"ok": failed == 0, "uploaded": uploaded, "failed": failed, "items": items}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
