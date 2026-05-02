# Parse TopJobs-scraped ads (text + image OCR optional), extract skills, and compute
# research-oriented matching with must-have gating + weighted scoring.

from __future__ import annotations

import json
import math
import os
import re
import sys
import base64
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

CV_EXTRACTOR_DIR = Path(__file__).resolve().parents[1] / "cv_extractor"
if str(CV_EXTRACTOR_DIR) not in sys.path:
    sys.path.insert(0, str(CV_EXTRACTOR_DIR))

from skill_config import EXTRA_SIGNALS, SECTION_HEADERS, SKILL_LEXICON

try:
    from rapidfuzz import fuzz, process
except Exception:
    fuzz = None
    process = None


WEIGHTS = {
    "must_have": 0.45,
    "core": 0.35,
    "nice_to_have": 0.15,
    "signals": 0.05,
}

MUST_HAVE_SECTION_HINTS = (
    "requirement",
    "must",
    "core skill",
    "technical skill",
    "what you bring",
    "qualification",
)

NICE_TO_HAVE_SECTION_HINTS = (
    "nice to have",
    "nice-to-have",
    "preferred",
    "bonus",
    "plus",
    "good to have",
)

FUZZY_ALIAS_THRESHOLD = 92


def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace("\u00a0", " ")
    # Keep characters used in technical skills such as c#, c++, .net, pl/sql.
    s = re.sub(r"[^\w\s\.\+\-/#]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_alias_index() -> tuple[Dict[str, str], List[str]]:
    alias_to_skill: Dict[str, str] = {}
    for canonical, aliases in SKILL_LEXICON.items():
        for alias in [canonical] + aliases:
            key = normalize_text(alias)
            if key and key not in alias_to_skill:
                alias_to_skill[key] = canonical
    ordered_aliases = sorted(alias_to_skill.keys(), key=len, reverse=True)
    return alias_to_skill, ordered_aliases


ALIAS_TO_SKILL, ORDERED_ALIASES = _build_alias_index()


def canonicalize_skill(skill: str) -> str:
    norm = normalize_text(skill)
    if not norm:
        return ""
    if norm in ALIAS_TO_SKILL:
        return ALIAS_TO_SKILL[norm]
    if len(norm) >= 4 and process is not None and fuzz is not None:
        best = process.extractOne(norm, ORDERED_ALIASES, scorer=fuzz.ratio)
        if best and best[1] >= FUZZY_ALIAS_THRESHOLD:
            return ALIAS_TO_SKILL[best[0]]
    return norm


def _skill_in_text(skill: str, text: str) -> bool:
    if not text:
        return False
    skill_aliases = SKILL_LEXICON.get(skill, [skill])
    norm_text = normalize_text(text)
    for alias in [skill] + skill_aliases:
        alias_norm = normalize_text(alias)
        if not alias_norm:
            continue
        if alias_norm == "c":
            # Avoid counting c# / c++ as plain c.
            pattern = r"(?<![a-z0-9])c(?![a-z0-9#+])"
        else:
            pattern = rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])"
        if re.search(pattern, norm_text):
            return True
    return False


def find_skills(text: str) -> List[str]:
    t = normalize_text(text)
    found = set()
    for skill, aliases in SKILL_LEXICON.items():
        for alias in [skill] + aliases:
            alias_norm = normalize_text(alias)
            if alias_norm == "c":
                pattern = r"(?<![a-z0-9])c(?![a-z0-9#+])"
            else:
                pattern = rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])"
            if re.search(pattern, t):
                found.add(skill)
                break
    return sorted(found)


def find_signals(text: str) -> List[str]:
    t = normalize_text(text)
    found = set()
    for signal, aliases in EXTRA_SIGNALS.items():
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if alias_norm in t:
                found.add(signal)
                break
    return sorted(found)


def split_sections(raw: str) -> Dict[str, str]:
    t = raw.replace("\r", "\n")
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    joined = "\n".join(lines)
    hdrs = sorted(set(SECTION_HEADERS), key=len, reverse=True)
    pattern = r"(?i)\b(" + "|".join(re.escape(h) for h in hdrs) + r")\b"
    matches = list(re.finditer(pattern, joined))

    if not matches:
        return {"full_text": joined}

    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(joined)
        title = m.group(1).lower().strip()
        body = joined[m.end():end].strip(" :\n\t-")
        sections[title] = body
    sections["full_text"] = joined
    return sections


def _section_block(sections: Dict[str, str], hints: tuple[str, ...]) -> str:
    parts: List[str] = []
    for key, val in sections.items():
        k = normalize_text(key)
        if any(h in k for h in hints):
            parts.append(val)
    return "\n".join(parts)


def infer_skill_priority(job: "JobParsed") -> tuple[List[str], List[str], List[str]]:
    must_text = _section_block(job.sections, MUST_HAVE_SECTION_HINTS)
    nice_text = _section_block(job.sections, NICE_TO_HAVE_SECTION_HINTS)
    must_skills = set()
    nice_skills = set()
    all_skills = set(normalize_skill_list(job.skills))

    for skill in all_skills:
        if must_text and _skill_in_text(skill, must_text):
            must_skills.add(skill)
        if nice_text and _skill_in_text(skill, nice_text):
            nice_skills.add(skill)

    nice_skills -= must_skills
    core_skills = all_skills - nice_skills
    if not core_skills and all_skills:
        core_skills = set(all_skills)

    return sorted(must_skills), sorted(nice_skills), sorted(core_skills)


@dataclass
class JobParsed:
    ref: str
    position: str
    employer: str
    url: str
    ad_type: str
    files: List[str]
    raw_text: str
    sections: Dict[str, str]
    skills: List[str]
    signals: List[str]


def ocr_image_to_text(image_path: str) -> Optional[str]:
    text: Optional[str] = None
    try:
        import pytesseract
        from PIL import Image

        tess_cmd = os.environ.get("TESSERACT_CMD")
        if tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
        else:
            default_tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(default_tess):
                pytesseract.pytesseract.tesseract_cmd = default_tess
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="eng")
    except Exception:
        text = None

    if text and text.strip():
        return text

    try:
        tess_cmd = os.environ.get("TESSERACT_CMD") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tess_cmd):
            proc = subprocess.run(
                [tess_cmd, image_path, "stdout", "-l", "eng", "--psm", "6"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=60,
                check=False,
            )
            cli_text = (proc.stdout or "").strip()
            if cli_text:
                return cli_text
    except Exception:
        pass

    # Fallback to the OCR stack already used by the CV extractor.
    try:
        from resume_pipeline import extract_text_hybrid

        result = extract_text_hybrid(image_path)
        fallback_text = str(result.get("text", "") or "").strip()
        if fallback_text:
            return fallback_text
    except Exception:
        pass

    if str(os.environ.get("ENABLE_JOB_IMAGE_VISION_FALLBACK", "")).lower() not in ("1", "true", "yes"):
        return None

    try:
        import requests

        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")

        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip().rstrip("/")
        model = os.environ.get("OLLAMA_VISION_MODEL", "granite3.2-vision:2b").strip()
        prompt = (
            "Extract all readable text from this job advertisement image. "
            "Return only the text content with line breaks preserved. "
            "Do not summarize or explain."
        )
        resp = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [encoded],
                "stream": False,
            },
            timeout=90,
        )
        if resp.status_code != 200:
            return None
        data = resp.json() if resp.content else {}
        vision_text = str(data.get("response", "") or "").strip()
        return vision_text or None
    except Exception:
        return None


def load_topjobs_metadata(folder: str) -> List[dict]:
    meta_path = os.path.join(folder, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text_if_exists(folder: str, files: List[str]) -> str:
    for fn in files:
        if fn.lower().endswith("_content.txt") or fn.lower().endswith(".txt"):
            p = os.path.join(folder, fn)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
    return ""


def read_image_ocr_if_exists(folder: str, files: List[str]) -> Optional[str]:
    for fn in files:
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            p = os.path.join(folder, fn)
            if os.path.exists(p):
                return ocr_image_to_text(p)
    return None


def parse_jobs(scraped_folder: str, enable_ocr: bool = False) -> List[JobParsed]:
    meta = load_topjobs_metadata(scraped_folder)
    parsed: List[JobParsed] = []

    for job in meta:
        raw = read_text_if_exists(scraped_folder, job.get("files", []))
        if not raw:
            ocr_text = read_image_ocr_if_exists(scraped_folder, job.get("files", []))
            if ocr_text:
                raw = ocr_text
        elif enable_ocr:
            ocr_text = read_image_ocr_if_exists(scraped_folder, job.get("files", []))
            if ocr_text and ocr_text.strip():
                raw = f"{raw}\n\n{ocr_text}"

        raw = raw or ""
        match_text = " ".join([raw, job.get("position", ""), job.get("employer", "")]).strip()
        sections = split_sections(match_text or raw)
        skills = normalize_skill_list(find_skills(match_text))
        signals = find_signals(match_text)

        parsed.append(
            JobParsed(
                ref=job.get("ref", ""),
                position=job.get("position", ""),
                employer=job.get("employer", ""),
                url=job.get("url", ""),
                ad_type=job.get("type", ""),
                files=job.get("files", []),
                raw_text=raw,
                sections=sections,
                skills=skills,
                signals=signals,
            )
        )
    return parsed


def normalize_skill_list(skills: List[str]) -> List[str]:
    out = set()
    for skill in skills:
        if not skill or not str(skill).strip():
            continue
        can = canonicalize_skill(str(skill))
        if can:
            out.add(can)
    return sorted(out)


def excerpt_text(text: str, limit: int = 700) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def full_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _idf_weights(jobs: List[JobParsed]) -> Dict[str, float]:
    doc_freq: Dict[str, int] = {}
    for job in jobs:
        for skill in set(normalize_skill_list(job.skills)):
            doc_freq[skill] = doc_freq.get(skill, 0) + 1
    total = max(len(jobs), 1)
    return {
        skill: math.log((total + 1) / (freq + 1)) + 1.0
        for skill, freq in doc_freq.items()
    }


def _weighted_coverage(job_skills: set[str], user_skills: set[str], idf: Dict[str, float]) -> float:
    if not job_skills:
        return 0.0
    denom = sum(idf.get(skill, 1.0) for skill in job_skills)
    if denom <= 0:
        return 0.0
    numer = sum(idf.get(skill, 1.0) for skill in job_skills if skill in user_skills)
    return numer / denom


def score_job(
    job: JobParsed,
    user_skills: List[str],
    user_signals: set[str],
    idf: Dict[str, float],
) -> Dict[str, object]:
    job_skills = set(normalize_skill_list(job.skills))
    user_skill_set = set(normalize_skill_list(user_skills))
    overlap = sorted(job_skills & user_skill_set)
    missing = sorted(job_skills - user_skill_set)
    baseline = ((len(overlap) / len(job_skills)) * 100.0) if job_skills else 0.0

    must_have, nice_to_have, core = infer_skill_priority(job)
    must_set = set(must_have)
    nice_set = set(nice_to_have)
    core_set = set(core)

    matched_must = sorted(must_set & user_skill_set)
    missing_must = sorted(must_set - user_skill_set)
    matched_nice = sorted(nice_set & user_skill_set)

    must_cov = (len(matched_must) / len(must_set)) if must_set else 0.0
    core_cov = _weighted_coverage(core_set, user_skill_set, idf) if core_set else 0.0
    nice_cov = (len(matched_nice) / len(nice_set)) if nice_set else 0.0
    signal_cov = (
        len(set(job.signals) & user_signals) / len(set(job.signals))
        if job.signals
        else 0.0
    )

    gate_pass = True
    gate_reason = "No explicit must-have section found."
    if must_set:
        required_hits = max(1, math.ceil(len(must_set) * 0.5))
        gate_pass = len(matched_must) >= required_hits
        if gate_pass:
            gate_reason = f"Matched {len(matched_must)}/{len(must_set)} must-have skills."
        else:
            gate_reason = (
                f"Matched {len(matched_must)}/{len(must_set)} must-have skills. "
                f"Need at least {required_hits}."
            )

    active_weights = {
        "must_have": WEIGHTS["must_have"] if must_set else 0.0,
        "core": WEIGHTS["core"] if core_set else 0.0,
        "nice_to_have": WEIGHTS["nice_to_have"] if nice_set else 0.0,
        "signals": WEIGHTS["signals"] if job.signals else 0.0,
    }
    total_active_weight = sum(active_weights.values())
    if total_active_weight > 0:
        weighted_raw = (
            active_weights["must_have"] * must_cov
            + active_weights["core"] * core_cov
            + active_weights["nice_to_have"] * nice_cov
            + active_weights["signals"] * signal_cov
        ) / total_active_weight * 100.0
    else:
        weighted_raw = 0.0
    if not gate_pass:
        weighted_raw *= 0.55

    final_score = round((0.2 * baseline) + (0.8 * weighted_raw), 2)
    explanations = [
        f"Matched {len(overlap)} of {len(job_skills)} extracted skills.",
        gate_reason,
    ]
    if missing_must:
        explanations.append(f"Critical gaps: {', '.join(missing_must[:5])}")

    return {
        "match_percent": final_score,
        "baseline_match_percent": round(baseline, 2),
        "overlap": overlap,
        "missing": missing,
        "job_skill_count": len(job_skills),
        "user_skill_count": len(user_skill_set),
        "must_have_skills": must_have,
        "nice_to_have_skills": nice_to_have,
        "core_skills": core,
        "matched_must_have": matched_must,
        "missing_must_have": missing_must,
        "must_have_gate_pass": gate_pass,
        "matched_nice_to_have": matched_nice,
        "weighted_components": {
            "must_have_coverage": round(must_cov, 4),
            "core_weighted_coverage": round(core_cov, 4),
            "nice_to_have_coverage": round(nice_cov, 4),
            "signal_coverage": round(signal_cov, 4),
        },
        "explanations": explanations,
    }


def rank_jobs(jobs: List[JobParsed], user_skills: List[str]) -> List[Dict[str, object]]:
    user_skills = normalize_skill_list(user_skills)
    user_signals = set(find_signals(" ".join(user_skills)))
    idf = _idf_weights(jobs)

    ranked: List[Dict[str, object]] = []
    for job in jobs:
        score = score_job(job, user_skills, user_signals, idf)
        ranked.append(
            {
                "ref": job.ref,
                "position": job.position,
                "employer": job.employer,
                "url": job.url,
                "text_excerpt": excerpt_text(job.raw_text),
                "text_full": full_text(job.raw_text),
                "skills_found": job.skills,
                "signals_found": job.signals,
                **score,
            }
        )
    ranked.sort(
        key=lambda item: (item.get("match_percent", 0), item.get("baseline_match_percent", 0)),
        reverse=True,
    )
    return ranked


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scraped_folder", required=True, help="Path to folder containing metadata.json + downloaded files")
    ap.add_argument("--user_skills", default="", help="Comma-separated skills. Example: 'react,javascript,html,css,rest api'")
    ap.add_argument("--user_skills_json", default="", help="Optional path to JSON containing skills list under key 'skills'")
    ap.add_argument("--enable_ocr", action="store_true", help="Try OCR for image ads (requires pytesseract installed)")
    ap.add_argument("--out_json", default="job_ranked.json", help="Output JSON file name")
    args = ap.parse_args()

    user_skills: List[str] = []
    if args.user_skills.strip():
        user_skills = [s.strip() for s in args.user_skills.split(",") if s.strip()]

    if args.user_skills_json.strip():
        with open(args.user_skills_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("skills"), list):
            user_skills.extend([str(x) for x in data["skills"]])

    user_skills = normalize_skill_list(user_skills)
    jobs = parse_jobs(args.scraped_folder, enable_ocr=args.enable_ocr)
    ranked = rank_jobs(jobs, user_skills)

    out_path = os.path.join(args.scraped_folder, args.out_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2, ensure_ascii=False)

    print("\nTop matches:")
    for i, row in enumerate(ranked[:10], 1):
        print(f"{i:2d}. {row['match_percent']:6.2f}% | {row['position']} - {row['employer']} ({row['ref']})")
        skills_preview = ", ".join(row.get("skills_found", [])[:15])
        print(f"    skills: {skills_preview or '(none found)'}")
        if row.get("missing"):
            print("    missing:", ", ".join(row["missing"][:10]) + (" ..." if len(row["missing"]) > 10 else ""))

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()


