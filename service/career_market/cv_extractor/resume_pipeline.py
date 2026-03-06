import re
import contextlib
import io
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text
from PIL import Image
import numpy as np

from paddleocr import PaddleOCR
from rapidfuzz import fuzz


# ------------------ CONFIG ------------------

DEFAULT_SECTION_HEADERS = {
    "summary": ["summary", "profile", "about", "objective"],
    "skills": [
        "skills",
        "technical skills",
        "technical skills and interests",
        "technical skills & interests",
        "skills and interests",
        "skills & interests",
        "core skills",
        "tools",
        "technologies",
    ],
    "experience": ["experience", "work experience", "employment", "professional experience", "work history"],
    "education": ["education", "academic", "qualifications"],
    "projects": ["projects", "personal projects", "key projects"],
    "certifications": ["certifications", "certificates"],
    "awards": ["awards", "achievements"],
}

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\bhttps?://[^\s]+\b|\bwww\.[^\s]+\b", re.IGNORECASE)
LINKEDIN_RE = re.compile(r"(linkedin\.com/[A-Za-z0-9_\-/%]+)", re.IGNORECASE)
GITHUB_RE = re.compile(r"(github\.com/[A-Za-z0-9_\-/%]+)", re.IGNORECASE)

# A practical phone regex (kept permissive; filtered by digit count later)
PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)\.]{7,}\d)")

SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}

TITLE_PATTERNS = [
    ("software engineer", "Software Engineer"),
    ("software developer", "Software Developer"),
    ("full stack", "Full Stack Developer"),
    ("frontend", "Frontend Developer"),
    ("front end", "Frontend Developer"),
    ("backend", "Backend Developer"),
    ("back end", "Backend Developer"),
    ("devops", "DevOps Engineer"),
    ("site reliability", "Site Reliability Engineer"),
    ("data analyst", "Data Analyst"),
    ("data scientist", "Data Scientist"),
    ("machine learning", "Machine Learning Engineer"),
    ("ml engineer", "Machine Learning Engineer"),
    ("qa engineer", "QA Engineer"),
    ("test engineer", "QA Engineer"),
    ("product manager", "Product Manager"),
    ("project manager", "Project Manager"),
    ("business analyst", "Business Analyst"),
]


@dataclass
class OCRLine:
    text: str
    conf: float


_OCR_INSTANCE: Optional[PaddleOCR] = None


def _get_ocr() -> PaddleOCR:
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        # Silence PaddleOCR startup noise that can break JSON-only stdout.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _OCR_INSTANCE = PaddleOCR(use_angle_cls=True, lang="en")
    return _OCR_INSTANCE


# ------------------ HELPERS ------------------

def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)         # reduce multiple blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)         # normalize spaces
    return text.strip()


# ------------------ EXTRACTION: PDF TEXT ------------------

def extract_pdf_text_direct(pdf_path: str) -> str:
    """
    Extracts text from PDF if selectable/embedded text exists.
    """
    try:
        text = pdfminer_extract_text(pdf_path) or ""
        return text.strip()
    except Exception:
        return ""


# ------------------ EXTRACTION: PDF -> IMAGES ------------------

def render_pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    Renders each page of the PDF to a PIL image for OCR fallback.
    """
    doc = fitz.open(pdf_path)
    images: List[Image.Image] = []

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images


# ------------------ OCR ------------------

def ocr_image(pil_img: Image.Image) -> List[OCRLine]:
    img = np.array(pil_img.convert("RGB"))
    result = _get_ocr().ocr(img, cls=True)

    lines: List[OCRLine] = []
    if not result:
        return lines

    # PaddleOCR result format: list of blocks, each block has items [box, (text, conf)]
    for block in result:
        for item in block:
            text, conf = item[1][0], float(item[1][1])
            text = (text or "").strip()
            if text:
                lines.append(OCRLine(text=text, conf=conf))
    return lines


def ocr_images_to_text(images: List[Image.Image], min_conf: float = 0.50) -> str:
    all_lines: List[str] = []
    for img in images:
        lines = ocr_image(img)
        for ln in lines:
            if ln.conf >= min_conf:
                all_lines.append(ln.text)
    return "\n".join(all_lines).strip()


# ------------------ HYBRID EXTRACTION ------------------

def extract_text_hybrid(path: str, pdf_text_min_chars: int = 400) -> Dict[str, Any]:
    """
    Strategy:
    - If PDF and extracted selectable text is long enough => use it
    - Else OCR (PDF rendered pages / image)
    """
    if is_pdf(path):
        direct = extract_pdf_text_direct(path)
        if len(direct) >= pdf_text_min_chars:
            text = normalize_text(direct)
            return {
                "text": text,
                "method": "pdf_text",
                "quality": {"char_count": len(text), "line_count": text.count("\n") + 1},
            }

        # scanned / image-based PDF fallback
        images = render_pdf_to_images(path)
        ocr_text = ocr_images_to_text(images)
        text = normalize_text(ocr_text)
        return {
            "text": text,
            "method": "ocr_pdf",
            "quality": {"char_count": len(text), "line_count": text.count("\n") + 1},
        }

    # image file
    img = Image.open(path)
    ocr_text = ocr_images_to_text([img])
    text = normalize_text(ocr_text)
    return {
        "text": text,
        "method": "ocr_image",
        "quality": {"char_count": len(text), "line_count": text.count("\n") + 1},
    }


# ------------------ SECTION SEGMENTATION ------------------

def _normalize_header_line(line: str) -> str:
    cleaned = line.strip().lower()
    cleaned = re.sub(r"^[\s•\-\*]+", "", cleaned)
    cleaned = re.sub(r"[:\-\s]+$", "", cleaned)
    return cleaned


def segment_sections(text: str, headers: Dict[str, List[str]] = DEFAULT_SECTION_HEADERS) -> Dict[str, str]:
    """
    Simple header-based segmentation.
    Works best when resumes have clear headings like "Education", "Experience", etc.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    header_map: Dict[str, str] = {}
    for section, variants in headers.items():
        for v in variants:
            header_map[v.lower()] = section

    sections: Dict[str, List[str]] = {"other": []}
    current = "other"
    header_variants = sorted(header_map.keys(), key=len, reverse=True)

    for ln in lines:
        key = _normalize_header_line(ln)
        if key in header_map:
            current = header_map[key]
            sections.setdefault(current, [])
            continue

        matched = None
        remainder = ""
        for variant in header_variants:
            pattern = r"^\s*[•\-\*]*\s*" + re.escape(variant) + r"\b[:\-\s]*(.*)$"
            m = re.match(pattern, ln, flags=re.IGNORECASE)
            if m:
                matched = variant
                remainder = (m.group(1) or "").strip()
                break

        if matched:
            current = header_map[matched]
            sections.setdefault(current, [])
            if remainder:
                sections[current].append(remainder)
            continue

        sections.setdefault(current, [])
        sections[current].append(ln)

    out: Dict[str, str] = {}
    for k, v in sections.items():
        joined = "\n".join(v).strip()
        if joined:
            out[k] = joined
    return out


# ------------------ RULE-BASED FIELDS ------------------

def extract_contacts(text: str) -> Dict[str, Any]:
    emails = sorted(set(EMAIL_RE.findall(text)))
    urls = sorted(set(URL_RE.findall(text)))
    linkedin = sorted(set(LINKEDIN_RE.findall(text)))
    github = sorted(set(GITHUB_RE.findall(text)))

    raw_phones = PHONE_RE.findall(text)
    phones = []
    for p in raw_phones:
        digits = re.sub(r"\D", "", p)
        if 9 <= len(digits) <= 15:
            phones.append(p.strip())
    phones = sorted(set(phones))

    return {
        "emails": emails,
        "phones": phones,
        "urls": urls,
        "linkedin": linkedin,
        "github": github,
    }


def guess_name(text: str) -> Optional[str]:
    """
    Heuristic: first 8 lines, skip contact lines, pick 2-4 tokens mostly alphabetic.
    """
    for ln in text.split("\n")[:8]:
        s = ln.strip()
        if not s:
            continue
        if EMAIL_RE.search(s) or URL_RE.search(s) or PHONE_RE.search(s):
            continue
        tokens = s.split()
        if 2 <= len(tokens) <= 4 and all(re.match(r"^[A-Za-z\.\-']+$", t) for t in tokens):
            return s
    return None


# ------------------ TITLE (HEURISTIC) ------------------

def extract_title(text: str, sections: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Heuristic title extraction from early lines and key sections.
    """
    sections = sections or {}
    chunks = []
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks.extend(lines[:12])
    if sections.get("summary"):
        chunks.append(sections["summary"])
    if sections.get("experience"):
        chunks.append(sections["experience"])
    combined = "\n".join(chunks)

    for needle, label in TITLE_PATTERNS:
        pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
        if re.search(pattern, combined, flags=re.IGNORECASE):
            return label

    return None


# ------------------ SKILLS (DICTIONARY MATCH BASELINE) ------------------

def extract_skills(text: str, skills_list: List[str]) -> List[str]:
    """
    Conservative skills extraction:
    - exact match using boundary-ish regex
    - fuzzy (only for multi-word skills) against lines
    """
    text_l = text.lower()
    found = set()

    for skill in skills_list:
        s = (skill or "").lower().strip()
        if not s:
            continue

        # boundary-ish match
        pattern = r"(?<!\w)" + re.escape(s) + r"(?!\w)"
        if re.search(pattern, text_l):
            found.add(skill)
            continue

        # fuzzy only for multi-word skills
        if " " in s and len(s) >= 6:
            for ln in text_l.split("\n"):
                if fuzz.partial_ratio(s, ln) >= 92:
                    found.add(skill)
                    break

    return sorted(found)


# ------------------ MAIN PARSER ------------------

def parse_resume(path: str, skills_list: Optional[List[str]] = None) -> Dict[str, Any]:
    extraction = extract_text_hybrid(path)
    text = extraction["text"]

    sections = segment_sections(text)
    contacts = extract_contacts(text)
    name = guess_name(text)
    title = extract_title(text, sections)

    skills = []
    if skills_list:
        skills_text = sections.get("skills", "") or text
        skills = extract_skills(skills_text, skills_list)

    return {
        "meta": extraction,
        "name": name,
        "title": title,
        "contacts": contacts,
        "sections": sections,
        "skills": skills,
        "raw_text": text,
    }
