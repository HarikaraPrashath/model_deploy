import os
import re
from groq import Groq

def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Set it in env or .env file.")
    return Groq(api_key=api_key)

def _semester_to_rank(sem: str) -> int:
    """
    Convert '1Y1S'..'4Y2S' into sortable int rank.
    Returns 0 if unknown.
    """
    if not sem:
        return 0
    sem = sem.strip().upper().replace(" ", "")
    m = re.match(r"(\d)Y(\d)S", sem)
    if not m:
        return 0
    y = int(m.group(1))
    s = int(m.group(2))
    if y < 1 or y > 4 or s < 1 or s > 2:
        return 0
    return (y - 1) * 2 + s  # 1Y1S=1 ... 4Y2S=8

def _gpa_band(gpa: float) -> str:
    if gpa is None:
        return "unknown"
    if gpa >= 3.2:
        return "good"
    if gpa >= 2.7:
        return "ok"
    return "low"

def _normalize(text: str) -> str:
    return (text or "").strip().lower()

def _spec_matches_role(spec: str, role: str) -> bool:
    """
    Heuristic match between SLIIT specialization and predicted role.
    Adjust keywords to match your dataset labels.
    """
    spec_n = _normalize(spec)
    role_n = _normalize(role)

    buckets = {
        "software engineering": ["software", "backend", "full stack", "frontend", "mobile", "devops", "qa", "sdet"],
        "cyber security": ["security", "soc", "pentest", "forensics", "network security", "cyber"],
        "data science": ["data", "ml", "ai", "analytics", "bi", "data engineer"],
        "information systems": ["business analyst", "ba", "erp", "product", "project manager", "it auditor"],
        "networking": ["network", "cloud", "sysadmin", "infrastructure"],
        "game development": ["game", "unity", "unreal"],
    }

    # find bucket for spec
    spec_bucket = None
    for k in buckets:
        if k in spec_n:
            spec_bucket = k
            break

    if not spec_bucket:
        # if unknown specialization, avoid false mismatch
        return True

    return any(kw in role_n for kw in buckets[spec_bucket])

def build_guidance_prompt(inp_dict: dict, top1: str, top3: list[str]) -> str:
    is_sliit = bool(inp_dict.get("Is_Sliit_Student"))
    semester = str(inp_dict.get("Current_semester") or "")
    sem_rank = _semester_to_rank(semester)
    gpa = inp_dict.get("GPA")
    gpa_band = _gpa_band(gpa if isinstance(gpa, (int, float)) else None)

    spec = str(inp_dict.get("Specialization") or "")
    spec_match = _spec_matches_role(spec, top1) if is_sliit else True

    # checkpoints used in your requirement
    gt_2y1s = sem_rank > _semester_to_rank("2Y1S")
    gt_2y2s = sem_rank > _semester_to_rank("2Y2S")

    # year label for LLM clarity
    year = (sem_rank + 1) // 2 if sem_rank else 0
    year_label = f"Year {year}" if year else "Unknown year"

    return f"""
You are a career advisor for IT/CS students and junior developers.

Student profile:
- Is SLIIT student: {is_sliit}
- Current semester: {semester} (rank={sem_rank}, {year_label})
- GPA: {gpa} (band={gpa_band})
- English score: {inp_dict.get("English_score")}
- Specialization (if any): {spec}
- Soft skills: {inp_dict.get("Soft_Skills")}
- Technical skills: {inp_dict.get("Key_Skils")}
- OCEAN: O={inp_dict.get("Ocean_Openness")}, C={inp_dict.get("Ocean_Conscientiousness")}, E={inp_dict.get("Ocean_Extraversion")}, A={inp_dict.get("Ocean_Agreeableness")}, N={inp_dict.get("Ocean_Neuroticism")}
- RIASEC: R={inp_dict.get("Riasec_Realistic")}, I={inp_dict.get("Riasec_Investigative")}, A={inp_dict.get("Riasec_Artistic")}, S={inp_dict.get("Riasec_Social")}, E={inp_dict.get("Riasec_Enterprising")}, C={inp_dict.get("Riasec_Conventional")}

Model predictions:
- Top 1: {top1}
- Top 3: {top3}

Special rules for SLIIT students (IMPORTANT):
1) If semester <= 2Y2S:
   - If GPA is good: encourage continuing towards Top-1 and recommend a suitable specialization path.
   - If GPA is ok/low: give a recovery plan (study strategy + retake/boost plan + small portfolio projects).
2) If semester > 2Y1S and specialization mismatches Top-1:
   - Explain mismatch briefly and give a skill/project bridge plan to align Top-1 with the current specialization.
3) If semester > 2Y2S (3rd/4th year):
   - If GPA is good: focus on internship readiness, capstone-quality project, interview prep.
   - If GPA is low: be realistic—suggest roles/paths still achievable, how to build proof via projects, certifications, open-source, and strong internship/portfolio to offset GPA.
4) Compare specialization vs predicted role:
   - Specialization matches Top-1? {spec_match}
   - If mismatch and semester is later: propose 2 options:
     (A) stay in specialization and target a close role
     (B) pivot to Top-1 with a bridge roadmap

Output requirements:
Generate a short actionable guidance message (max 10 lines) in Markdown.
Must include:
1) A 1-line summary: why Top-1 fits
2) 3 bullet next-steps (projects + learning)
3) If SLIIT student: include semester-aware advice based on GPA and year (1st/2nd vs 3rd/4th)
Be friendly, direct, and practical.
""".strip()