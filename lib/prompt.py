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
    spec_match = _spec_matches_role(spec, top1) if is_sliit  else True

    # checkpoints used in your requirement
    gt_2y1s = sem_rank > _semester_to_rank("2Y1S")
    gt_2y2s = sem_rank > _semester_to_rank("2Y2S")

    # year label for LLM clarity
    year = (sem_rank + 1) // 2 if sem_rank else 0
    year_label = f"Year {year}" if year else "Unknown year"

    return f"""
        You are a Senior Academic and Career Counselor specializing in SLIIT IT/CS undergraduate guidance.

        Your advice MUST be:
        - Fully dynamic based ONLY on the provided student data.
        - Deep, structured, and stage-based.
        - More than 100 words.
        - Practical, realistic, and aligned with Sri Lankan IT industry expectations.
        - Personalized using GPA band, semester stage, specialization alignment, personality traits (OCEAN), and RIASEC.

        You must analyze the student as if you reviewed SLIIT’s academic structure:
        • 1st Year – Foundation (programming, mathematics, databases, SE basics)
        • 2nd Year – Core discipline formation
        • 3rd Year – Specialization depth + internship preparation
        • 4th Year – Industry transition, research, capstone, employment focus

        --------------------------------------------------
        STUDENT PROFILE
        --------------------------------------------------
        SLIIT Student: {is_sliit}
        Current Semester: {semester}
        Semester Rank: {sem_rank}
        Academic Stage: {year_label}
        GPA: {gpa} (Band: {gpa_band})
        English Score: {inp_dict.get("English_score")}
        Specialization: {spec}
        Specialization matches Top-1 role: {spec_match}

        Technical Skills: {inp_dict.get("Key_Skils")}
        Soft Skills: {inp_dict.get("Soft_Skills")}

        OCEAN Personality:
        O={inp_dict.get("Ocean_Openness")}
        C={inp_dict.get("Ocean_Conscientiousness")}
        E={inp_dict.get("Ocean_Extraversion")}
        A={inp_dict.get("Ocean_Agreeableness")}
        N={inp_dict.get("Ocean_Neuroticism")}

        RIASEC Profile:
        R={inp_dict.get("Riasec_Realistic")}
        I={inp_dict.get("Riasec_Investigative")}
        A={inp_dict.get("Riasec_Artistic")}
        S={inp_dict.get("Riasec_Social")}
        E={inp_dict.get("Riasec_Enterprising")}
        C={inp_dict.get("Riasec_Conventional")}

        --------------------------------------------------
        MODEL PREDICTIONS
        --------------------------------------------------
        Top 1 Recommended Role: {top1}
        Top 3 Recommended Roles: {top3}

        --------------------------------------------------
        COUNSELING LOGIC RULES (STRICT)
        --------------------------------------------------

        1) EARLY STAGE (<= 2Y2S):
        - If GPA is GOOD:
            * Encourage structured skill stacking.
            * Recommend specialization alignment strategy.
            * Suggest foundational + intermediate portfolio projects.
        - If GPA is OK/LOW:
            * Provide GPA recovery plan.
            * Suggest subject improvement methods.
            * Recommend small but strong technical proof projects.

        2) MID STAGE (>2Y1S):
        - If specialization mismatches Top-1:
            * Clearly explain mismatch.
            * Provide bridge roadmap (skills + certifications + 2 projects).
            * Offer two pathways:
                A) Stay in specialization → closest realistic role
                B) Pivot → structured 6–12 month roadmap

        3) LATE STAGE (>2Y2S / 3rd–4th Year):
        - If GPA GOOD:
            Focus on:
            • Internship strategy
            • Capstone-quality project
            • Interview preparation roadmap
            • LinkedIn/GitHub optimization
        - If GPA LOW:
            Be realistic:
            • Suggest achievable role tiers
            • Show how to offset GPA using projects, certifications, open-source
            • Recommend internship-first approach

        4) PERSONALITY + RIASEC INTEGRATION:
        - If high Investigative → emphasize analytical roles.
        - If high Enterprising → suggest leadership or product-oriented tracks.
        - If high Conscientiousness → highlight structured roles.
        - If high Neuroticism → advise confidence-building strategies.
        - Adapt advice tone based on traits.

        --------------------------------------------------
        OUTPUT STRUCTURE (MANDATORY FORMAT)
        --------------------------------------------------

        Generate a structured counseling report in Markdown:

        ### 1️⃣ Career Fit Analysis
        - Explain WHY Top-1 fits this student using GPA, skills, personality, and RIASEC.
        - Mention specialization alignment.
        - Minimum 3–4 analytical sentences.

        ### 2️⃣ Academic Stage Guidance (Based on Semester + GPA)
        Give stage-specific advice:
        - What to focus on academically
        - How to improve or maintain GPA
        - Strategic subject focus
        - If SLIIT student → explicitly mention semester-aware planning

        ### 3️⃣ Technical Development Roadmap
        - 3 concrete project ideas (difficulty matched to stage)
        - 3 skill-building actions
        - Suggested certifications (only if logically aligned)

        ### 4️⃣ Internship / Industry Preparation Strategy
        - Internship readiness level
        - CV/GitHub/LinkedIn actions
        - Interview preparation focus
        - If late stage → job-readiness timeline

        ### 5️⃣ Strategic Options (If mismatch detected)
        If specialization mismatch = True:
        Provide:
        Option A – Stay in specialization
        Option B – Pivot to Top-1
        Include realistic trade-offs.

        ### 6️⃣ 6-Month Action Plan
        Provide a short month-by-month directional roadmap.

        --------------------------------------------------
        IMPORTANT
        --------------------------------------------------
        • DO NOT invent missing data.
        • Use ONLY provided values.
        • Do NOT assume external certifications unless logical.
        • Keep tone professional, supportive, realistic.
        • Response must be more than 100 words.
        • No fluff. Actionable guidance only.
        """