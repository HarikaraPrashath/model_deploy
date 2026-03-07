import pandas as pd
import numpy as np
import joblib

from lib.prompt import build_guidance_prompt, get_groq_client

# =========================
# Model Loading
# =========================
MODEL_PATH = "models/Career_Guide/career_prediction_model.joblib"
ENC_PATH   = "models/Career_Guide/career_label_encoder.joblib"

model = joblib.load(MODEL_PATH)
label_enc = joblib.load(ENC_PATH)


# =========================
# Dynamic Suggestions
# =========================
def build_dynamic_suggestions(inp_dict: dict, top1: str, top3: list[str]):
    is_sliit = bool(inp_dict.get("Is_Sliit_Student"))
    semester = str(inp_dict.get("Current_semester") or "")
    gpa = float(inp_dict.get("GPA") or 0)
    spec = str(inp_dict.get("Specialization") or "")

    from lib.prompt import _semester_to_rank, _spec_matches_role, _gpa_band

    sem_rank = _semester_to_rank(semester)
    gpa_band = _gpa_band(gpa)
    spec_match = _spec_matches_role(spec, top1) if is_sliit else True

    over_2y2s = sem_rank > _semester_to_rank("2Y2S")
    over_2y1s = sem_rank > _semester_to_rank("2Y1S")

    if is_sliit:
        if not over_2y2s:
            if gpa_band == "good":
                title = "✅ SLIIT Plan (Early stage + strong GPA)"
                bullets = [
                    f"Stay consistent toward **{top1}** (pick 1 roadmap).",
                    "Do 1 mini project per month (small but complete).",
                    "Strengthen fundamentals (DSA + core modules weekly).",
                ]
            else:
                title = "⚠️ SLIIT Recovery Plan (Early stage + GPA needs boost)"
                bullets = [
                    "Fix 1 weak subject at a time (weekly plan + past papers).",
                    "Build 1–2 simple projects to show proof (GitHub).",
                    f"Move step-by-step toward **{top1}** (don’t switch too often).",
                ]
        else:
            if spec_match:
                title = "✅ SLIIT Plan (Later stage + specialization aligned)"
                bullets = [
                    "Build an internship-ready project (auth + DB + deploy).",
                    "Prepare CV + LinkedIn + GitHub portfolio.",
                    "Practice interviews (DSA + role fundamentals).",
                ]
            else:
                title = "⚠️ SLIIT Mismatch (Specialization vs predicted role)"
                bullets = [
                    f"Option A: stay in **{spec}** and target a close role in Top-3: **{top3[1]} / {top3[2]}**",
                    f"Option B: pivot to **{top1}** using 1 bridge project + 1 certification.",
                    "Pick 1 option and execute for 6–8 weeks (no zig-zag).",
                ]

        return {
            "audience": "sliit",
            "semester": semester,
            "gpa_band": gpa_band,
            "specialization": spec,
            "specialization_matches_top1": spec_match,
            "title": title,
            "bullets": bullets,
            "modules": [
                "DSA: revise weekly + solve 5 problems/week",
                "Database: SQL + normalization + build 1 schema for your project",
                "Software Engineering: UML + design patterns + clean architecture",
                "Networking/Security: basics + hands-on labs depending on your path",
            ],
        }

    return {
        "audience": "non_sliit",
        "title": "✅ Non-Student Plan (Portfolio-first)",
        "bullets": [
            f"Pick **{top1}** and follow one roadmap (4–6 weeks).",
            "Build 1 portfolio project and deploy it (Vercel/Render).",
            "Apply weekly + practice interviews (fundamentals).",
        ],
    }


# =========================
# Prediction Logic
# =========================
def run_prediction(inp):
    inp_dict = inp.dict()

    df = pd.DataFrame([inp_dict]).drop(
        columns=["Is_Sliit_Student", "Specialization"],
        errors="ignore"
    )

    probs = model.predict_proba(df)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_labels = label_enc.inverse_transform(top3_idx).tolist()
    top1 = top3_labels[0]

    prompt = build_guidance_prompt(inp_dict, top1, top3_labels)

    try:
        client = get_groq_client()
        llm = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful career guidance assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=250,
        )
        guidance = llm.choices[0].message.content
    except Exception:
        guidance = (
            "✅ **Guidance:** Focus on building 1 strong project and improving your core skills.\n"
            "• Pick one roadmap (based on Top-1)\n"
            "• Build a portfolio project\n"
            "• Practice interviews + fundamentals"
        )

    dynamic = build_dynamic_suggestions(inp_dict, top1, top3_labels)

    return {
        "top_1_prediction": top1,
        "top_3_predictions": top3_labels,
        "guidance": guidance,
        "dynamic_suggestions": dynamic,
    }