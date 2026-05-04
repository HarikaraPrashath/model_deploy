# services/prediction_service.py
import pandas as pd
import joblib
import os
from model_schema.schema import StudentInput

# ------------------------
# Load Model & Encoder (only once)
# ------------------------
model_path = "models/Career_Prediction/career_model-random.joblib"
encoder_path = "models/Career_Prediction/label_encoder-random.joblib"

if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError("Model or encoder not found!")

rf_pipeline = joblib.load(model_path)
label_enc = joblib.load(encoder_path)


# ------------------------
# Convert Frontend JSON → Model Input
# ------------------------
def frontend_to_model_input(data: StudentInput) -> pd.DataFrame:
    row = {
        "student_Programming_learned": ", ".join(data.technicalSkills.programming),
        "student_freamwork_know": ", ".join(data.technicalSkills.frameworks),
        "student_Databases_Tools": ", ".join(data.technicalSkills.databases),
        "student_Cloud_Platforms_Infra_Tools": ", ".join(data.technicalSkills.cloudPlatforms),
        "student_expectation": "",
        "gender": data.personalInfo.gender,
        "languages_selected": ", ".join(data.personalInfo.languages),
        "Education Level": data.academicBackground.educationLevel,
        "major/ field of study": data.academicBackground.major,
        "current_year": str(data.academicBackground.currentYear),
        "current_semester": str(data.academicBackground.currentSemester),
        "learning_style": data.career.learningStyle,
        "preferred_work_environment": data.careerInterests.workEnvironment,
        "work_life_balance": data.careerInterests.workLifeBalance,
        "internship_experience": data.career.internship,
        "certifications": data.career.certifications,
        "mind_stress_management": data.career.stressManagement,
        "cgpa": data.academicBackground.gpa,
        "experties_major_gpa": data.academicBackground.gpa,
        "real_world_projects_completed": data.career.projects
    }

    return pd.DataFrame([row])


# ------------------------
# Main Prediction Service Function
# ------------------------
def predict_career_service(student: StudentInput):
    try:
        model_input = frontend_to_model_input(student)

        pred_encoded = rf_pipeline.predict(model_input)
        pred_label = label_enc.inverse_transform(pred_encoded)

        print(f"🎯 Predicted Career: {pred_label[0]}")

        return {"predicted_career": pred_label[0]}

    except Exception as e:
        print("❌ Prediction Error:", e)
        return {"error": str(e)}