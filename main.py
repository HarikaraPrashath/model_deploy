from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

# Load saved model and label encoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

xgb_clf = joblib.load(os.path.join(
    BASE_DIR, "models", "career_prediction_xgboost_updated.joblib"
))
label_enc = joblib.load(os.path.join(
    BASE_DIR, "models", "career_label_encoder_updated.joblib"
))

app = FastAPI()

origins = [
    o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ HOME / HEALTH CHECK FUNCTION
@app.get("/")
def home():
    return {
        "status": "success",
        "message": "🚀 Career Prediction API deployed successfully!",
        "cors_allowed_origins": origins
    }

# ------------------ MODEL INPUT ------------------
class StudentData(BaseModel):
    Soft_Skills: str = ""
    Key_Skils: str = ""
    Current_semester: str
    Learning_Style: str = "Unknown"

    GPA: float
    English_score: float

    Ocean_Openness: float
    Ocean_Conscientiousness: float
    Ocean_Extraversion: float
    Ocean_Agreeableness: float
    Ocean_Neuroticism: float

    Riasec_Realistic: float
    Riasec_Investigative: float
    Riasec_Artistic: float
    Riasec_Social: float
    Riasec_Enterprising: float
    Riasec_Conventional: float

# ------------------ PREDICT ------------------
@app.post("/predict")
def predict(student: StudentData):
    df = pd.DataFrame([student.model_dump()])
    pred_encoded = xgb_clf.predict(df)[0]
    pred_label = label_enc.inverse_transform([pred_encoded])[0]
    return {"predicted_career": pred_label}
