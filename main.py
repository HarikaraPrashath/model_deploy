from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

# -------------------------------------------------
# LOAD MODEL FILES FROM ROOT DIRECTORY
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "career_prediction_xgboost_updated.joblib"
)
ENCODER_PATH = os.path.join(
    BASE_DIR, "career_label_encoder_updated.joblib"
)

xgb_clf = joblib.load(MODEL_PATH)
label_enc = joblib.load(ENCODER_PATH)

# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------
app = FastAPI(title="Career guide")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# HEALTH CHECK (VERY IMPORTANT)
# -------------------------------------------------
@app.get("/","include_in_schema=False)
def index():
    return {
       RedirectResponse("/docs",status_code=308)
    }

# -------------------------------------------------
# INPUT SCHEMA
# -------------------------------------------------
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

# -------------------------------------------------
# PREDICTION ENDPOINT
# -------------------------------------------------
@app.post("/predict")
def predict(student: StudentData):
    df = pd.DataFrame([student.model_dump()])
    pred_encoded = xgb_clf.predict(df)[0]
    pred_label = label_enc.inverse_transform([pred_encoded])[0]
    return {"predicted_career": pred_label}
