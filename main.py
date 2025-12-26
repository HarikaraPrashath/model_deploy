from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

# Load saved model and label encoder
xgb_clf = joblib.load("career_prediction_xgboost_updated.joblib")  # this was a model
label_enc = joblib.load("career_label_encoder_updated.joblib") # this is value to datatype conventer

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema (replace feature names with your CSV columns)
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
@app.post("/predict")
def predict(student: StudentData):
    # Convert input to DataFrame
    df = pd.DataFrame([student.model_dump()])  # Pydantic v2
    # If you're using Pydantic v1, replace with: student.dict()

    pred_encoded = xgb_clf.predict(df)[0]
    pred_label = label_enc.inverse_transform([pred_encoded])[0]

    return {"predicted_career": pred_label}
