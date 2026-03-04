from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from model_schema.schema import StudentInput
from service.Career_preddiction import predict_career_service

#files are import from service folder
from service.career_guide_service import predict_career

#this is Prevent the CORS issues
app = FastAPI(title="Career Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudentInput(BaseModel):
    Soft_Skills: str
    Key_Skils: str
    Current_semester: str
    Learning_Style: str
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

    # Make optional
    Is_Sliit_Student: Optional[bool] = False
    Specialization: Optional[str] = ""

#this is the root path
@app.get("/")
def root():
    return {"status": "Career Prediction API running"}

# this is for the career Guide endpoint
@app.post("/predict-career")
def predict(inp: StudentInput):
    return predict_career(inp)

@app.post("/predict")
def predict_career(student: StudentInput):
    print("✅ Request received for Random Forest prediction")
    return predict_career_service(student)

@app.post("/debug")
def debug_endpoint(student: StudentInput):
    return {"received": student.dict()}
