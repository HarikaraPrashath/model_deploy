from typing import Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from model_schema.schema import StudentInput
from model_schema.student_model import StudentInputGuide
from service.Career_preddiction import predict_career_service

#files are import from service folder
from service.career_guide_service import predict_career

#this is Prevent the CORS issues
app = FastAPI(title="Career Prediction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#this is the root path
@app.get("/")
def root():
    return {"status": "Career Prediction API running"}

# this is for the career Guide endpoint
@app.post("/predict-career")
def predict_career(student: StudentInputGuide):
    return predict_career(student)

@app.post("/predict")
def predict_career(student: StudentInput):
    print("✅ Request received for Random Forest prediction")
    return predict_career_service(student)

@app.post("/debug")
def debug_endpoint(student: StudentInput):
    return {"received": student.dict()}
