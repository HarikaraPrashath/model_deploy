from __future__ import annotations

import os
import sys
from typing import Any


from typing import Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from model_schema.schema import StudentInput
from model_schema.student_model import StudentInputGuide
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from service.Career_preddiction import predict_career_service

#files are import from service folder
from service.career_guide_service import predict_career


from service.career_market import cv_service

from service.career_market.auth_endpoints_service import (
    signup_service,
    login_service,
    forgot_password_service,
)

from service.career_market.profile_service import (
    get_profile_service,
    put_profile_service,
)
from service.career_market.jobs_service import (
    get_jobs_service,
    get_job_file_service,
    refresh_jobs_service,
)
from service.career_market.trends_service import (
    get_trend_history_service,
    get_trends_service,
    seed_trends_service,
)
from service.career_market.analysis_service import analyse_service
from service.career_market.root_health_service import health_service
from service.career_market.ranked_service import (
    get_ranked_service,
    get_ranked_summary_service,
)

from service.career_market.utils.config import (
    ANALYSIS_BACKEND_DIR,
    CV_EXTRACTOR_DIR,
    MAX_FILE_SIZE,
    ensure_storage_dirs,
)


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

#---career-market---------------------------------------------------------------------------------------

@app.post("/auth/signup")
async def signup_endpoint(payload: dict[str, Any]) -> JSONResponse:
    return signup_service(payload)

@app.post("/auth/login")
async def login_endpoint(payload: dict[str, Any]) -> JSONResponse:
    return login_service(payload)

@app.post("/auth/forgot-password")
async def forgot_password_endpoint(payload: dict[str, Any]) -> JSONResponse:
    return forgot_password_service(payload)
#----------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return health_service()

@app.post("/parse")
def parse_cv_endpoint(file: UploadFile = File(...)):
    return cv_service.parse_cv(file, max_file_size=MAX_FILE_SIZE)

@app.get("/cv")
def get_latest_cv_endpoint(request: Request):
    return cv_service.get_latest_cv_response(request)

@app.get("/cv/file")
def get_cv_file_endpoint(id: str, request: Request):
    return cv_service.get_cv_file_response(id, request)

@app.get("/profile")
def get_profile_endpoint(request: Request) -> JSONResponse:
    return get_profile_service(request)

@app.put("/profile")
async def put_profile_endpoint(payload: dict[str, Any], request: Request) -> JSONResponse:
    return put_profile_service(payload, request)

@app.get("/jobs")
def get_jobs_endpoint() -> JSONResponse:
    return get_jobs_service()

@app.get("/jobs/file")
def get_job_file_endpoint(name: str) -> FileResponse:
    return get_job_file_service(name)

@app.post("/jobs/refresh")
async def refresh_jobs_endpoint(request: Request, payload: dict[str, Any] | None = None) -> JSONResponse:
    return refresh_jobs_service(request, payload)

@app.get("/trends/history")
def get_trend_history_endpoint() -> JSONResponse:
    return get_trend_history_service()

@app.get("/trends")
def get_trends_endpoint() -> JSONResponse:
    return get_trends_service()

@app.post("/trends/seed")
async def seed_trends_endpoint(request: Request, payload: dict[str, Any] | None = None) -> JSONResponse:
    return seed_trends_service(request, payload)

@app.get("/ranked")
def get_ranked_endpoint() -> JSONResponse:
    return get_ranked_service()

@app.get("/ranked/summary")
def get_ranked_summary_endpoint() -> JSONResponse:
    return get_ranked_summary_service()

@app.post("/analyse")
async def analyse_endpoint(request: Request, payload: dict[str, Any] | None = None) -> JSONResponse:
    return await analyse_service(request, payload)
