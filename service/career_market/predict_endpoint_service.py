from __future__ import annotations

from model_schema.schema import StudentInput
from model_schema.student_model import StudentInputGuide

from service.Career_preddiction import predict_career_service
from service.career_guide_service import predict_career as predict_career_guide


def predict_career_guide_service(student: StudentInputGuide):
    return predict_career_guide(student)


def predict_service(student: StudentInput):
    print("✅ Request received for Random Forest prediction")
    return predict_career_service(student)


def debug_service(student: StudentInput):
    return {"received": student.dict()}
