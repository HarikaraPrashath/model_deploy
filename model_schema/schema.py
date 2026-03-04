# models_schema.py
from pydantic import BaseModel
from typing import List

class AcademicBackground(BaseModel):
    currentSemester: str
    currentYear: int
    educationLevel: str
    gpa: float
    major: str

class Career(BaseModel):
    certifications: str
    internship: str
    learningStyle: str
    projects: int
    stressManagement: str

class CareerInterests(BaseModel):
    workEnvironment: str
    workLifeBalance: str

class PersonalInfo(BaseModel):
    gender: str
    languages: List[str]

class TechnicalSkills(BaseModel):
    cloudPlatforms: List[str]
    databases: List[str]
    frameworks: List[str]
    programming: List[str]

class StudentInput(BaseModel):
    academicBackground: AcademicBackground
    career: Career
    careerInterests: CareerInterests
    personalInfo: PersonalInfo
    technicalSkills: TechnicalSkills