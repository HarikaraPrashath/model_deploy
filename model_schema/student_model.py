from typing import Optional
from pydantic import BaseModel

class StudentInputGuide(BaseModel):
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

    # ✅ Make optional
    Is_Sliit_Student: Optional[bool] = False
    Specialization: Optional[str] = ""
