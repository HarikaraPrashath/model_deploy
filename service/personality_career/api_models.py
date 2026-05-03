from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional

class Question(BaseModel):
    id: int = Field(..., gt=0)
    question: str = Field(..., min_length=5)
    category: str
    duration: int = Field(..., gt=0)

class EmotionData(BaseModel):
    angry: float = Field(0.0, ge=0.0, le=1.0)
    disgust: float = Field(0.0, ge=0.0, le=1.0)
    fear: float = Field(0.0, ge=0.0, le=1.0)
    happy: float = Field(0.0, ge=0.0, le=1.0)
    sad: float = Field(0.0, ge=0.0, le=1.0)
    surprise: float = Field(0.0, ge=0.0, le=1.0)
    neutral: float = Field(0.0, ge=0.0, le=1.0)

class QuestionEmotions(BaseModel):
    questionId: int = Field(..., gt=0)
    emotions: EmotionData
    timestamp: Optional[str] = None

class AnalyzeRequest(BaseModel):
    userName: str = Field(..., min_length=2, max_length=50)
    emotionHistory: List[QuestionEmotions] = Field(..., min_length=1)
    blinkRate: Optional[float] = Field(None, ge=0.0, le=100.0)

    @field_validator('userName')
    @classmethod
    def name_must_be_alphabetic(cls, v):
        if not all(c.isalpha() or c.isspace() for c in v):
            raise ValueError('User name must only contain letters and spaces')
        return v

class WSMessage(BaseModel):
    emotions: Optional[EmotionData] = None
    image: Optional[str] = None # Base64 frame
    type: Optional[str] = "EMOTION_STREAM"

class PersonalityResult(BaseModel):
    openness: float = Field(..., ge=0.0, le=1.0)
    conscientiousness: float = Field(..., ge=0.0, le=1.0)
    extraversion: float = Field(..., ge=0.0, le=1.0)
    agreeableness: float = Field(..., ge=0.0, le=1.0)
    neuroticism: float = Field(..., ge=0.0, le=1.0)

class CareerRecommendation(BaseModel):
    career: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: str
    skills: List[str] = []
    growth_path: Optional[str] = None
    justification: Optional[str] = None

class AnalyzeResponse(BaseModel):
    userName: str
    aggregatedEmotions: Dict[str, float]
    personality: PersonalityResult
    topCareers: List[CareerRecommendation]
    otherCareers: List[CareerRecommendation]
    emotionTimeline: List[Dict]
    insights: List[str]
    blinkRate: Optional[float] = None
