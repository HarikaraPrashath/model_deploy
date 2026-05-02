from typing import Any, Dict, List
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from service.personality_career.constants import CAREER_DETAILS


def analyze_interview_service(payload: Dict[str, Any]) -> JSONResponse:
    """
    Analyze interview emotions and predict career based on emotional patterns.
    """
    try:
        user_name = payload.get("userName", "Candidate")
        emotion_history = payload.get("emotionHistory", [])
        blink_rate = payload.get("blinkRate", 0)

        if not emotion_history:
            raise HTTPException(status_code=400, detail="No emotion data provided")

        # Aggregate emotions across all questions
        aggregated_emotions = {
            "angry": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "happy": 0.0,
            "sad": 0.0,
            "neutral": 0.0,
            "surprise": 0.0
        }

        # Calculate average emotions
        for question_data in emotion_history:
            emotions = question_data.get("emotions", {})
            for emotion, value in emotions.items():
                if emotion in aggregated_emotions:
                    aggregated_emotions[emotion] += float(value)

        # Normalize by number of questions
        num_questions = len(emotion_history)
        for emotion in aggregated_emotions:
            aggregated_emotions[emotion] /= num_questions

        # Calculate personality traits based on emotions
        personality = calculate_personality_traits(aggregated_emotions, blink_rate)

        # Predict top careers based on personality
        top_careers, other_careers = predict_careers(personality)

        # Create emotion timeline
        emotion_timeline = [
            {
                "questionId": q.get("questionId"),
                "emotions": q.get("emotions", {}),
                "timestamp": q.get("timestamp")
            }
            for q in emotion_history
        ]

        # Generate insights
        insights = generate_insights(aggregated_emotions, personality, blink_rate)

        result = {
            "userName": user_name,
            "aggregatedEmotions": aggregated_emotions,
            "personality": personality,
            "topCareers": top_careers,
            "otherCareers": other_careers,
            "emotionTimeline": emotion_timeline,
            "insights": insights,
            "blinkRate": blink_rate
        }

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def calculate_personality_traits(emotions: Dict[str, float], blink_rate: float) -> Dict[str, float]:
    """
    Calculate Big Five personality traits based on emotional patterns.
    """
    # Simplified mapping based on emotional tendencies
    openness = (emotions.get("surprise", 0) + emotions.get("happy", 0)) / 2
    conscientiousness = (emotions.get("neutral", 0) + (1 - emotions.get("fear", 0))) / 2
    extraversion = (emotions.get("happy", 0) + emotions.get("surprise", 0)) / 2
    agreeableness = (emotions.get("happy", 0) + (1 - emotions.get("angry", 0)) + (1 - emotions.get("disgust", 0))) / 3
    neuroticism = (emotions.get("fear", 0) + emotions.get("sad", 0) + emotions.get("angry", 0)) / 3

    # Adjust based on blink rate (higher blink rate may indicate nervousness)
    if blink_rate > 20:  # blinks per minute
        neuroticism += 0.1
        conscientiousness -= 0.1

    return {
        "openness": min(1.0, max(0.0, openness)),
        "conscientiousness": min(1.0, max(0.0, conscientiousness)),
        "extraversion": min(1.0, max(0.0, extraversion)),
        "agreeableness": min(1.0, max(0.0, agreeableness)),
        "neuroticism": min(1.0, max(0.0, neuroticism))
    }


def predict_careers(personality: Dict[str, float]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Predict career recommendations based on personality traits.
    """
    careers = []

    # Define career mappings based on personality traits
    career_mappings = {
        "Data Scientist": {
            "openness": 0.8, "conscientiousness": 0.7, "extraversion": 0.4,
            "agreeableness": 0.6, "neuroticism": 0.3
        },
        "Software Developer": {
            "openness": 0.6, "conscientiousness": 0.8, "extraversion": 0.4,
            "agreeableness": 0.7, "neuroticism": 0.4
        },
        "UI/UX Designer": {
            "openness": 0.9, "conscientiousness": 0.6, "extraversion": 0.7,
            "agreeableness": 0.8, "neuroticism": 0.5
        },
        "DevOps Engineer": {
            "openness": 0.5, "conscientiousness": 0.9, "extraversion": 0.3,
            "agreeableness": 0.6, "neuroticism": 0.4
        },
        "Cybersecurity Analyst": {
            "openness": 0.6, "conscientiousness": 0.8, "extraversion": 0.3,
            "agreeableness": 0.5, "neuroticism": 0.6
        },
        "Cloud Architect": {
            "openness": 0.7, "conscientiousness": 0.8, "extraversion": 0.4,
            "agreeableness": 0.6, "neuroticism": 0.3
        }
    }

    # Calculate similarity scores
    for career_name, required_traits in career_mappings.items():
        similarity = 1.0
        for trait, required_value in required_traits.items():
            user_value = personality.get(trait, 0.5)
            similarity *= (1 - abs(user_value - required_value))

        confidence = similarity ** 0.5  # Square root to normalize

        career_detail = CAREER_DETAILS.get(career_name, {})
        careers.append({
            "career": career_name,
            "confidence": confidence,
            "description": career_detail.get("description", f"Career in {career_name}"),
            "skills": career_detail.get("skills", []),
            "growth_path": career_detail.get("growth_path", ""),
            "justification": f"Based on your personality traits, this career aligns with your {trait} tendencies."
        })

    # Sort by confidence and split into top and other careers
    careers.sort(key=lambda x: x["confidence"], reverse=True)
    top_careers = careers[:3]
    other_careers = careers[3:]

    return top_careers, other_careers


def generate_insights(emotions: Dict[str, float], personality: Dict[str, float], blink_rate: float) -> List[str]:
    """
    Generate insights based on emotional patterns and personality.
    """
    insights = []

    # Emotional insights
    if emotions.get("happy", 0) > 0.6:
        insights.append("You show high enthusiasm and positivity during interviews")
    if emotions.get("neutral", 0) > 0.7:
        insights.append("You maintain composure and focus well under pressure")
    if emotions.get("fear", 0) > 0.4:
        insights.append("Consider building confidence in high-stakes situations")

    # Personality insights
    if personality.get("openness", 0) > 0.7:
        insights.append("Your creative and open-minded nature suits innovative roles")
    if personality.get("conscientiousness", 0) > 0.7:
        insights.append("Your organized and detail-oriented approach fits technical careers")
    if personality.get("extraversion", 0) > 0.7:
        insights.append("Your outgoing personality would thrive in collaborative environments")

    # Blink rate insights
    if blink_rate > 25:
        insights.append("Higher blink rate may indicate nervousness - consider relaxation techniques")
    elif blink_rate < 10:
        insights.append("Your steady gaze shows confidence and focus")

    return insights