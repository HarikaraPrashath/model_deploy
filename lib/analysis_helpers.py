from typing import List, Dict
from service.personality_career.api_models import QuestionEmotions

def aggregate_emotions(emotion_history: List[QuestionEmotions]) -> Dict[str, float]:
    """Aggregate emotions from all questions."""
    if not emotion_history:
        return {
            'angry': 0.1, 'disgust': 0.05, 'fear': 0.1,
            'happy': 0.3, 'sad': 0.1, 'surprise': 0.1, 'neutral': 0.25
        }

    totals = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}

    for item in emotion_history:
        emotions = item.emotions
        totals['angry'] += emotions.angry
        totals['disgust'] += emotions.disgust
        totals['fear'] += emotions.fear
        totals['happy'] += emotions.happy
        totals['sad'] += emotions.sad
        totals['surprise'] += emotions.surprise
        totals['neutral'] += emotions.neutral

    # Average
    count = len(emotion_history)
    for key in totals:
        totals[key] /= count

    # Normalize
    total_sum = sum(totals.values())
    if total_sum > 0:
        for key in totals:
            totals[key] /= total_sum

    return totals

def generate_insights(personality: Dict, top_career: str, emotions: Dict) -> List[str]:
    """Generate personalized insights based on analysis."""
    insights = []

    # Personality insights
    if personality['openness'] > 0.7:
        insights.append("Your high openness suggests you thrive in creative and innovative environments.")
    if personality['conscientiousness'] > 0.7:
        insights.append("Your strong conscientiousness indicates excellent organizational skills.")
    if personality['extraversion'] > 0.7:
        insights.append("Your extraverted nature makes you well-suited for collaborative roles.")
    if personality['agreeableness'] > 0.7:
        insights.append("Your high agreeableness shows strong team collaboration potential.")
    if personality['neuroticism'] < 0.4:
        insights.append("Your low neuroticism indicates good stress management abilities.")

    # Emotion insights
    if emotions.get('happy', 0) > 0.3:
        insights.append("You displayed positive engagement throughout the interview.")
    if emotions.get('neutral', 0) > 0.4:
        insights.append("Your composed demeanor suggests professional maturity.")

    # Career insight
    insights.append(f"{top_career} aligns well with your personality profile and emotional patterns.")

    return insights[:5]
