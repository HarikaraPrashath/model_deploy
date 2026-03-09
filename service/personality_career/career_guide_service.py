import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import io
import base64
import timm
from typing import Dict, List, Optional
from lib.logger import get_logger
from service.personality_career.api_models import AnalyzeRequest, AnalyzeResponse, CareerRecommendation, PersonalityResult
from service.personality_career.constants import CAREER_DETAILS

logger = get_logger(__name__)

class CareerService:
    """
    Advanced Career Service using Random Forest (Career) and EfficientNet-B0 (Emotion).
    """

    EMOTION_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    PERSONALITY_NAMES = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    ENGINEERED_NAMES = ['dominant_emotion_idx', 'emotion_entropy', 'emotion_variance',
                        'neutral_ratio', 'positive_ratio', 'negative_ratio']

    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        self._load_models()
        logger.info("CareerService initialized with Random Forest model")

    def _load_models(self) -> None:
        """Load the Random Forest model and EfficientNet-B0 Emotion model."""
        try:
            # 1. Main Random Forest Prediction Model
            rf_path = os.path.join(self.model_dir, 'career_prediction_model.joblib')
            self.model = joblib.load(rf_path)

            # 2. Preprocessors
            scaler_path = os.path.join(self.model_dir, 'feature_scaler.pkl')
            self.scaler = joblib.load(scaler_path)

            encoder_path = os.path.join(self.model_dir, 'career_label_encoder.joblib')
            self.label_encoder = joblib.load(encoder_path)

            # 3. EfficientNet-B0 Emotion Model (PyTorch)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            emotion_model_path = os.path.join(self.model_dir, 'best_effnetb0_pytorch.pth')

            if os.path.exists(emotion_model_path):
                # We need to know num_classes (default 7 for FER2013)
                num_classes = 7
                self.emotion_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
                self.emotion_model.load_state_dict(torch.load(emotion_model_path, map_location=self.device))
                self.emotion_model.to(self.device)
                self.emotion_model.eval()

                # Setup Preprocessing Transform for B0
                from torchvision import transforms
                self.emotion_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                logger.info("[DONE] PyTorch EfficientNet-B0 Emotion model loaded")
            else:
                self.emotion_model = None
                logger.warning("! Emotion model file not found. Real-time backend detection disabled.")

            logger.info("[DONE] Random Forest and preprocessors loaded successfully")
        except Exception as e:
            logger.error(f"Critical failure loading models: {str(e)}")
            raise

    def calculate_emotion_features(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """Convert raw emotion probabilities into engineered features for the RF model."""
        emotion_values = np.array([emotions.get(e, 0.0) for e in self.EMOTION_NAMES])

        # Normalize just in case
        if emotion_values.sum() > 0:
            emotion_values = emotion_values / emotion_values.sum()
        else:
            emotion_values = np.array([0, 0, 0, 0, 0, 0, 1])

        dominant_idx = int(np.argmax(emotion_values))
        entropy = float(-np.sum(emotion_values * np.log(emotion_values + 1e-10)))
        variance = float(np.var(emotion_values))

        feat_dict = {name: float(val) for name, val in zip(self.EMOTION_NAMES, emotion_values)}
        feat_dict.update({
            'dominant_emotion_idx': dominant_idx,
            'emotion_entropy': entropy,
            'emotion_variance': variance,
            'neutral_ratio': float(emotion_values[6]),
            'positive_ratio': float(emotion_values[3] + emotion_values[5]),
            'negative_ratio': float(emotion_values[0] + emotion_values[1] + emotion_values[2] + emotion_values[4])
        })
        return feat_dict

    def predict_personality(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """Heuristic-based personality prediction to provide features for the RF model."""
        happy = emotions.get('happy', 0)
        sad = emotions.get('sad', 0)
        angry = emotions.get('angry', 0)
        neutral = emotions.get('neutral', 0)
        fear = emotions.get('fear', 0)
        surprise = emotions.get('surprise', 0)
        disgust = emotions.get('disgust', 0)

        # Simplified psychology formulas
        personality = {
            'openness': float(np.clip(0.4 + surprise * 0.8 + neutral * 0.2, 0.1, 0.95)),
            'conscientiousness': float(np.clip(0.5 + neutral * 0.5 - angry * 0.4, 0.1, 0.95)),
            'extraversion': float(np.clip(0.3 + happy * 0.9 - sad * 0.3, 0.1, 0.95)),
            'agreeableness': float(np.clip(0.5 + happy * 0.5 - angry * 0.6 - disgust * 0.3, 0.1, 0.95)),
            'neuroticism': float(np.clip(0.2 + fear * 0.7 + sad * 0.4 - happy * 0.3, 0.1, 0.95))
        }
        return personality

    def predict_emotion(self, image_data: str) -> Dict[str, float]:
        """Predict emotion probabilities from a base64 encoded image frame."""
        if not self.emotion_model:
            return {e: 0.0 for e in self.EMOTION_NAMES}

        try:
            # Decode base64
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]

            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            # Preprocess
            tensor = self.emotion_transform(img).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.emotion_model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]

            return {name: float(probs[i]) for i, name in enumerate(self.EMOTION_NAMES[:probs.size(0)])}
        except Exception as e:
            logger.error(f"Backend emotion prediction failed: {str(e)}")
            return {e: 0.0 for e in self.EMOTION_NAMES}

    def predict_career(self, emotions: Dict[str, float], model_name: str = 'rf') -> Dict:
        """Main prediction pipeline: Emotion -> Personality -> Career."""
        # 1. Feature Prep
        emo_feats = self.calculate_emotion_features(emotions)
        personality = self.predict_personality(emotions)

        # 2. Build Feature Vector (Must match training order)
        # Order in CSV: Emotions(7), Personality(5), Engineered(6)
        feature_vector = [
            emo_feats['angry'], emo_feats['disgust'], emo_feats['fear'],
            emo_feats['happy'], emo_feats['sad'], emo_feats['surprise'], emo_feats['neutral'],
            personality['openness'], personality['conscientiousness'], personality['extraversion'],
            personality['agreeableness'], personality['neuroticism'],
            emo_feats['dominant_emotion_idx'], emo_feats['emotion_entropy'], emo_feats['emotion_variance'],
            emo_feats['neutral_ratio'], emo_feats['positive_ratio'], emo_feats['negative_ratio']
        ]

        # 3. Model Inference
        X_scaled = self.scaler.transform(np.array([feature_vector]))
        probabilities = self.model.predict_proba(X_scaled)[0]

        # 4. Results
        pred_idx = np.argmax(probabilities)
        predicted_career = self.label_encoder.classes_[pred_idx]

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_careers = []
        for idx in top_indices:
            career_name = self.label_encoder.classes_[idx]
            top_careers.append({
                'career': career_name,
                'confidence': float(probabilities[idx]),
                'justification': self.generate_justification(career_name, personality, emotions)
            })

        return {
            'predicted_career': predicted_career,
            'top_careers': top_careers,
            'personality': personality,
            'model_used': "Random Forest (EPCDF-trained)"
        }

    def generate_justification(self, career: str, personality: Dict[str, float], emotions: Dict[str, float]) -> str:
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "focused"
        reason = f"Matches your cognitive profile. Your strong {dominant_emotion} indicators align with the core requirements of a {career} role."
        return reason

    def get_careers(self) -> List[str]:
        return list(self.label_encoder.classes_)