"""
Intelligent inference engine for personality prediction

Features:
- Model loading and prediction
- Feature importance explanation
- Confidence calibration
- Behavioral insight generation
- Advanced analysis (comparison, shifts, etc.)
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

from src.feature_extractor import PersonalityFeatureExtractor
from src.interpretation import (
    ConfidenceCalibrator,
    FEATURE_LABELS,
    FEATURE_EXPLANATIONS
)
from utils import get_config, score_to_label, TRAITS, TRAIT_DESCRIPTIONS


class PersonalityPredictor:
    """Complete prediction pipeline with explainability and insights."""
    
    def __init__(self, model_path: str = None):
        self.scaler = None  # FIX: define first
        
        self.model = self._load_model(model_path)
        self.feature_extractor = self._load_feature_extractor()
        
        self.confidence_calibrator = ConfidenceCalibrator()
        
    def _load_model(self, path: str = None):
        """Load trained model."""
        
        if path is None:
            path = os.path.join(get_config('model_dir'), 'best_model.pkl')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Extract scaler if present
        if hasattr(model, 'scaler') and model.scaler is not None:
            self.scaler = model.scaler
        else:
            self.scaler = None  # Explicit (important for clarity)

        return model
    
    def _load_feature_extractor(self) -> PersonalityFeatureExtractor:
        """Load trained feature extractor."""

        fe_path = os.path.join(get_config('model_dir'), 'feature_extractor.pkl')

        if not os.path.exists(fe_path):
            raise FileNotFoundError(
                "Feature extractor not found. Please run training first."
            )

        try:
            with open(fe_path, 'rb') as f:
                fe = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load feature extractor: {e}")

        # Critical safety check
        if not getattr(fe, "is_fitted", False):
            raise RuntimeError("Feature extractor is not fitted properly.")

        return fe
    
    def predict_personality(self,text: str,return_confidence: bool = True) -> Dict[str, Dict[str, float]]:
        """Predict personality traits."""
        # Validate input
        min_len = get_config('min_text_length')
        if len(text) < min_len:
            raise ValueError(f"Text must be at least {min_len} characters")

        # Extract features
        features = self.feature_extractor.extract_features(text).reshape(1, -1)

        # Scale if needed
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Predict safely
        try:
            predictions = self.model.predict(features)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

        # Ensure correct shape
        if predictions.ndim != 2 or predictions.shape[1] != len(TRAITS):
            raise RuntimeError("Model output shape mismatch")

        predictions = predictions[0]

        # Clamp values cleanly
        predictions = np.clip(predictions, 0.0, 1.0)

        # Build result
        results = {}
        for i, trait in enumerate(TRAITS):
            score = float(predictions[i])

            results[trait] = {
                "score": score,
                "label": score_to_label(score),
                "confidence": (
                    self.confidence_calibrator.normalize_confidence(score)
                    if return_confidence else None
                ),
                "description": TRAIT_DESCRIPTIONS[trait],
            }

        return results
    
    def explain_prediction(
        self,
        text: str,
        predictions: Dict = None
    ) -> Dict[str, any]:
        """Explain which features influenced predictions."""

        # Ensure predictions exist (for context-aware explanation)
        if predictions is None:
            predictions = self.predict_personality(text)

        # Extract features
        ling_features = self.feature_extractor.extract_linguistic_features(text)
        emotion_features = self.feature_extractor.emotion_extractor.extract(text)

        importance_scores = {}

        # ---- Dynamic weighting (slightly smarter) ----
        feature_weights = {
            'ling_feat_1': 0.18,
            'ling_feat_3': 0.15,
            'ling_feat_0': 0.12,
            'ling_feat_5': 0.10,
            'ling_feat_2': 0.08,
            'ling_feat_4': 0.08,
            'ling_feat_19': 0.07,
            'ling_feat_17': 0.04,
        }

        # Linguistic importance (deviation-based)
        for feat_name, weight in feature_weights.items():
            if feat_name in ling_features:
                value = ling_features[feat_name]
                importance_scores[feat_name] = abs(value - 0.5) * weight

        # ---- Emotion importance (refined) ----
        if "emotion_intensity" in emotion_features:
            importance_scores["emotion_intensity"] = emotion_features["emotion_intensity"] * 0.12

        if "emotion_diversity" in emotion_features:
            importance_scores["emotion_diversity"] = emotion_features["emotion_diversity"] * 0.08

        if "emotion_polarity" in emotion_features:
            importance_scores["emotion_polarity"] = abs(emotion_features["emotion_polarity"]) * 0.12

        if "emotion_stability" in emotion_features:
            # Lower stability = more important
            importance_scores["emotion_stability"] = (1 - emotion_features["emotion_stability"]) * 0.10

        # ---- Sort ----
        top_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:8]

        # ---- Human-readable ----
        readable_features = []
        for name, score in top_features:
            readable_features.append({
                "feature": FEATURE_LABELS.get(name, name),
                "importance": float(score),
                "explanation": FEATURE_EXPLANATIONS.get(
                    name,
                    "This signal influenced the prediction."
                )
            })

        # ---- Context-aware summary (NEW) ----
        top_traits = sorted(
            predictions.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )[:2]

        trait_context = ", ".join([t[0] for t in top_traits])

        if readable_features:
            top_names = [f["feature"] for f in readable_features[:3]]

            explanation_text = (
                f"Your {trait_context.lower()} traits are mainly influenced by {', '.join(top_names)}. "
                f"These reflect how your emotional tone, structure, and thinking patterns appear in the text."
            )
        else:
            explanation_text = "The prediction is based on overall writing patterns."

        return {
            "top_features": readable_features,
            "summary": explanation_text,
            "raw_scores": importance_scores,
        }
    
    def generate_insights(
        self,
        text: str,
        predictions: Dict = None
    ) -> Dict[str, str]:
        """Generate richer behavioral insights from predictions + emotion."""

        if predictions is None:
            predictions = self.predict_personality(text)

        # Extract emotion features
        emotion_features = self.feature_extractor.emotion_extractor.extract(text)
        dominant = emotion_features.get("dominant_emotion", None)
        polarity = emotion_features.get("emotion_polarity", 0.0)

        # Trait grouping (more flexible)
        traits_high = [t for t, v in predictions.items() if v['score'] >= 0.65]
        traits_low = [t for t, v in predictions.items() if v['score'] <= 0.35]

        narrative_parts = []

        # ---- High traits ----
        if traits_high:
            descriptions = []

            if "Openness" in traits_high:
                descriptions.append("reflective and open to exploring ideas")
            if "Conscientiousness" in traits_high:
                descriptions.append("structured and goal-oriented")
            if "Extraversion" in traits_high:
                descriptions.append("expressive and outwardly engaged")
            if "Agreeableness" in traits_high:
                descriptions.append("empathetic and cooperative")
            if "Neuroticism" in traits_high:
                descriptions.append("emotionally sensitive and reactive")

            if descriptions:
                narrative_parts.append(
                    f"You show strong tendencies toward being {', '.join(descriptions)}."
                )

        # ---- Low traits ----
        if traits_low:
            descriptions = []

            if "Openness" in traits_low:
                descriptions.append("more practical and grounded")
            if "Conscientiousness" in traits_low:
                descriptions.append("flexible rather than rigidly structured")
            if "Extraversion" in traits_low:
                descriptions.append("more inward-focused and reserved")
            if "Agreeableness" in traits_low:
                descriptions.append("more independent in decision-making")
            if "Neuroticism" in traits_low:
                descriptions.append("emotionally steady and less reactive")

            if descriptions:
                narrative_parts.append(
                    f"You also show patterns of being {', '.join(descriptions)}."
                )

        # ---- Emotion-aware insight (NEW) ----
        if dominant is not None:
            emotion_map = {
                0: "anger",
                2: "anxiety",
                3: "positivity",
                5: "sadness"
            }

            emotion_label = emotion_map.get(dominant, None)

            if emotion_label == "anxiety":
                narrative_parts.append(
                    "Your current emotional tone suggests underlying tension or concern influencing your thinking."
                )
            elif emotion_label == "sadness":
                narrative_parts.append(
                    "There is a heavier emotional layer present, which may slow down or deepen your thought process."
                )
            elif emotion_label == "positivity":
                narrative_parts.append(
                    "The emotional tone carries a constructive or optimistic direction."
                )
            elif emotion_label == "anger":
                narrative_parts.append(
                    "There are signs of frustration or intensity shaping how the situation is being interpreted."
                )

        # ---- Polarity nuance ----
        if polarity > 0.25:
            narrative_parts.append("Overall, your emotional direction leans positive.")
        elif polarity < -0.25:
            narrative_parts.append("Overall, the emotional tone leans heavier or more negative.")

        # Final behavioral text
        behavioral_text = (
            " ".join(narrative_parts)
            if narrative_parts
            else "Your personality profile appears relatively balanced across traits."
        )

        # ---- Communication style (improved) ----
        extraversion = predictions["Extraversion"]["score"]
        agreeableness = predictions["Agreeableness"]["score"]
        conscientiousness = predictions["Conscientiousness"]["score"]
        openness = predictions["Openness"]["score"]

        if extraversion > 0.6 and agreeableness > 0.55:
            comm_style = "Collaborative and expressive"
        elif conscientiousness > 0.6:
            comm_style = "Structured and methodical"
        elif openness > 0.6:
            comm_style = "Exploratory and idea-driven"
        elif extraversion < 0.4:
            comm_style = "Reserved and introspective"
        else:
            comm_style = "Balanced and adaptive"

        return {
            "behavioral_text": behavioral_text,
            "communication_style": {
                "primary": comm_style,
                "description": (
                    f"Your communication style is {comm_style.lower()}, "
                    f"reflecting how you organize thoughts and express ideas."
                )
            },
            "key_characteristics": [
                f"{trait}: {predictions[trait]['label'].lower()}"
                for trait in TRAITS
            ]
        }
    
    def compare_texts(
        self,
        text1: str,
        text2: str
    ) -> Dict[str, any]:
        """Compare personality and highlight meaningful changes."""

        pred1 = self.predict_personality(text1)
        pred2 = self.predict_personality(text2)

        differences = {}
        significant_changes = []

        for trait in TRAITS:
            score1 = pred1[trait]['score']
            score2 = pred2[trait]['score']
            diff = score2 - score1

            direction = (
                'increased' if diff > 0.05
                else 'decreased' if diff < -0.05
                else 'stable'
            )

            differences[trait] = {
                'text1_score': score1,
                'text2_score': score2,
                'difference': diff,
                'direction': direction,
            }

            # Track meaningful shifts
            if abs(diff) > 0.1:
                significant_changes.append((trait, diff))

        # Summary generation
        if significant_changes:
            top_trait, top_diff = sorted(
                significant_changes,
                key=lambda x: abs(x[1]),
                reverse=True
            )[0]

            summary = (
                f"The most noticeable shift is in {top_trait.lower()}, "
                f"which has {'increased' if top_diff > 0 else 'decreased'} significantly."
            )
        else:
            summary = "There are no major personality shifts between the two texts."

        return {
            'predictions_1': pred1,
            'predictions_2': pred2,
            'differences': differences,
            'summary': summary,
            'significant_changes': significant_changes,
        }
    
    def detect_emotional_shifts(self, text: str) -> Dict[str, any]:
        """Detect emotional shifts with interpretation."""

        from src.preprocessing import tokenize_sentences
        from nltk.sentiment import SentimentIntensityAnalyzer

        sia = SentimentIntensityAnalyzer()
        sentences = tokenize_sentences(text)

        sentiments = []
        for sent in sentences:
            if sent.strip():
                sentiments.append(sia.polarity_scores(sent)['compound'])

        if not sentiments:
            return {
                "shifts": [],
                "overall_trend": "neutral",
                "volatility": 0,
                "interpretation": "Not enough emotional content detected."
            }

        # ---- Detect shifts ----
        shifts = []
        for i in range(1, len(sentiments)):
            change = sentiments[i] - sentiments[i - 1]

            if abs(change) > 0.3:
                shifts.append({
                    "position": i,
                    "from": sentiments[i - 1],
                    "to": sentiments[i],
                    "magnitude": change,
                    "type": "positive jump" if change > 0 else "negative drop"
                })

        # ---- Trend ----
        trend_value = sentiments[-1] - sentiments[0]

        if trend_value > 0.2:
            overall_trend = "improving"
        elif trend_value < -0.2:
            overall_trend = "declining"
        else:
            overall_trend = "stable"

        # ---- Volatility ----
        volatility = float(np.std(sentiments))
        avg_sentiment = float(np.mean(sentiments))

        # ---- Interpretation (NEW) ----
        if volatility > 0.4:
            interpretation = "Emotions fluctuate strongly, suggesting instability or shifting perspective."
        elif volatility > 0.2:
            interpretation = "There is moderate emotional variation across the text."
        else:
            interpretation = "Emotions remain relatively steady throughout."

        if overall_trend == "improving":
            interpretation += " The emotional tone becomes more positive over time."
        elif overall_trend == "declining":
            interpretation += " The emotional tone becomes more negative over time."

        return {
            "shifts": shifts,
            "overall_trend": overall_trend,
            "volatility": volatility,
            "average_sentiment": avg_sentiment,
            "interpretation": interpretation,
        }