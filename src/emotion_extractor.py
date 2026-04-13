"""
Emotion extraction using HuggingFace transformer pipelines.

Design rules:
- NO streamlit import here. This module is pure ML; UI concerns live in app.py.
- Models are class-level singletons loaded lazily on first use.
- truncation=True and _MAX_CHARS pre-clip ensure fast CPU inference always.
"""

from transformers import pipeline
import numpy as np


class EmotionExtractor:
    _emotion_model = None
    _sentiment_model = None

    EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    # Hard character cap before tokenisation — keeps CPU inference under ~3 s
    _MAX_CHARS = 1500

    def __init__(self):
        # Lazy singleton — models are built once, reused forever across all instances
        if EmotionExtractor._emotion_model is None:
            EmotionExtractor._emotion_model = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
                truncation=True,
                max_length=512,
            )
        if EmotionExtractor._sentiment_model is None:
            EmotionExtractor._sentiment_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512,
            )

        self.emotion_model = EmotionExtractor._emotion_model
        self.sentiment_model = EmotionExtractor._sentiment_model

    def extract(self, text: str) -> dict:
        truncated = text[: self._MAX_CHARS]

        emotions = self.emotion_model(truncated)[0]
        emotion_dict = {e["label"]: e["score"] for e in emotions}

        sentiment = self.sentiment_model(truncated)[0]
        sentiment_score = sentiment["score"]
        if sentiment["label"] == "NEGATIVE":
            sentiment_score *= -1

        values = np.array([emotion_dict[label] for label in self.EMOTION_LABELS])

        intensity = float(np.max(values))

        diversity = -np.sum(values * np.log(values + 1e-9))
        diversity = float(diversity / np.log(len(values)))

        dominant_emotion = max(emotion_dict, key=emotion_dict.get)
        dominant_idx = self.EMOTION_LABELS.index(dominant_emotion)

        positive = emotion_dict.get("joy", 0)
        negative = (
            emotion_dict.get("anger", 0)
            + emotion_dict.get("sadness", 0)
            + emotion_dict.get("fear", 0)
        )
        total = positive + negative + 1e-9
        polarity_balance = float((positive - negative) / total)
        stability = float(1 - np.std(values))

        return {
            **emotion_dict,
            "sentiment": float(sentiment_score),
            "emotion_intensity": intensity,
            "emotion_diversity": diversity,
            "emotion_polarity": polarity_balance,
            "emotion_stability": stability,
            "dominant_emotion": dominant_idx,
        }