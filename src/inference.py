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
from src.interpretation import ConfidenceCalibrator
from utils import get_config, score_to_label, TRAITS, TRAIT_DESCRIPTIONS


class PersonalityPredictor:
    """Complete prediction pipeline with explainability and insights."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model. Uses 'best_model' by default.
        """
        self.model = self._load_model(model_path)
        self.feature_extractor = self._load_feature_extractor()
        self.scaler = None
        self.confidence_calibrator = ConfidenceCalibrator()
    
    def _load_model(self, path: str = None):
        """Load trained model."""
        if path is None:
            path = os.path.join(get_config('model_dir'), 'best_model.pkl')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        # Extract scaler if it exists (for MLP)
        if hasattr(model, 'scaler'):
            self.scaler = model.scaler
        
        return model
    
    def _load_feature_extractor(self) -> PersonalityFeatureExtractor:
        """Load or create feature extractor."""
        fe_path = os.path.join(get_config('model_dir'), 'feature_extractor.pkl')
        
        if os.path.exists(fe_path):
            with open(fe_path, 'rb') as f:
                return pickle.load(f)
        else:
            # Create fresh extractor
            return PersonalityFeatureExtractor()
    
    def predict_personality(
        self,
        text: str,
        return_confidence: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict personality traits.
        
        Args:
            text: Input text
            return_confidence: Include confidence scores
            
        Returns:
            Dictionary of trait scores and labels
        """
        # Validate input
        if len(text) < get_config('min_text_length'):
            raise ValueError(f"Text must be at least {get_config('min_text_length')} characters")
        
        # Extract features
        features = self.feature_extractor.extract_features(text)
        
        # Scale if needed (MLP)
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))[0]
        
        # Predict
        features = features.reshape(1, -1)
        predictions = self.model.predict(features)[0]
        
        # Process results
        results = {}
        for i, trait in enumerate(TRAITS):
            score = float(predictions[i])
            score = max(0.0, min(1.0, score))  # Clamp to 0-1
            
            results[trait] = {
                'score': score,
                'label': score_to_label(score),
                'confidence': self.confidence_calibrator.normalize_confidence(score) if return_confidence else None,
                'description': TRAIT_DESCRIPTIONS[trait],
            }
        
        return results
    
    def explain_prediction(
        self,
        text: str,
        predictions: Dict = None
    ) -> Dict[str, any]:
        """
        Explain which features drove predictions.
        
        Args:
            text: Input text
            predictions: Previous prediction results (for efficiency)
            
        Returns:
            Dictionary with explanation data
        """
        # Get features
        ling_features = self.feature_extractor.extract_linguistic_features(text)
        
        # Calculate importance (SHAP-inspired)
        importance_scores = {}
        
        # Map linguistic features to importance
        feature_weights = {
            'ling_feat_1': 0.18,  # lexical_diversity
            'ling_feat_3': 0.15,  # sentiment_score
            'ling_feat_0': 0.12,  # avg_word_length
            'ling_feat_5': 0.10,  # emotional_word_ratio
            'ling_feat_2': 0.08,  # avg_sentence_length
            'ling_feat_4': 0.08,  # sentiment_variance
            'ling_feat_19': 0.07, # complex_word_ratio
            'ling_feat_17': 0.04, # readability_score
        }
        
        for feat_name, weight in feature_weights.items():
            if feat_name in ling_features:
                importance_scores[feat_name.replace('ling_feat_', '')] = (
                    ling_features[feat_name] * weight
                )
        
        # Top features
        top_features = sorted(
            importance_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        
        # Explanation text
        if top_features:
            top_names = [f.replace('ling_feat_', '') for f, _ in top_features[:3]]
            explanation = (
                f"Prediction influenced by: {', '.join(top_names)} "
                f"and other linguistic patterns in your text."
            )
        else:
            explanation = "Prediction based on overall text characteristics."
        
        return {
            'top_features': top_features,
            'feature_explanation': explanation,
            'importance_scores': importance_scores,
        }
    
    def generate_insights(
        self,
        text: str,
        predictions: Dict = None
    ) -> Dict[str, str]:
        """
        Generate behavioral insights from predictions.
        
        Args:
            text: Input text
            predictions: Personality predictions
            
        Returns:
            Dictionary with insight narratives
        """
        if predictions is None:
            predictions = self.predict_personality(text)
        
        # Extract key traits
        traits_high = [t for t, v in predictions.items() if v['score'] > 0.7]
        traits_low = [t for t, v in predictions.items() if v['score'] < 0.3]
        
        # Generate narrative
        narrative_parts = []
        
        if traits_high:
            high_str = ', '.join(traits_high)
            narrative_parts.append(
                f"You show strong {high_str.lower()} - this suggests you are "
                f"{'thoughtful and introspective' if 'Openness' in traits_high else ''}"
                f"{'organized and reliable' if 'Conscientiousness' in traits_high else ''}"
                f"{'socially engaged and expressive' if 'Extraversion' in traits_high else ''}"
                f"{'cooperative and empathetic' if 'Agreeableness' in traits_high else ''}"
                f"{'emotionally reactive and sensitive' if 'Neuroticism' in traits_high else ''}."
            )
        
        if traits_low:
            low_str = ', '.join(traits_low)
            narrative_parts.append(
                f"You demonstrate lower {low_str.lower()}, indicating you may be "
                f"{'more pragmatic and grounded' if 'Openness' in traits_low else ''}"
                f"{'more flexible and spontaneous' if 'Conscientiousness' in traits_low else ''}"
                f"{'more reserved and reflective' if 'Extraversion' in traits_low else ''}"
                f"{'more focused on personal interests' if 'Agreeableness' in traits_low else ''}"
                f"{'emotionally stable' if 'Neuroticism' in traits_low else ''}."
            )
        
        behavioral_text = ' '.join(narrative_parts) if narrative_parts else "You display balanced personality traits."
        
        # Communication style
        if predictions['Extraversion']['score'] > 0.6 and predictions['Agreeableness']['score'] > 0.5:
            comm_style = 'Collaborative and expressive'
        elif predictions['Conscientiousness']['score'] > 0.6:
            comm_style = 'Structured and methodical'
        elif predictions['Openness']['score'] > 0.6:
            comm_style = 'Creative and exploratory'
        else:
            comm_style = 'Balanced and adaptive'
        
        return {
            'behavioral_text': behavioral_text,
            'communication_style': {
                'primary': comm_style,
                'description': f"Your communication style is {comm_style.lower()}. "
                             f"You express yourself clearly and thoughtfully."
            },
            'key_characteristics': [
                f"{t}: {predictions[t]['label'].lower()}"
                for t in TRAITS
            ]
        }
    
    def compare_texts(
        self,
        text1: str,
        text2: str
    ) -> Dict[str, any]:
        """
        Compare personality from two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Comparison results
        """
        pred1 = self.predict_personality(text1)
        pred2 = self.predict_personality(text2)
        
        differences = {}
        for trait in TRAITS:
            diff = pred2[trait]['score'] - pred1[trait]['score']
            differences[trait] = {
                'text1_score': pred1[trait]['score'],
                'text2_score': pred2[trait]['score'],
                'difference': diff,
                'direction': 'higher' if diff > 0 else 'lower' if diff < 0 else 'same',
            }
        
        return {
            'predictions_1': pred1,
            'predictions_2': pred2,
            'differences': differences,
        }
    
    def detect_emotional_shifts(self, text: str) -> Dict[str, any]:
        """
        Detect emotional/sentiment shifts across text.
        
        Args:
            text: Input text
            
        Returns:
            Emotion shift analysis
        """
        from src.preprocessing import tokenize_sentences
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        sia = SentimentIntensityAnalyzer()
        sentences = tokenize_sentences(text)
        
        sentiments = []
        for sent in sentences:
            if sent.strip():
                sentiment = sia.polarity_scores(sent)['compound']
                sentiments.append(sentiment)
        
        if not sentiments:
            return {
                'shifts': [],
                'overall_trend': 'neutral',
                'volatility': 0,
            }
        
        # Detect shifts
        shifts = []
        for i in range(1, len(sentiments)):
            shift = sentiments[i] - sentiments[i-1]
            if abs(shift) > 0.3:
                shifts.append({
                    'position': i,
                    'from_sentiment': sentiments[i-1],
                    'to_sentiment': sentiments[i],
                    'magnitude': shift,
                })
        
        # Overall trend
        if len(sentiments) > 1:
            trend = sentiments[-1] - sentiments[0]
            if trend > 0.2:
                overall_trend = 'increasingly positive'
            elif trend < -0.2:
                overall_trend = 'increasingly negative'
            else:
                overall_trend = 'stable'
        else:
            overall_trend = 'insufficient data'
        
        return {
            'shifts': shifts,
            'overall_trend': overall_trend,
            'volatility': np.std(sentiments) if sentiments else 0,
            'average_sentiment': np.mean(sentiments) if sentiments else 0,
        }
