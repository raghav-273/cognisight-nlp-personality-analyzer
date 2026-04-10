"""
Cognisight - Advanced NLP-Based Conversation Personality Analyzer

This module provides comprehensive personality analysis from text conversations
using linguistic features and sentiment analysis to predict Big Five personality traits.

Author: AI Engineer
Version: 2.0.0
"""

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()


class TextPreprocessor:
    """Handles text cleaning and preprocessing operations."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize input text.

        Args:
            text: Raw input text

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep punctuation for analysis
        text = re.sub(r'[^a-zA-Z0-9\s.!?]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @staticmethod
    def tokenize_sentences(text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        if not text.strip():
            return []

        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except:
            # Fallback tokenization
            return [s.strip() for s in text.split('.') if s.strip()]

    @staticmethod
    def tokenize_words(text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of words
        """
        if not text.strip():
            return []

        try:
            words = word_tokenize(text)
            return [w.lower() for w in words if w.isalnum()]
        except:
            # Fallback tokenization
            return [w.lower() for w in text.split() if w.isalnum()]


class SentimentAnalyzer:
    """Advanced sentiment analysis with multiple model support."""

    def __init__(self):
        """Initialize sentiment analyzer."""
        self.vader_available = True
        self.roberta_available = False

        # Try to load RoBERTa model
        try:
            from transformers import pipeline
            self.roberta_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=False
            )
            self.roberta_available = True
        except:
            self.roberta_model = None

    def analyze_vader(self, sentences: List[str]) -> Dict[str, Union[float, List[float]]]:
        """
        Analyze sentiment using VADER.

        Args:
            sentences: List of sentences to analyze

        Returns:
            Dictionary with sentiment scores and statistics
        """
        if not sentences:
            return {"scores": [], "average": 0.0, "variance": 0.0}

        scores = []
        for sentence in sentences:
            if sentence.strip():
                score = sia.polarity_scores(sentence)['compound']
                scores.append(score)
            else:
                scores.append(0.0)

        average = np.mean(scores) if scores else 0.0
        variance = np.var(scores) if len(scores) > 1 else 0.0

        return {
            "scores": scores,
            "average": average,
            "variance": variance,
            "emotional_stability": 1.0 - min(variance, 1.0)  # Higher = more stable
        }

    def analyze_roberta(self, sentences: List[str]) -> Dict[str, Union[float, List[float]]]:
        """
        Analyze sentiment using RoBERTa model.

        Args:
            sentences: List of sentences to analyze

        Returns:
            Dictionary with sentiment scores and statistics
        """
        if not self.roberta_available or not sentences:
            return self.analyze_vader(sentences)  # Fallback

        scores = []
        for sentence in sentences:
            if sentence.strip():
                try:
                    result = self.roberta_model(sentence)[0]
                    label, confidence = result['label'], result['score']

                    # Convert to compound-like score (-1 to 1)
                    if label == 'LABEL_2':  # Positive
                        score = confidence
                    elif label == 'LABEL_0':  # Negative
                        score = -confidence
                    else:  # Neutral
                        score = 0.0
                    scores.append(score)
                except:
                    scores.append(0.0)
            else:
                scores.append(0.0)

        average = np.mean(scores) if scores else 0.0
        variance = np.var(scores) if len(scores) > 1 else 0.0

        return {
            "scores": scores,
            "average": average,
            "variance": variance,
            "emotional_stability": 1.0 - min(variance, 1.0)
        }

    def analyze_combined(self, sentences: List[str]) -> Dict[str, Union[float, List[float]]]:
        """
        Analyze sentiment using combined VADER + RoBERTa approach.

        Args:
            sentences: List of sentences to analyze

        Returns:
            Dictionary with combined sentiment analysis
        """
        vader_result = self.analyze_vader(sentences)
        roberta_result = self.analyze_roberta(sentences)

        # Weighted combination (60% RoBERTa, 40% VADER for better accuracy)
        combined_scores = []
        for v, r in zip(vader_result["scores"], roberta_result["scores"]):
            combined = (r * 0.6) + (v * 0.4)
            combined_scores.append(combined)

        average = np.mean(combined_scores) if combined_scores else 0.0
        variance = np.var(combined_scores) if len(combined_scores) > 1 else 0.0

        # Calculate model agreement
        agreements = 0
        for v, r in zip(vader_result["scores"], roberta_result["scores"]):
            if (v > 0.1 and r > 0.1) or (v < -0.1 and r < -0.1) or (abs(v) < 0.1 and abs(r) < 0.1):
                agreements += 1

        agreement = agreements / len(vader_result["scores"]) if vader_result["scores"] else 1.0

        return {
            "scores": combined_scores,
            "average": average,
            "variance": variance,
            "emotional_stability": 1.0 - min(variance, 1.0),
            "vader_scores": vader_result["scores"],
            "roberta_scores": roberta_result["scores"],
            "model_agreement": agreement,
            "confidence": agreement  # Higher agreement = higher confidence
        }

    def detect_emotional_shifts(self, scores: List[float], threshold: float = 0.3) -> List[Dict]:
        """
        Detect significant emotional shifts in sentiment scores.

        Args:
            scores: List of sentiment scores
            threshold: Minimum change to consider a shift

        Returns:
            List of detected shifts with positions and magnitudes
        """
        shifts = []
        if len(scores) < 2:
            return shifts

        for i in range(1, len(scores)):
            change = scores[i] - scores[i-1]
            if abs(change) >= threshold:
                shift_type = "positive" if change > 0 else "negative"
                shifts.append({
                    "position": i,
                    "magnitude": abs(change),
                    "type": shift_type,
                    "from_score": scores[i-1],
                    "to_score": scores[i]
                })

        return shifts


class FeatureEngineer:
    """Extracts linguistic and psychological features from text."""

    def __init__(self):
        """Initialize feature engineer."""
        self.preprocessor = TextPreprocessor()

    def extract_features(self, text: str) -> Dict[str, Union[float, int, str]]:
        """
        Extract comprehensive linguistic features for personality analysis.

        Args:
            text: Cleaned input text

        Returns:
            Dictionary of linguistic features
        """
        if not text or not text.strip():
            return self._get_empty_features()

        words = self.preprocessor.tokenize_words(text)
        sentences = self.preprocessor.tokenize_sentences(text)

        if not words:
            return self._get_empty_features()

        # Basic counts
        total_words = len(words)
        total_sentences = len(sentences)
        unique_words = len(set(words))

        # Punctuation analysis
        questions = text.count('?')
        exclamations = text.count('!')
        periods = text.count('.')

        # Pronoun analysis
        pronouns = self._analyze_pronouns(words)

        # Word characteristics
        word_lengths = [len(word) for word in words]
        avg_word_length = np.mean(word_lengths)

        # Sentence characteristics
        sentence_lengths = [len(sent.split()) for sent in sentences] if sentences else [0]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        sentence_variability = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0

        # Lexical features
        lexical_diversity = unique_words / total_words if total_words > 0 else 0

        # Readability (simplified Flesch-Kincaid)
        readability = self._calculate_readability(avg_sentence_length, total_words, unique_words)

        # Emotional intensity
        emotional_intensity = (questions + exclamations) / max(total_words, 1) * 100

        # Cognitive complexity
        cognitive_complexity = self._calculate_cognitive_complexity(
            lexical_diversity, sentence_variability, avg_word_length
        )

        # Social orientation
        social_orientation = self._calculate_social_orientation(pronouns)

        # Communication style
        communication_style = self._classify_communication_style(
            questions, exclamations, periods, avg_sentence_length
        )

        # NEW FEATURES: Sentiment variance and keyword intensity
        sentiment_variance = self._calculate_sentiment_variance(words)
        keyword_intensity = self._calculate_keyword_intensity(words)

        return {
            # Basic metrics
            "word_count": total_words,
            "sentence_count": total_sentences,
            "unique_words": unique_words,

            # Ratios and proportions
            "question_ratio": questions / max(periods + 1, 1),
            "lexical_diversity": lexical_diversity,

            # Length metrics
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "sentence_variability": sentence_variability,

            # Pronoun analysis
            "self_focus": pronouns["first_person_singular"],
            "social_focus": pronouns["second_person"] + pronouns["first_person_plural"],

            # Communication features
            "emotional_intensity": emotional_intensity,
            "readability_score": readability,
            "cognitive_complexity": cognitive_complexity,
            "social_orientation": social_orientation,
            "communication_style": communication_style,

            # NEW FEATURES
            "sentiment_variance": sentiment_variance,
            "keyword_intensity": keyword_intensity
        }

    def _get_empty_features(self) -> Dict[str, Union[float, int, str]]:
        """Return empty feature set for invalid input."""
        return {
            "word_count": 0, "sentence_count": 0, "unique_words": 0,
            "question_ratio": 0.0, "lexical_diversity": 0.0,
            "avg_word_length": 0.0, "avg_sentence_length": 0.0, "sentence_variability": 0.0,
            "self_focus": 0, "social_focus": 0,
            "emotional_intensity": 0.0, "readability_score": 0.0,
            "cognitive_complexity": 0.0, "social_orientation": 0.0,
            "communication_style": "neutral",
            "sentiment_variance": 0.0, "keyword_intensity": 0.0
        }

    def _analyze_pronouns(self, words: List[str]) -> Dict[str, int]:
        """Analyze pronoun usage in text."""
        first_person_singular = sum(1 for w in words if w in ['i', 'me', 'my', 'mine', 'myself'])
        first_person_plural = sum(1 for w in words if w in ['we', 'us', 'our', 'ours', 'ourselves'])
        second_person = sum(1 for w in words if w in ['you', 'your', 'yours', 'yourself', 'yourselves'])
        third_person = sum(1 for w in words if w in ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs', 'themselves', 'it', 'its'])

        return {
            "first_person_singular": first_person_singular,
            "first_person_plural": first_person_plural,
            "second_person": second_person,
            "third_person": third_person
        }

    def _calculate_readability(self, avg_sentence_length: float, total_words: int, unique_words: int) -> float:
        """Calculate readability score using simplified Flesch-Kincaid formula."""
        if total_words == 0 or unique_words == 0:
            return 0.0

        # Simplified readability score (0-100 scale)
        score = 206.835 - 1.015 * avg_sentence_length - 84.6 * (total_words / unique_words)
        return max(0.0, min(100.0, score))

    def _calculate_cognitive_complexity(self, lexical_diversity: float,
                                      sentence_variability: float,
                                      avg_word_length: float) -> float:
        """Calculate cognitive complexity score."""
        complexity = (lexical_diversity * 50) + (sentence_variability * 2) + (avg_word_length * 10)
        return min(complexity, 100.0)

    def _calculate_social_orientation(self, pronouns: Dict[str, int]) -> float:
        """Calculate social orientation score (0-1 scale)."""
        social_pronouns = pronouns["second_person"] + pronouns["first_person_plural"]
        self_pronouns = pronouns["first_person_singular"]
        total_pronouns = social_pronouns + self_pronouns

        return social_pronouns / max(total_pronouns, 1)

    def _classify_communication_style(self, questions: int, exclamations: int,
                                    periods: int, avg_sentence_length: float) -> str:
        """Classify communication style based on linguistic patterns."""
        if questions > periods and questions > exclamations:
            return "inquisitive"
        elif exclamations > periods and exclamations > questions:
            return "expressive"
        elif avg_sentence_length > 20:
            return "elaborate"
        elif avg_sentence_length < 10:
            return "concise"
        else:
            return "balanced"

    def _calculate_sentiment_variance(self, words: List[str]) -> float:
        """
        Calculate sentiment variance of words (NEW FEATURE).
        Measures emotional consistency in word choice.
        """
        if not words:
            return 0.0

        word_sentiments = []
        for word in words[:100]:  # Limit for performance
            try:
                sentiment = sia.polarity_scores(word)['compound']
                word_sentiments.append(sentiment)
            except:
                continue

        if len(word_sentiments) < 2:
            return 0.0

        return np.var(word_sentiments)

    def _calculate_keyword_intensity(self, words: List[str]) -> float:
        """
        Calculate keyword intensity (NEW FEATURE).
        Measures focus on emotionally charged or significant words.
        """
        if not words:
            return 0.0

        # Define emotionally significant words
        emotional_words = {
            'love', 'hate', 'amazing', 'terrible', 'excited', 'angry', 'happy', 'sad',
            'wonderful', 'awful', 'fantastic', 'horrible', 'delighted', 'furious',
            'thrilled', 'depressed', 'ecstatic', 'devastated', 'joy', 'sorrow'
        }

        # Count emotional words
        emotional_count = sum(1 for word in words if word in emotional_words)

        # Calculate intensity as ratio of emotional words
        intensity = emotional_count / len(words) if words else 0.0

        return min(intensity * 100, 100.0)  # Scale to 0-100


class PersonalityScorer:
    """Calculates Big Five personality traits from linguistic features."""

    def __init__(self):
        """Initialize personality scorer."""
        self.feature_engineer = FeatureEngineer()

    def score_personality(self, features: Dict, sentiment_avg: float = 0.0,
                         sentiment_variance: float = 0.0) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Calculate enhanced personality traits with practical labels.

        Args:
            features: Linguistic features from FeatureEngineer
            sentiment_avg: Average sentiment score
            sentiment_variance: Sentiment variance for emotional stability

        Returns:
            Dictionary with trait scores, confidence levels, and practical labels
        """
        # Normalize features to 0-1 scale for consistent scoring
        normalized_features = self._normalize_features(features)

        # Calculate Big Five trait scores
        big_five = self._calculate_big_five_scores(normalized_features, sentiment_avg, sentiment_variance)

        # Calculate additional practical traits
        practical_traits = self._calculate_practical_traits(normalized_features, sentiment_avg, sentiment_variance)

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence(big_five, features)

        # Combine all traits with practical labels
        result = {}

        # Big Five with enhanced labels
        for trait, score in big_five.items():
            confidence = confidence_scores.get(trait, 0.5)
            practical_label = self._get_practical_label(trait, score, features)

            result[trait] = {
                "score": round(score, 2),
                "confidence": round(confidence, 2),
                "label": practical_label,
                "description": self._get_enhanced_description(trait, score, practical_label)
            }

        # Add practical traits
        for trait, score in practical_traits.items():
            practical_label = self._get_practical_trait_label(trait, score)

            result[trait] = {
                "score": round(score, 2),
                "confidence": 0.7,  # Default confidence for practical traits
                "label": practical_label,
                "description": self._get_practical_description(trait, score, practical_label)
            }

        return result

    def _normalize_features(self, features: Dict) -> Dict[str, float]:
        """Normalize features to 0-1 scale."""
        normalized = {}

        # Define normalization ranges (approximate)
        ranges = {
            "lexical_diversity": (0, 1),
            "cognitive_complexity": (0, 100),
            "emotional_intensity": (0, 20),
            "readability_score": (0, 100),
            "social_orientation": (0, 1),
            "question_ratio": (0, 2),
            "sentiment_variance": (0, 1),
            "keyword_intensity": (0, 100)
        }

        for feature, value in features.items():
            if feature in ranges:
                min_val, max_val = ranges[feature]
                if isinstance(value, (int, float)):
                    normalized[feature] = max(0, min(1, (value - min_val) / (max_val - min_val)))
                else:
                    normalized[feature] = 0.5  # Default for invalid values
            elif isinstance(value, (int, float)):
                # Default normalization for numeric features not in ranges
                normalized[feature] = min(1.0, max(0.0, value / 100))
            else:
                # Skip non-numeric features
                continue

        return normalized

        return normalized

        return normalized

    def _calculate_big_five_scores(self, features: Dict, sentiment_avg: float,
                                   sentiment_variance: float) -> Dict[str, float]:
        """Calculate Big Five trait scores using weighted feature combinations."""
        # Openness: Curiosity, vocabulary, cognitive complexity
        openness = (
            features.get("lexical_diversity", 0) * 0.4 +
            features.get("cognitive_complexity", 0) * 0.3 +
            features.get("question_ratio", 0) * 0.3
        )

        # Conscientiousness: Organization, clarity, emotional control
        conscientiousness = (
            features.get("readability_score", 0) * 0.4 +
            (1 - features.get("emotional_intensity", 0)) * 0.3 +
            (1 - features.get("sentiment_variance", 0)) * 0.3
        )

        # Extraversion: Social focus, emotional expression, engagement
        extraversion = (
            features.get("social_orientation", 0) * 0.4 +
            features.get("emotional_intensity", 0) * 0.3 +
            abs(sentiment_avg) * 0.3
        )

        # Agreeableness: Positive sentiment, social harmony, cooperation
        agreeableness = (
            (1 - abs(sentiment_avg)) * 0.4 +
            features.get("social_orientation", 0) * 0.3 +
            (1 - features.get("keyword_intensity", 0)) * 0.3
        )

        # Neuroticism: Emotional volatility, negative expression
        neuroticism = (
            abs(sentiment_avg) * 0.3 +
            features.get("emotional_intensity", 0) * 0.3 +
            features.get("sentiment_variance", 0) * 0.4
        )

        # Scale to 0-100
        traits = {
            "Openness": min(100, openness * 100),
            "Conscientiousness": min(100, conscientiousness * 100),
            "Extraversion": min(100, extraversion * 100),
            "Agreeableness": min(100, agreeableness * 100),
            "Neuroticism": min(100, neuroticism * 100)
        }

        return traits

    def _calculate_practical_traits(self, features: Dict, sentiment_avg: float,
                                   sentiment_variance: float) -> Dict[str, float]:
        """Calculate practical behavioral traits."""
        # Emotional Stability: Low variance + balanced sentiment
        emotional_stability = (
            (1 - features.get("sentiment_variance", 0)) * 0.5 +
            (1 - abs(sentiment_avg)) * 0.3 +
            (1 - features.get("emotional_intensity", 0)) * 0.2
        )

        # Analytical Thinking: High complexity + lexical diversity
        analytical_thinking = (
            features.get("cognitive_complexity", 0) * 0.4 +
            features.get("lexical_diversity", 0) * 0.3 +
            features.get("readability_score", 0) * 0.3
        )

        # Social Orientation: Social pronouns + question ratio
        social_orientation = (
            features.get("social_orientation", 0) * 0.5 +
            features.get("question_ratio", 0) * 0.3 +
            (1 - features.get("keyword_intensity", 0)) * 0.2
        )

        # Communication Style Score: Based on sentence structure
        comm_style_score = (
            features.get("avg_sentence_length", 0) * 0.4 +
            features.get("question_ratio", 0) * 0.3 +
            features.get("emotional_intensity", 0) * 0.3
        )

        # Scale to 0-100
        practical = {
            "Emotional Stability": min(100, emotional_stability * 100),
            "Analytical Thinking": min(100, analytical_thinking * 100),
            "Social Orientation": min(100, social_orientation * 100),
            "Communication Style": min(100, comm_style_score * 100)
        }

        return practical

    def _calculate_confidence(self, traits: Dict[str, float], features: Dict) -> Dict[str, float]:
        """Calculate confidence scores for each trait based on feature consistency."""
        confidence_scores = {}

        # Base confidence on feature completeness and consistency
        feature_count = sum(1 for v in features.values() if v != 0)
        base_confidence = min(1.0, feature_count / 10)  # 10 key features

        for trait in traits:
            # Adjust confidence based on trait score extremity
            score = traits[trait]
            extremity_factor = 1 - abs(score - 50) / 50  # Less extreme = more confident

            confidence_scores[trait] = base_confidence * extremity_factor

        return confidence_scores

    def _get_practical_label(self, trait: str, score: float, features: Dict) -> str:
        """Get practical, meaningful labels for Big Five traits."""
        if trait == "Openness":
            if score >= 70:
                return "Highly Creative & Curious"
            elif score >= 50:
                return "Open to New Ideas"
            else:
                return "Practical & Traditional"
        elif trait == "Conscientiousness":
            if score >= 70:
                return "Highly Organized"
            elif score >= 50:
                return "Reliable & Structured"
            else:
                return "Flexible & Spontaneous"
        elif trait == "Extraversion":
            if score >= 70:
                return "Very Outgoing"
            elif score >= 50:
                return "Socially Engaged"
            else:
                return "Reserved & Introspective"
        elif trait == "Agreeableness":
            if score >= 70:
                return "Very Cooperative"
            elif score >= 50:
                return "Team Player"
            else:
                return "Independent & Direct"
        elif trait == "Neuroticism":
            if score >= 70:
                return "Emotionally Reactive"
            elif score >= 50:
                return "Sensitive to Stress"
            else:
                return "Calm & Composed"
        return "Balanced"

    def _get_practical_trait_label(self, trait: str, score: float) -> str:
        """Get labels for practical traits."""
        if trait == "Emotional Stability":
            if score >= 70:
                return "Very Stable"
            elif score >= 50:
                return "Generally Calm"
            else:
                return "Easily Stressed"
        elif trait == "Analytical Thinking":
            if score >= 70:
                return "Highly Analytical"
            elif score >= 50:
                return "Logical Thinker"
            else:
                return "Intuitive Approach"
        elif trait == "Social Orientation":
            if score >= 70:
                return "Very Social"
            elif score >= 50:
                return "Socially Aware"
            else:
                return "Prefers Solitude"
        elif trait == "Communication Style":
            if score >= 70:
                return "Detailed & Elaborate"
            elif score >= 50:
                return "Clear & Balanced"
            else:
                return "Direct & Concise"
        return "Balanced"

    def _get_enhanced_description(self, trait: str, score: float, label: str) -> str:
        """Get enhanced, practical descriptions."""
        descriptions = {
            "Openness": f"You show {label.lower()} tendencies. This suggests you're {'very imaginative and open to new experiences' if score >= 70 else 'moderately curious about new ideas' if score >= 50 else 'more comfortable with familiar approaches'}.",
            "Conscientiousness": f"You demonstrate {label.lower()} behavior. This indicates you're {'highly disciplined and organized' if score >= 70 else 'generally reliable and structured' if score >= 50 else 'more flexible in your approach'}.",
            "Extraversion": f"You appear {label.lower()}. This means you're {'very energized by social interactions' if score >= 70 else 'comfortable in social settings' if score >= 50 else 'more comfortable in quieter environments'}.",
            "Agreeableness": f"You show {label.lower()} traits. This suggests you're {'very focused on harmony and cooperation' if score >= 70 else 'generally considerate of others' if score >= 50 else 'more direct and independent'}.",
            "Neuroticism": f"You display {label.lower()} patterns. This indicates you're {'quite sensitive to emotional stimuli' if score >= 70 else 'moderately affected by stress' if score >= 50 else 'generally calm under pressure'}."
        }
        return descriptions.get(trait, f"You show {label.lower()} characteristics in this trait.")

    def _get_practical_description(self, trait: str, score: float, label: str) -> str:
        """Get descriptions for practical traits."""
        descriptions = {
            "Emotional Stability": f"You demonstrate {label.lower()} emotional responses. This suggests you're {'very resilient to stress' if score >= 70 else 'generally steady emotionally' if score >= 50 else 'more easily affected by emotional situations'}.",
            "Analytical Thinking": f"You show {label.lower()} thinking patterns. This indicates you're {'highly logical and detail-oriented' if score >= 70 else 'balanced between logic and intuition' if score >= 50 else 'more intuitive in your decision-making'}.",
            "Social Orientation": f"You exhibit {label.lower()} tendencies. This means you're {'very focused on relationships and social dynamics' if score >= 70 else 'aware of social contexts' if score >= 50 else 'more comfortable with independent activities'}.",
            "Communication Style": f"You use a {label.lower()} communication approach. This suggests your writing is {'very detailed and comprehensive' if score >= 70 else 'clear and well-structured' if score >= 50 else 'concise and to the point'}."
        }
        return descriptions.get(trait, f"You show {label.lower()} characteristics.")

    def _get_trait_description(self, trait: str, score: float) -> str:
        """Get detailed description for trait score."""
        descriptions = {
            "Openness": {
                "high": "Highly imaginative and open to new experiences",
                "moderate": "Moderately curious and open-minded",
                "low": "Prefers familiar routines and practical approaches"
            },
            "Conscientiousness": {
                "high": "Highly organized and disciplined",
                "moderate": "Moderately responsible and organized",
                "low": "More spontaneous and flexible"
            },
            "Extraversion": {
                "high": "Very outgoing and socially engaged",
                "moderate": "Socially balanced and adaptable",
                "low": "More reserved and introspective"
            },
            "Agreeableness": {
                "high": "Very cooperative and considerate",
                "moderate": "Generally helpful and understanding",
                "low": "More independent and assertive"
            },
            "Neuroticism": {
                "high": "Emotionally sensitive and responsive",
                "moderate": "Moderately emotionally responsive",
                "low": "Generally calm and emotionally stable"
            }
        }

        if score >= 60:
            level = "high"
        elif score >= 40:
            level = "moderate"
        else:
            level = "low"

        return descriptions.get(trait, {}).get(level, "Moderate level")


class PersonalityAnalyzer:
    """Main analyzer class that orchestrates the entire analysis pipeline."""

    def __init__(self):
        """Initialize the personality analyzer."""
        self.preprocessor = TextPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feature_engineer = FeatureEngineer()
        self.personality_scorer = PersonalityScorer()

    def analyze(self, text: str, model: str = "combined",
               include_wordcloud: bool = True) -> Dict:
        """
        Perform comprehensive personality analysis on input text.

        Args:
            text: Input text to analyze
            model: Sentiment model to use ("basic", "advanced", "combined")
            include_wordcloud: Whether to generate word cloud

        Returns:
            Dictionary containing complete analysis results
        """
        if not text or not text.strip():
            return self._get_error_result("No text provided for analysis")

        try:
            # Step 1: Preprocessing
            cleaned_text = self.preprocessor.clean_text(text)
            sentences = self.preprocessor.tokenize_sentences(cleaned_text)

            # Step 2: Sentiment Analysis
            if model == "combined":
                sentiment_results = self.sentiment_analyzer.analyze_combined(sentences)
            elif model == "advanced":
                sentiment_results = self.sentiment_analyzer.analyze_roberta(sentences)
            else:  # basic
                sentiment_results = self.sentiment_analyzer.analyze_vader(sentences)

            # Step 3: Feature Engineering
            features = self.feature_engineer.extract_features(cleaned_text)

            # Step 4: Personality Scoring
            personality_scores = self.personality_scorer.score_personality(
                features,
                sentiment_results.get("average", 0),
                sentiment_results.get("variance", 0)
            )

            # Step 5: Generate Insights
            insights = self._generate_insights(features, personality_scores)

            # Step 6: Emotional Shift Detection
            emotional_shifts = self.sentiment_analyzer.detect_emotional_shifts(
                sentiment_results.get("scores", [])
            )

            # Step 7: Word Cloud (optional)
            wordcloud_img = None
            if include_wordcloud:
                wordcloud_img = self._generate_wordcloud(cleaned_text)

            # Step 8: Compile Results
            return {
                "success": True,
                "text_analysis": {
                    "original_length": len(text),
                    "cleaned_length": len(cleaned_text),
                    "sentence_count": len(sentences),
                    "word_count": features.get("word_count", 0)
                },
                "sentiment": sentiment_results,
                "features": features,
                "personality": personality_scores,
                "insights": insights,
                "emotional_shifts": emotional_shifts,
                "wordcloud": wordcloud_img,
                "metadata": {
                    "model_used": model,
                    "analysis_timestamp": None,
                    "version": "2.0.0"
                }
            }

        except Exception as e:
            return self._get_error_result(f"Analysis failed: {str(e)}")

    def compare_texts(self, text1: str, text2: str, model: str = "combined") -> Dict:
        """
        Compare personality profiles between two texts.

        Args:
            text1: First text to analyze
            text2: Second text to analyze
            model: Sentiment model to use

        Returns:
            Dictionary containing comparison results
        """
        analysis1 = self.analyze(text1, model, include_wordcloud=False)
        analysis2 = self.analyze(text2, model, include_wordcloud=False)

        if not analysis1.get("success") or not analysis2.get("success"):
            return self._get_error_result("Comparison failed - invalid input texts")

        # Calculate differences
        differences = {}
        for trait in analysis1["personality"]:
            score1 = analysis1["personality"][trait]["score"]
            score2 = analysis2["personality"][trait]["score"]
            diff = score2 - score1
            differences[trait] = {
                "text1_score": score1,
                "text2_score": score2,
                "difference": diff,
                "direction": "higher in text 2" if diff > 0 else "higher in text 1" if diff < 0 else "equal"
            }

        return {
            "success": True,
            "text1_analysis": analysis1,
            "text2_analysis": analysis2,
            "comparison": differences,
            "similarity_score": self._calculate_similarity(analysis1, analysis2)
        }

    def _generate_insights(self, features: Dict, personality: Dict) -> List[str]:
        """Generate human-readable insights from features and personality scores."""
        insights = []

        # Communication style insights
        comm_style = features.get("communication_style", "balanced")
        style_insights = {
            "inquisitive": "Highly curious communication style with frequent questions",
            "expressive": "Expressive style with strong emotional emphasis",
            "elaborate": "Detailed and comprehensive communication approach",
            "concise": "Direct and efficient communication style",
            "balanced": "Well-balanced communication approach"
        }
        insights.append(style_insights.get(comm_style, "Balanced communication style"))

        # Social orientation
        social_ori = features.get("social_orientation", 0)
        if social_ori > 0.6:
            insights.append("Strong focus on social interactions and relationships")
        elif social_ori > 0.3:
            insights.append("Balanced focus between personal and social topics")
        else:
            insights.append("Primarily focused on personal experiences")

        # Emotional stability (NEW)
        sentiment_var = features.get("sentiment_variance", 0)
        if sentiment_var < 0.1:
            insights.append("Highly emotionally stable and consistent")
        elif sentiment_var < 0.3:
            insights.append("Moderately emotionally stable")
        else:
            insights.append("Emotionally variable and responsive")

        # Cognitive complexity
        cog_complex = features.get("cognitive_complexity", 0)
        if cog_complex > 70:
            insights.append("High cognitive complexity with sophisticated thinking")
        elif cog_complex > 40:
            insights.append("Moderate cognitive complexity")
        else:
            insights.append("Straightforward and practical thinking style")

        # Add personality-based insights
        for trait, data in personality.items():
            score = data["score"]
            if score >= 75:
                insights.append(f"Exceptionally {trait.lower()}: {data['description']}")
            elif score <= 25:
                insights.append(f"Particularly low {trait.lower()}: {data['description']}")

        return insights

    def _generate_wordcloud(self, text: str) -> Optional[str]:
        """Generate word cloud visualization."""
        if not text.strip():
            return None

        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            import io
            import base64

            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                contour_width=1,
                contour_color='steelblue'
            ).generate(text)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            return f"data:image/png;base64,{image_base64}"
        except:
            return None

    def _calculate_similarity(self, analysis1: Dict, analysis2: Dict) -> float:
        """Calculate similarity score between two analyses (0-1 scale)."""
        if not analysis1.get("personality") or not analysis2.get("personality"):
            return 0.0

        total_diff = 0
        trait_count = 0

        for trait in analysis1["personality"]:
            if trait in analysis2["personality"]:
                score1 = analysis1["personality"][trait]["score"]
                score2 = analysis2["personality"][trait]["score"]
                total_diff += abs(score1 - score2)
                trait_count += 1

        if trait_count == 0:
            return 0.0

        avg_diff = total_diff / trait_count
        similarity = 1 - (avg_diff / 100)  # Convert to 0-1 scale

        return max(0.0, min(1.0, similarity))

    def _get_error_result(self, message: str) -> Dict:
        """Return standardized error result."""
        return {
            "success": False,
            "error": message,
            "text_analysis": {},
            "sentiment": {},
            "features": {},
            "personality": {},
            "insights": [],
            "emotional_shifts": [],
            "wordcloud": None,
            "metadata": {"version": "2.0.0"}
        }


# Legacy function for backward compatibility
def analyze_text(text: str, sentiment_model: str = "combined",
                include_wordcloud: bool = True) -> Dict:
    """
    Legacy function for backward compatibility.
    Use PersonalityAnalyzer().analyze() for new code.
    """
    analyzer = PersonalityAnalyzer()
    return analyzer.analyze(text, sentiment_model, include_wordcloud)