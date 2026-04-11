"""
Main analysis interface for Cognisight.

The core model is unchanged. This layer turns the model's latent personality
signals into a hybrid journaling assistant by combining text analysis with a
lightweight MBTI micro-questionnaire.
"""

import html
import math
import os
import pickle
import re
from typing import Dict, List

import numpy as np
import pandas as pd

from src.preprocessing import tokenize_sentences
from src.feature_extractor import PersonalityFeatureExtractor
from src.interpretation import (
    BehavioralInsightGenerator,
    ConfidenceCalibrator,
    FeatureImportanceExplainer,
    PersonalityInterpreter,
    ReflectionPromptGenerator,
)
from utils import TRAITS, get_config, mbti_to_big_five


MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

QUESTION_DIMENSIONS = {
    "IE": ("I", "E"),
    "NS": ("N", "S"),
    "TF": ("T", "F"),
    "JP": ("J", "P"),
}


class PersonalityAnalyzer:
    """Primary inference interface used by the app."""

    def __init__(self, model_path: str = None):
        self.scaler = None
        self.model = self._load_model(model_path)
        self.feature_extractor = self._load_feature_extractor()
        self.interpreter = PersonalityInterpreter()
        self.behavior_generator = BehavioralInsightGenerator()
        self.feature_explainer = FeatureImportanceExplainer()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.prompt_generator = ReflectionPromptGenerator()
        self.safety_patterns = [
            r"\bkill myself\b",
            r"\bwant to die\b",
            r"\bend my life\b",
            r"\bsuicide\b",
            r"\bhurt myself\b",
            r"\bself harm\b",
        ]

    def _load_model(self, path: str = None):
        if path is None:
            path = os.path.join(get_config("model_dir"), "best_model.pkl")

        candidate_paths = [
            path,
            os.path.join("models", "best_model.pkl"),
            os.path.join(get_config("model_dir"), "random_forest.pkl"),
            os.path.join("models", "random_forest.pkl"),
        ]

        model = None
        last_module_error = None
        resolved_path = None
        for candidate in candidate_paths:
            if not os.path.exists(candidate):
                continue
            try:
                with open(candidate, "rb") as file:
                    model = pickle.load(file)
                resolved_path = candidate
                break
            except ModuleNotFoundError as exc:
                last_module_error = exc

        if model is None:
            if last_module_error is not None:
                missing_module = getattr(last_module_error, "name", "required dependency")
                raise RuntimeError(
                    f"Could not load the trained model because '{missing_module}' is missing. "
                    "Install project dependencies from requirements.txt before running inference."
                ) from last_module_error
            raise FileNotFoundError(f"Model not found in any expected location: {candidate_paths}")

        if hasattr(model, "scaler"):
            self.scaler = model.scaler

        if hasattr(model, "estimators_"):
            for estimator in model.estimators_:
                if not hasattr(estimator, "monotonic_cst"):
                    estimator.monotonic_cst = None

        self.model_source = resolved_path or path
        return model

    def _load_feature_extractor(self) -> PersonalityFeatureExtractor:
        extractor_path = os.path.join(get_config("model_dir"), "feature_extractor.pkl")
        fallback_path = os.path.join("models", "feature_extractor.pkl")

        for path in (extractor_path, fallback_path):
            if os.path.exists(path):
                with open(path, "rb") as file:
                    return pickle.load(file)

        return PersonalityFeatureExtractor()

    def analyze(
        self,
        text: str,
        questionnaire: Dict[str, float] = None,
        model: str = "combined",
        include_wordcloud: bool = False,
    ) -> Dict:
        del model, include_wordcloud

        cleaned_text = (text or "").strip()
        min_length = get_config("min_text_length")
        if len(cleaned_text) < min_length:
            return {"success": False, "error": f"Text must be at least {min_length} characters."}

        text_analysis = {
            "character_count": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "sentence_count": len(tokenize_sentences(cleaned_text)),
        }

        safety_flags = self._check_safety(cleaned_text)
        if safety_flags:
            return {
                "success": True,
                "safe_mode": True,
                "text_analysis": text_analysis,
                "support_message": (
                    "This text suggests you may be going through something intense. "
                    "It might help to talk to someone you trust or seek support."
                ),
                "disclaimer": (
                    "This tool is for self-reflection only and not a substitute for professional mental health support."
                ),
                "safety_flags": safety_flags,
            }

        quality_result = self._check_quality(cleaned_text)
        if not quality_result["usable"]:
            return {
                "success": True,
                "safe_mode": False,
                "low_signal": True,
                "text_analysis": text_analysis,
                "mental_state": {
                    "label": "Unclear",
                    "summary": "This entry is too repetitive or too narrow to support a reliable reading.",
                },
                "thought_patterns": quality_result["patterns"],
                "suggestions": quality_result["suggestions"],
                "reflection_summary": quality_result["summary"],
                "disclaimer": (
                    "This tool is for self-reflection only and not a substitute for professional mental health support."
                ),
            }

        try:
            features = self.feature_extractor.extract_features(cleaned_text)
            model_input = features.reshape(1, -1)
            if self.scaler is not None:
                model_input = self.scaler.transform(model_input)

            raw_predictions = np.asarray(self.model.predict(model_input)[0], dtype=float)
            latent_profile = self._process_predictions(raw_predictions)
            feature_values = self.feature_extractor.extract_linguistic_features(cleaned_text)
            text_distribution = self._mbti_distribution_from_profile(latent_profile)
            text_strength = self._text_strength(text_analysis, feature_values)

            questionnaire = questionnaire or {}
            questionnaire_distribution = self._questionnaire_distribution(questionnaire) if questionnaire else None
            text_weight, questionnaire_weight = self._fusion_weights(text_strength, questionnaire_distribution is not None)
            fused_distribution = self._fuse_distributions(
                text_distribution,
                questionnaire_distribution,
                text_weight,
                questionnaire_weight,
            )
            mbti_matches = self._top_mbti_matches(fused_distribution)

            mental_state = self.interpreter.classify_mental_state(latent_profile, feature_values)
            emotional_analysis = self.interpreter.emotional_analysis(latent_profile, feature_values)
            thought_patterns = self.interpreter.thought_patterns(feature_values)
            communication_style = self.interpreter.communication_style(latent_profile, feature_values, cleaned_text)
            mental_signals = self.interpreter.mental_signals(latent_profile, feature_values)
            strengths = self.interpreter.strengths(latent_profile, feature_values)
            suggestions = self.interpreter.suggestions(latent_profile, feature_values)
            reflection_prompts = self.prompt_generator.generate_prompts(latent_profile, mental_state["label"], thought_patterns, len(cleaned_text))
            self_awareness = self._self_awareness_score(latent_profile, feature_values)

            return {
                "success": True,
                "safe_mode": False,
                "low_signal": False,
                "text_analysis": text_analysis,
                "reflection_summary": self.behavior_generator.reflection_summary(mental_state, thought_patterns),
                "mental_state": mental_state,
                "mbti_matches": mbti_matches,
                "mbti_primary": mbti_matches[0],
                "mbti_insights": self.interpreter.mbti_type_explanations(mbti_matches),
                "emotional_analysis": emotional_analysis,
                "thought_patterns": thought_patterns,
                "communication_style": communication_style,
                "mental_signals": mental_signals,
                "strengths": strengths,
                "suggestions": suggestions,
                "reflection_prompts": reflection_prompts,
                "growth_mode": suggestions[:3],
                "self_awareness": self_awareness,
                "fusion": {
                    "text_weight": text_weight,
                    "questionnaire_weight": questionnaire_weight,
                    "text_strength": text_strength,
                    "questionnaire_dimensions": self._questionnaire_summary(questionnaire),
                },
                "timeline": self._build_timeline(cleaned_text),
                "highlighted_text_html": self._build_highlighted_text(cleaned_text),
                "highlight_legend": [
                    "Pink highlights emotional wording.",
                    "Amber highlights repeated wording.",
                    "Blue highlights informal or slang-like wording.",
                ],
                "key_signals": self.feature_explainer.get_top_contributing_features(feature_values, n=5),
                "confidence_note": self.interpreter.confidence_context(),
                "disclaimer": (
                    "This tool is for self-reflection only and not a substitute for professional mental health support."
                ),
                "latent_profile": latent_profile,
                "metadata": {
                    "model_used": type(self.model).__name__,
                    "model_path": self.model_source,
                    "analysis_timestamp": str(pd.Timestamp.now()),
                    "version": "8.0.0",
                },
            }
        except Exception as exc:
            return {"success": False, "error": f"Analysis failed: {exc}"}

    def compare_texts(self, text1: str, text2: str) -> Dict:
        result1 = self.analyze(text1)
        result2 = self.analyze(text2)

        if not result1.get("success") or not result2.get("success"):
            return {
                "success": False,
                "error": result1.get("error") or result2.get("error") or "Comparison failed.",
            }
        if result1.get("safe_mode") or result2.get("safe_mode"):
            return {
                "success": True,
                "safe_mode": True,
                "comparison_summary": "One of the entries triggered the safety layer, so comparison was paused.",
                "text1_analysis": result1,
                "text2_analysis": result2,
            }
        if result1.get("low_signal") or result2.get("low_signal"):
            return {
                "success": True,
                "low_signal": True,
                "comparison_summary": "One of the entries is too low-signal for a trustworthy comparison.",
                "text1_analysis": result1,
                "text2_analysis": result2,
            }

        return {
            "success": True,
            "safe_mode": False,
            "low_signal": False,
            "comparison_summary": self.behavior_generator.comparison_summary(
                result1["mental_state"], result2["mental_state"]
            ),
            "text1_analysis": result1,
            "text2_analysis": result2,
            "shift_cards": [
                {
                    "label": "Mental state",
                    "before": result1["mental_state"]["label"],
                    "after": result2["mental_state"]["label"],
                },
                {
                    "label": "Self-awareness",
                    "before": f"{result1['self_awareness']['score']}/100",
                    "after": f"{result2['self_awareness']['score']}/100",
                },
                {
                    "label": "Tone",
                    "before": result1["emotional_analysis"]["tone"],
                    "after": result2["emotional_analysis"]["tone"],
                },
                {
                    "label": "Primary MBTI fit",
                    "before": result1["mbti_primary"]["type"],
                    "after": result2["mbti_primary"]["type"],
                },
            ],
            "differences": self._compare_latent_profiles(
                result1["latent_profile"],
                result2["latent_profile"],
            ),
        }

    def _process_predictions(self, raw_predictions: np.ndarray) -> Dict[str, Dict]:
        results: Dict[str, Dict] = {}
        for index, trait in enumerate(TRAITS):
            if index >= len(raw_predictions):
                break
            score = float(np.clip(raw_predictions[index], 0.0, 1.0))
            results[trait] = {
                "score": score,
                "confidence": self.confidence_calibrator.normalize_confidence(score),
            }
        return results

    def _mbti_distribution_from_profile(self, profile: Dict[str, Dict]) -> Dict[str, float]:
        profile_vector = np.array([profile[trait]["score"] for trait in TRAITS], dtype=float)
        logits = {}
        for mbti_type in MBTI_TYPES:
            prototype = mbti_to_big_five(mbti_type)
            prototype_vector = np.array([prototype[trait] for trait in TRAITS], dtype=float)
            distance = np.linalg.norm(profile_vector - prototype_vector)
            logits[mbti_type] = -distance * 8.0
        return self._softmax_distribution(logits)

    def _questionnaire_distribution(self, questionnaire: Dict[str, float]) -> Dict[str, float]:
        if not questionnaire:
            return {}

        logits = {}
        for mbti_type in MBTI_TYPES:
            score = 1.0
            for dimension_key, (left_letter, right_letter) in QUESTION_DIMENSIONS.items():
                right_probability = float(questionnaire.get(dimension_key, 0.5))
                left_probability = 1.0 - right_probability
                letter = mbti_type["IENSPFTJ".find(left_letter)//2] if False else None
                letter = {
                    "IE": mbti_type[0],
                    "NS": mbti_type[1],
                    "TF": mbti_type[2],
                    "JP": mbti_type[3],
                }[dimension_key]
                score *= right_probability if letter == right_letter else left_probability
            logits[mbti_type] = math.log(score + 1e-6) * 1.8
        return self._softmax_distribution(logits)

    def _fusion_weights(self, text_strength: float, has_questionnaire: bool) -> tuple:
        if not has_questionnaire:
            return 1.0, 0.0
        if text_strength < 0.40:
            return 0.55, 0.45
        if text_strength < 0.60:
            return 0.62, 0.38
        return 0.70, 0.30

    def _fuse_distributions(
        self,
        text_distribution: Dict[str, float],
        questionnaire_distribution: Dict[str, float],
        text_weight: float,
        questionnaire_weight: float,
    ) -> Dict[str, float]:
        if not questionnaire_distribution:
            return self._sharpen_distribution(text_distribution, power=2.2)

        fused = {}
        for mbti_type in MBTI_TYPES:
            fused[mbti_type] = (
                text_weight * text_distribution.get(mbti_type, 0.0)
                + questionnaire_weight * questionnaire_distribution.get(mbti_type, 0.0)
            )
        return self._sharpen_distribution(fused, power=2.0)

    def _top_mbti_matches(self, distribution: Dict[str, float], n: int = 3) -> List[Dict]:
        ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:n]
        return [
            {"type": mbti_type, "probability": float(probability * 100)}
            for mbti_type, probability in ranked
        ]

    def _softmax_distribution(self, logits: Dict[str, float]) -> Dict[str, float]:
        values = np.array(list(logits.values()), dtype=float)
        values -= np.max(values)
        probs = np.exp(values)
        probs /= np.sum(probs) or 1.0
        return {mbti_type: float(prob) for mbti_type, prob in zip(logits.keys(), probs)}

    def _sharpen_distribution(self, distribution: Dict[str, float], power: float) -> Dict[str, float]:
        sharpened = {key: value ** power for key, value in distribution.items()}
        total = sum(sharpened.values()) or 1.0
        return {key: float(value / total) for key, value in sharpened.items()}

    def _questionnaire_summary(self, questionnaire: Dict[str, float]) -> List[Dict]:
        rows = []
        for key, value in questionnaire.items():
            left_letter, right_letter = QUESTION_DIMENSIONS[key]
            lean = right_letter if value > 0.58 else left_letter if value < 0.42 else "Balanced"
            rows.append(
                {
                    "dimension": key,
                    "left": left_letter,
                    "right": right_letter,
                    "score": float(value),
                    "lean": lean,
                }
            )
        return rows

    def _text_strength(self, text_analysis: Dict[str, int], features: Dict[str, float]) -> float:
        word_factor = min(1.0, max(0.0, (text_analysis["word_count"] - 40) / 160))
        lexical = min(1.0, max(0.0, features.get("ling_feat_1", 0.0)))
        repetition_health = min(1.0, max(0.0, features.get("ling_feat_25", 0.0)))
        return float(0.35 * word_factor + 0.30 * lexical + 0.35 * repetition_health)

    def _self_awareness_score(self, profile: Dict[str, Dict], features: Dict[str, float]) -> Dict[str, object]:
        structure_score = max(0.0, 1 - abs(features.get("ling_feat_2", 0.5) - 0.55) / 0.55)
        clarity = np.clip(
            0.40 * features.get("ling_feat_1", 0.0)
            + 0.35 * features.get("ling_feat_25", 0.0)
            + 0.25 * structure_score,
            0.0,
            1.0,
        )
        stability = np.clip(
            0.55 * (1 - min(1.0, features.get("ling_feat_5", 0.0) * 3))
            + 0.45 * (1 - profile.get("Neuroticism", {}).get("score", 0.5)),
            0.0,
            1.0,
        )
        reflection_depth = np.clip(
            0.30 * features.get("ling_feat_1", 0.0)
            + 0.20 * features.get("ling_feat_10", 0.0) * 4
            + 0.25 * features.get("ling_feat_24", 0.0)
            + 0.25 * profile.get("Openness", {}).get("score", 0.5),
            0.0,
            1.0,
        )
        score = int(round((0.35 * clarity + 0.30 * stability + 0.35 * reflection_depth) * 100))
        if score >= 75:
            label = "Strong"
        elif score >= 55:
            label = "Developing"
        else:
            label = "Emerging"
        return {
            "score": score,
            "label": label,
            "breakdown": {
                "clarity": int(round(clarity * 100)),
                "emotional_stability": int(round(stability * 100)),
                "reflection_depth": int(round(reflection_depth * 100)),
            },
        }

    def _build_timeline(self, text: str) -> List[Dict]:
        sentences = [sentence.strip() for sentence in tokenize_sentences(text) if sentence.strip()]
        timeline = []
        for index, sentence in enumerate(sentences, start=1):
            sentiment = self.feature_extractor.sia.polarity_scores(sentence)["compound"]
            label = "negative" if sentiment <= -0.2 else "positive" if sentiment >= 0.2 else "neutral"
            timeline.append(
                {
                    "step": index,
                    "sentence": sentence,
                    "sentiment": sentiment,
                    "label": label,
                }
            )
        return timeline

    def _build_highlighted_text(self, text: str) -> str:
        tokens = re.findall(r"\w+|\s+|[^\w\s]", text)
        word_counts = {}
        slang_terms = {"idk", "lol", "gonna", "wanna", "kinda", "sorta", "bro", "omg"}
        emotional_terms = {
            "love", "hate", "sad", "angry", "happy", "anxious", "overwhelmed",
            "stressed", "scared", "calm", "tired", "excited", "confused",
        }

        for token in tokens:
            lowered = token.lower()
            if lowered.isalpha():
                word_counts[lowered] = word_counts.get(lowered, 0) + 1

        parts = []
        for token in tokens:
            lowered = token.lower()
            escaped = html.escape(token)
            if not lowered.isalpha():
                parts.append(escaped)
                continue

            css = ""
            title = ""
            if lowered in emotional_terms:
                css = "background: rgba(255, 158, 209, 0.26); border-bottom: 2px solid rgba(255, 158, 209, 0.85);"
                title = "Emotional wording"
            elif word_counts.get(lowered, 0) >= 3 and len(lowered) > 3:
                css = "background: rgba(246, 207, 122, 0.28); border-bottom: 2px solid rgba(246, 207, 122, 0.85);"
                title = "Repeated wording"
            elif lowered in slang_terms:
                css = "background: rgba(139, 182, 255, 0.28); border-bottom: 2px solid rgba(139, 182, 255, 0.85);"
                title = "Informal wording"

            if css:
                parts.append(
                    f"<span title='{html.escape(title)}' style='padding:0 0.12rem;border-radius:0.3rem;{css}'>{escaped}</span>"
                )
            else:
                parts.append(escaped)

        return "".join(parts)

    def _compare_latent_profiles(self, before: Dict[str, Dict], after: Dict[str, Dict]) -> List[Dict]:
        rows = []
        for trait in TRAITS:
            diff = after[trait]["score"] - before[trait]["score"]
            rows.append(
                {
                    "dimension": trait,
                    "difference": float(diff),
                    "direction": "higher in second entry" if diff > 0.03 else "higher in first entry" if diff < -0.03 else "similar",
                }
            )
        return rows

    def _check_safety(self, text: str) -> List[str]:
        normalized = text.lower()
        matches = []
        for pattern in self.safety_patterns:
            if re.search(pattern, normalized):
                matches.append(pattern.replace(r"\b", ""))
        return matches

    def _check_quality(self, text: str) -> Dict:
        words = re.findall(r"\b[\w']+\b", text.lower())
        unique_ratio = len(set(words)) / len(words) if words else 0.0
        repetition_ratio = self._repetition_ratio(words, 3)
        patterns: List[str] = []
        suggestions: List[str] = []

        if unique_ratio < 0.28:
            patterns.append("The entry uses very limited vocabulary, so the output would likely feel random.")
            suggestions.append("Write a few fuller sentences about what happened, how you felt, and what you need.")
        if repetition_ratio > 0.40:
            patterns.append("The entry repeats itself heavily, which looks more like looping than reflective writing.")
            suggestions.append("Try describing the situation in complete sentences instead of repeating one line.")
        if len(set(words)) < 12:
            patterns.append("There is not enough semantic variety for a deep reading.")

        return {
            "usable": not (unique_ratio < 0.20 or repetition_ratio > 0.40),
            "summary": "This sample is too narrow or repetitive for a meaningful journaling analysis.",
            "patterns": patterns or ["The entry needs more detail before the app can say anything useful."],
            "suggestions": suggestions or ["Add more context and emotional detail, then re-analyze the entry."],
        }

    def _repetition_ratio(self, words: List[str], n: int) -> float:
        if len(words) < n:
            return 0.0
        ngrams = [" ".join(words[index:index + n]) for index in range(len(words) - n + 1)]
        return 1 - (len(set(ngrams)) / len(ngrams)) if ngrams else 0.0
