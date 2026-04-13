"""
Feature importance extraction and visualization helpers.
"""

from typing import Any, Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
except ModuleNotFoundError:
    xgb = None


# ---------------- FEATURE NAME MAPPINGS ---------------- #

FEATURE_NAMES = {
    **{f"feat_{i}": f"Linguistic Feature {i}" for i in range(30)},
    **{f"feat_{i}": f"Vocabulary Pattern {i-30}" for i in range(30, 130)},
}

LINGUISTIC_FEATURE_NAMES = {
    "feat_0": "Average Word Length",
    "feat_1": "Lexical Diversity",
    "feat_2": "Average Sentence Length",
    "feat_3": "Sentence Length Variance",
    "feat_4": "Overall Sentiment",
    "feat_5": "Sentiment Variance",
    "feat_6": "Sentiment Intensity",
    "feat_7": "Emotional Word Ratio",
    "feat_8": "Positive Word Ratio",
    "feat_9": "Negative Word Ratio",
    "feat_10": "First-Person Focus",
    "feat_11": "Second-Person Focus",
    "feat_12": "Third-Person Focus",
    "feat_13": "Question Frequency",
    "feat_14": "Exclamation Frequency",
    "feat_15": "Punctuation Diversity",
    "feat_16": "Uppercase Ratio",
    "feat_17": "Digit Ratio",
    "feat_18": "Stopword Ratio",
    "feat_19": "Readability",
    "feat_20": "Pronoun Variety",
    "feat_21": "Complex Word Ratio",
    "feat_22": "Coordinating Conjunction Usage",
    "feat_23": "Subordinating Conjunction Usage",
    "feat_24": "Word Frequency Entropy",
    "feat_25": "Vocabulary Repetition",
    "feat_26": "Short Word Ratio",
    "feat_27": "Long Word Ratio",
    "feat_28": "Contraction Ratio",
    "feat_29": "Tense Diversity",
}


# ---------------- FEATURE IMPORTANCE ---------------- #

class FeatureImportanceExtractor:
    """Extract and interpret feature importances from models."""

    @staticmethod
    def extract_xgboost_importance(
        model: Any,
        importance_type: str = "gain"
    ) -> Dict[str, float]:

        if xgb is None or not hasattr(model, "get_booster"):
            return {}

        booster = model.get_booster()
        raw_importances = booster.get_score(importance_type=importance_type)

        if not raw_importances:
            return {}

        total = sum(raw_importances.values())
        if total == 0:
            return {}

        return {
            f"feat_{int(k[1:])}": v / total
            for k, v in raw_importances.items()
        }

    @staticmethod
    def extract_random_forest_importance(
        model: Any,
        feature_names: List[str] = None
    ) -> Dict[str, float]:

        if not hasattr(model, "feature_importances_"):
            return {}

        importances = model.feature_importances_

        result = {}
        for i, importance in enumerate(importances):
            if feature_names and i < len(feature_names):
                feat_name = feature_names[i]
            else:
                feat_name = f"feat_{i}"

            result[feat_name] = float(importance)

        return result

    @staticmethod
    def get_top_features(
        importances: Dict[str, float],
        n: int = 10,
        readable_names: bool = True
    ) -> List[Tuple[str, float]]:

        sorted_features = sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

        if not readable_names:
            return sorted_features

        result = []
        for feat_name, importance in sorted_features:
            readable_name = LINGUISTIC_FEATURE_NAMES.get(
                feat_name,
                FEATURE_NAMES.get(feat_name, feat_name)
            )
            result.append((readable_name, importance))

        return result

    @staticmethod
    def explain_feature_contribution(
        feature_name: str,
        importance_score: float,
        selected_trait: str = None
    ) -> str:

        readable_name = LINGUISTIC_FEATURE_NAMES.get(feature_name, feature_name)

        if importance_score > 0.15:
            impact = "a dominant driver"
        elif importance_score > 0.08:
            impact = "a meaningful contributor"
        elif importance_score > 0.04:
            impact = "a supporting factor"
        else:
            impact = "a minor signal"

        explanation = f"{readable_name} strongly influences how the model interprets your writing"

        if selected_trait:
            explanation += f", especially for {selected_trait}"

        return explanation


# ---------------- FEATURE METRICS ---------------- #

class FeatureMetricsCalculator:
    """Calculate descriptive metrics about extracted features."""

    @staticmethod
    def get_feature_type_distribution(importances: Dict[str, float]) -> Dict[str, float]:

        linguistic_importance = 0
        tfidf_importance = 0

        for feat_name, importance in importances.items():
            try:
                idx = int(feat_name.split("_")[1])
                if idx < 30:
                    linguistic_importance += importance
                else:
                    tfidf_importance += importance
            except (ValueError, IndexError):
                continue

        total = linguistic_importance + tfidf_importance

        if total == 0:
            return {"linguistic": 0.5, "tfidf": 0.5}

        return {
            "linguistic": linguistic_importance / total,
            "tfidf": tfidf_importance / total
        }

    @staticmethod
    def get_concentration_metric(importances: Dict[str, float]) -> Dict[str, Any]:

        if not importances:
            return {"score": 0, "interpretation": "No feature importance available"}

        values = list(importances.values())
        total = sum(abs(v) for v in values)

        if total == 0:
            return {"score": 0, "interpretation": "No meaningful contribution detected"}

        normalized = [abs(v) / total for v in values]
        concentration = sum(v ** 2 for v in normalized)

        interpretation = (
            "Prediction depends heavily on a few dominant features"
            if concentration > 0.4
            else "Prediction is influenced by a broad mix of features"
        )

        return {
            "score": min(1.0, concentration),
            "interpretation": interpretation
        }

    @staticmethod
    def get_linguistic_vs_content_analysis(importances: Dict[str, float]) -> Dict[str, Any]:

        distribution = FeatureMetricsCalculator.get_feature_type_distribution(importances)
        linguistic_ratio = distribution["linguistic"]

        if linguistic_ratio > 0.7:
            analysis = (
                "The prediction is driven mostly by how you write — your tone, structure, "
                "and emotional expression — rather than specific word choices."
            )
        elif linguistic_ratio < 0.3:
            analysis = (
                "The prediction is influenced more by the specific words and topics you use "
                "than by your overall writing style."
            )
        else:
            analysis = (
                "Both writing style and word choices contribute meaningfully to this prediction."
            )

        return {
            "analysis": analysis,
            "linguistic_ratio": linguistic_ratio,
            "content_ratio": distribution["tfidf"]
        }