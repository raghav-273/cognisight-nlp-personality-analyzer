"""
Feature importance extraction and visualization helpers.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    xgb = None


# Feature name mapping (for display)
FEATURE_NAMES = {
    # Linguistic features (0-29)
    **{f"feat_{i}": f"Linguistic Feature {i}" for i in range(30)},
    # TF-IDF features (30-129)
    **{f"feat_{i}": f"Vocabulary Pattern {i-30}" for i in range(30, 130)},
}

# More readable feature names for the first 30 linguistic features
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


class FeatureImportanceExtractor:
    """Extract and interpret feature importances from models."""
    
    @staticmethod
    def extract_xgboost_importance(
        model: Any,
        feature_names: List[str] = None,
        importance_type: str = 'gain'
    ) -> Dict[str, float]:
        """
        Extract feature importances from XGBoost model.
        
        Args:
            model: Trained XGBoost regressor
            feature_names: Optional list of feature names
            importance_type: Type of importance ('weight', 'gain', 'cover')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if xgb is None or not hasattr(model, 'get_booster'):
            return {}
        
        booster = model.get_booster()
        importances = booster.get_score(importance_type=importance_type)
        
        # Normalize importances
        if importances:
            total = sum(importances.values())
            importances = {k: v / total for k, v in importances.items()}
        
        return importances
    
    @staticmethod
    def extract_random_forest_importance(
        model: RandomForestClassifier,
        feature_names: List[str] = None
    ) -> Dict[str, float]:
        """
        Extract feature importances from Random Forest.
        
        Args:
            model: Trained Random Forest classifier
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            return {}
        
        importances = model.feature_importances_
        
        # Map to feature names
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
        """
        Get top N most important features.
        
        Args:
            importances: Feature importance dictionary
            n: Number of top features
            readable_names: Whether to use human-readable names
            
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        sorted_features = sorted(
            importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:n]
        
        if readable_names:
            result = []
            for feat_name, importance in sorted_features:
                readable_name = LINGUISTIC_FEATURE_NAMES.get(
                    feat_name,
                    FEATURE_NAMES.get(feat_name, feat_name)
                )
                result.append((readable_name, importance))
            return result
        
        return sorted_features
    
    @staticmethod
    def get_feature_contribution_by_trait(
        feature_importances: Dict[str, float],
        trait_index: int
    ) -> Dict[str, float]:
        """
        Get feature contributions specific to a trait prediction.
        
        In practice, this uses the overall feature importances,
        but could be extended to per-trait importances from models
        that support it (e.g., SHAP values).
        
        Args:
            feature_importances: Overall feature importance dictionary
            trait_index: Index of the trait (0-4 for Big Five)
            
        Returns:
            Dictionary of feature contributions
        """
        # Simple approach: scale importances by trait relevance
        # This could be extended with per-trait feature importance
        return feature_importances
    
    @staticmethod
    def explain_feature_contribution(
        feature_name: str,
        importance_score: float,
        selected_trait: str = None
    ) -> str:
        """
        Generate explanation for how a feature contributes to prediction.
        
        Args:
            feature_name: Name of the feature
            importance_score: Importance score (0-1)
            selected_trait: Trait being explained (optional)
            
        Returns:
            Natural language explanation
        """
        # Get readable feature name
        readable_name = LINGUISTIC_FEATURE_NAMES.get(
            feature_name,
            feature_name
        )
        
        # Interpret importance level
        if importance_score > 0.15:
            impact = "strong influence"
        elif importance_score > 0.08:
            impact = "moderate influence"
        elif importance_score > 0.04:
            impact = "notable influence"
        else:
            impact = "minor influence"
        
        explanation = f"**{readable_name}** has {impact} on the prediction"
        
        if selected_trait:
            explanation += f" for {selected_trait}"
        
        return explanation


class FeatureMetricsCalculator:
    """Calculate descriptive metrics about extracted features."""
    
    @staticmethod
    def get_feature_type_distribution(
        importances: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate distribution of importance across feature types.
        
        Args:
            importances: Feature importance dictionary
            
        Returns:
            Dictionary with type distribution
        """
        linguistic_importance = 0
        tfidf_importance = 0
        
        for feat_name, importance in importances.items():
            try:
                feat_idx = int(feat_name.split('_')[1])
                if feat_idx < 30:
                    linguistic_importance += importance
                else:
                    tfidf_importance += importance
            except (ValueError, IndexError):
                pass
        
        total = linguistic_importance + tfidf_importance
        if total == 0:
            return {"linguistic": 0.5, "tfidf": 0.5}
        
        return {
            "linguistic": linguistic_importance / total,
            "tfidf": tfidf_importance / total
        }
    
    @staticmethod
    def get_concentration_metric(
        importances: Dict[str, float]
    ) -> float:
        """
        Calculate how concentrated feature importance is.
        
        Returns:
            Concentration score (0-1, where 1 = very concentrated)
        """
        if not importances:
            return 0
        
        values = list(importances.values())
        # Calculate entropy-like metric
        normalized = [abs(v) / sum(abs(v) for v in values) if values else 0 for v in values]
        
        # Calculate concentration (higher = more concentrated)
        concentration = sum(v ** 2 for v in normalized)
        
        return min(1.0, concentration)
    
    @staticmethod
    def get_linguistic_vs_content_analysis(
        importances: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Provide analysis of linguistic vs content-based features.
        
        Args:
            importances: Feature importance dictionary
            
        Returns:
            Analysis and interpretation
        """
        distribution = FeatureMetricsCalculator.get_feature_type_distribution(importances)
        linguistic_ratio = distribution['linguistic']
        
        if linguistic_ratio > 0.7:
            analysis = (
                "Your prediction is driven primarily by **writing style patterns** "
                "(sentence structure, emotional expression, etc.) rather than specific vocabulary."
            )
        elif linguistic_ratio < 0.3:
            analysis = (
                "Your prediction is driven primarily by **specific vocabulary choices** "
                "more than overall writing style characteristics."
            )
        else:
            analysis = (
                "Your prediction is driven by a **balanced mix** of writing style patterns "
                "and specific vocabulary choices."
            )
        
        return {
            "analysis": analysis,
            "linguistic_ratio": linguistic_ratio,
            "content_ratio": distribution['tfidf']
        }
