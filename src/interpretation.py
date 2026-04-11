"""
Interpretation layer for Cognisight.

The model remains unchanged, but this module turns its output plus linguistic
features into more useful journaling insights.
"""

from typing import Dict, List, Tuple


FEATURE_LABELS = {
    "ling_feat_1": "Lexical diversity",
    "ling_feat_2": "Sentence length",
    "ling_feat_4": "Overall tone",
    "ling_feat_5": "Emotional fluctuation",
    "ling_feat_6": "Emotional intensity",
    "ling_feat_7": "Emotional wording",
    "ling_feat_8": "Positive wording",
    "ling_feat_9": "Negative wording",
    "ling_feat_10": "Self-focus",
    "ling_feat_11": "Interpersonal focus",
    "ling_feat_13": "Questioning",
    "ling_feat_24": "Word variety",
    "ling_feat_25": "Repetition",
    "ling_feat_28": "Informality",
}


FEATURE_EXPLANATIONS = {
    "ling_feat_1": "Richer vocabulary usually reflects more reflective or expansive thinking.",
    "ling_feat_2": "Sentence length influences whether the writing feels structured or abrupt.",
    "ling_feat_4": "This tracks whether the entry leans positive, neutral, or heavy in tone.",
    "ling_feat_5": "High fluctuation suggests the emotional state shifts across the entry.",
    "ling_feat_6": "High intensity means feelings are strongly present in the writing.",
    "ling_feat_7": "Emotional wording adds visible affective charge.",
    "ling_feat_8": "Positive wording can indicate relief, hope, or warmth.",
    "ling_feat_9": "Negative wording can indicate stress, frustration, or heaviness.",
    "ling_feat_10": "High self-focus suggests the entry is inward-looking and personal.",
    "ling_feat_11": "Interpersonal focus suggests the writer is thinking about relationships and audience.",
    "ling_feat_13": "Frequent questions can signal reflection, uncertainty, or rumination.",
    "ling_feat_24": "More variety generally means less looping and more range in thought.",
    "ling_feat_25": "Lower values here often mean the text repeats itself more.",
    "ling_feat_28": "Higher informality suggests a more casual, spoken style.",
}


MBTI_DESCRIPTIONS = {
    "INTJ": "strategic, future-focused, and independent",
    "INTP": "analytical, curious, and idea-driven",
    "ENTJ": "decisive, structured, and goal-oriented",
    "ENTP": "inventive, exploratory, and mentally restless",
    "INFJ": "insightful, values-driven, and quietly organized",
    "INFP": "reflective, idealistic, and emotionally aware",
    "ENFJ": "encouraging, purposeful, and people-aware",
    "ENFP": "imaginative, expressive, and possibility-focused",
    "ISTJ": "reliable, steady, and detail-aware",
    "ISFJ": "supportive, grounded, and quietly caring",
    "ESTJ": "practical, direct, and systems-oriented",
    "ESFJ": "warm, attentive, and socially responsive",
    "ISTP": "calm, hands-on, and adaptive",
    "ISFP": "gentle, present-focused, and emotionally tuned in",
    "ESTP": "bold, energetic, and action-first",
    "ESFP": "expressive, sociable, and experience-seeking",
}


class FeatureImportanceExplainer:
    """Return readable signal descriptions."""

    def get_top_contributing_features(
        self, feature_scores: Dict[str, float], n: int = 5
    ) -> List[Tuple[str, float, str]]:
        ranked = sorted(
            feature_scores.items(),
            key=lambda item: abs(item[1] - 0.5),
            reverse=True,
        )[:n]
        return [
            (
                FEATURE_LABELS.get(name, name),
                value,
                FEATURE_EXPLANATIONS.get(name, "This language signal influenced the interpretation."),
            )
            for name, value in ranked
        ]


class PersonalityInterpreter:
    """Turn model scores and language features into journaling insights."""

    def classify_mental_state(self, profile: Dict[str, Dict], features: Dict[str, float]) -> Dict[str, str]:
        neuroticism = profile.get("Neuroticism", {}).get("score", 0.5)
        conscientiousness = profile.get("Conscientiousness", {}).get("score", 0.5)
        openness = profile.get("Openness", {}).get("score", 0.5)
        fluctuation = features.get("ling_feat_5", 0.0)
        intensity = features.get("ling_feat_6", 0.0)
        tone = features.get("ling_feat_4", 0.5)
        repetition = features.get("ling_feat_25", 1.0)

        if neuroticism >= 0.74 or (intensity >= 0.58 and fluctuation >= 0.18 and tone <= 0.45):
            return {
                "label": "Stressed",
                "summary": "This entry reads as emotionally loaded, with tension carrying a lot of the momentum.",
            }
        if repetition <= 0.58 and fluctuation >= 0.16:
            return {
                "label": "Overthinking",
                "summary": "The writing suggests looping thought patterns and difficulty fully settling on one clear thread.",
            }
        if conscientiousness >= 0.65 and features.get("ling_feat_2", 0.0) >= 0.70:
            return {
                "label": "Analytical",
                "summary": "The entry feels structured and deliberate, with noticeable effort to reason things through.",
            }
        if openness >= 0.64 and tone >= 0.48:
            return {
                "label": "Reflective",
                "summary": "The writing feels introspective, curious, and focused on understanding inner experience.",
            }
        if tone >= 0.58 and intensity <= 0.36 and fluctuation <= 0.12:
            return {
                "label": "Calm",
                "summary": "The emotional tone appears fairly steady and manageable across the entry.",
            }
        return {
            "label": "Mixed",
            "summary": "The entry blends reflection, emotion, and uncertainty without one single state dominating.",
        }

    def mbti_type_explanations(self, mbti_matches: List[Dict]) -> List[str]:
        explanations = []
        for match in mbti_matches:
            mbti_type = match["type"]
            probability = match["probability"]
            explanations.append(
                f"{mbti_type} ({probability:.0f}%) — {MBTI_DESCRIPTIONS.get(mbti_type, 'personality style match')}"
            )
        return explanations

    def emotional_analysis(self, profile: Dict[str, Dict], features: Dict[str, float]) -> Dict[str, object]:
        tone = features.get("ling_feat_4", 0.5)
        fluctuation = features.get("ling_feat_5", 0.0)
        intensity = features.get("ling_feat_6", 0.0)

        if intensity >= 0.55:
            intensity_label = "High"
        elif intensity >= 0.34:
            intensity_label = "Moderate"
        else:
            intensity_label = "Low"

        if fluctuation >= 0.18:
            stability_label = "Fluctuating"
        elif fluctuation >= 0.10:
            stability_label = "Somewhat variable"
        else:
            stability_label = "Steady"

        if tone <= 0.42:
            tone_label = "Negative"
        elif tone >= 0.58:
            tone_label = "Positive"
        else:
            tone_label = "Neutral"

        insights = [
            f"Emotional intensity is {intensity_label.lower()}, so feelings are {'strongly present' if intensity_label == 'High' else 'present but contained' if intensity_label == 'Moderate' else 'fairly contained'} in the writing.",
            f"The emotional flow appears {stability_label.lower()}.",
            f"The overall tone reads as {tone_label.lower()}.",
        ]

        if profile.get("Neuroticism", {}).get("score", 0) >= 0.68:
            insights.append("The model also sees stress-sensitive language patterns in this sample.")

        return {
            "intensity": intensity_label,
            "stability": stability_label,
            "tone": tone_label,
            "insights": insights,
        }

    def thought_patterns(self, features: Dict[str, float]) -> List[str]:
        patterns: List[str] = []

        if features.get("ling_feat_25", 1.0) <= 0.58:
            patterns.append("There are signs of repetition or rumination in the wording.")
        if features.get("ling_feat_13", 0.0) >= 0.14:
            patterns.append("Frequent questioning suggests uncertainty or active self-interrogation.")
        if features.get("ling_feat_2", 0.0) <= 0.42 and features.get("ling_feat_5", 0.0) >= 0.15:
            patterns.append("Short, unstable phrasing suggests mental clutter or confusion.")
        if features.get("ling_feat_1", 0.0) >= 0.68 or features.get("ling_feat_24", 0.0) >= 0.48:
            patterns.append("The language shows reflective range rather than one narrow track of thought.")
        if not patterns:
            patterns.append("The thought flow appears reasonably clear without strong looping or chaos.")

        return patterns[:4]

    def communication_style(self, profile: Dict[str, Dict], features: Dict[str, float], text: str) -> List[str]:
        style: List[str] = []
        informal_markers = features.get("ling_feat_28", 0.0)
        emotional_intensity = features.get("ling_feat_6", 0.0)
        extraversion = profile.get("Extraversion", {}).get("score", 0.5)
        slang_ratio = self._slang_ratio(text)

        if slang_ratio >= 0.03 or informal_markers >= 0.06:
            style.append("The tone is more informal and conversational than formal.")
        else:
            style.append("The tone leans more formal and deliberate than casual.")

        if emotional_intensity >= 0.50 or extraversion >= 0.62:
            style.append("The writing style feels expressive rather than reserved.")
        else:
            style.append("The writing style feels measured and somewhat reserved.")

        return style

    def mental_signals(self, profile: Dict[str, Dict], features: Dict[str, float]) -> List[str]:
        signals: List[str] = []

        if profile.get("Neuroticism", {}).get("score", 0) >= 0.68 or features.get("ling_feat_6", 0.0) >= 0.56:
            signals.append("The entry contains stress indicators and heightened emotional load.")
        if features.get("ling_feat_5", 0.0) >= 0.18:
            signals.append("Emotional overload may be making the tone less steady across the entry.")
        if features.get("ling_feat_2", 0.0) <= 0.42 and features.get("ling_feat_25", 1.0) <= 0.60:
            signals.append("There are signs of cognitive fatigue or cluttered thinking.")
        if not signals:
            signals.append("No strong mental strain signal dominates this entry.")

        return signals[:4]

    def strengths(self, profile: Dict[str, Dict], features: Dict[str, float]) -> List[str]:
        strengths: List[str] = []

        if profile.get("Openness", {}).get("score", 0) >= 0.62:
            strengths.append("**Reflective curiosity:** You're not just reporting what happened—you're asking yourself *why*, which is the beginning of real understanding.")
        if features.get("ling_feat_10", 0.0) >= 0.08:
            strengths.append("**Self-awareness:** The entry reveals you're tuned in to your own patterns, feelings, and reactions rather than avoiding the harder parts.")
        if profile.get("Conscientiousness", {}).get("score", 0) >= 0.60 or features.get("ling_feat_2", 0.0) >= 0.70:
            strengths.append("**Structured thinking:** You organize ideas clearly, which means progress from confusion to clarity is a natural next step.")
        if not strengths:
            strengths.append("**Honesty:** Putting your genuine experience into words—without filters—is itself a valuable reflective practice.")

        return strengths[:3]

    def suggestions(self, profile: Dict[str, Dict], features: Dict[str, float]) -> List[str]:
        suggestions: List[str] = []

        if features.get("ling_feat_25", 1.0) <= 0.58:
            suggestions.append("**Break looping thoughts:** Separate 'what happened' from 'what I felt' from 'what I need.' Writing them as distinct thoughts first can help you finish each one.")
        if profile.get("Neuroticism", {}).get("score", 0) >= 0.68 or features.get("ling_feat_6", 0.0) >= 0.56:
            suggestions.append("**Slow your pace:** Write one sentence at a time. If energy drops, stop, breathe, and return later. Quality over volume.")
        if features.get("ling_feat_2", 0.0) <= 0.42:
            suggestions.append("**Create structure first:** Use bullets for the bare facts, then expand into fuller sentences. This helps shift from scattered to organized.")
        if features.get("ling_feat_13", 0.0) >= 0.14:
            suggestions.append("**Pick one thread:** Choose the most pressing thought, write about it fully, then move to the next. Circular thinking often clears when focused deeper.")
        if not suggestions:
            suggestions.append("**Keep this momentum:** Your entry shows genuine reflection and clarity. The journaling approach you're using is already working well.")

        return suggestions[:4]

    def confidence_context(self) -> str:
        return (
            "These insights come from language patterns in a single entry. "
            "They are meant for self-reflection, not diagnosis."
        )

    def _slang_ratio(self, text: str) -> float:
        slang_terms = {"idk", "lol", "gonna", "wanna", "kinda", "sorta", "bro", "omg"}
        words = [word.lower() for word in text.split()]
        if not words:
            return 0.0
        hits = sum(1 for word in words if word.strip(".,!?;:") in slang_terms)
        return hits / len(words)


class BehavioralInsightGenerator:
    """High-level summary helpers."""

    def reflection_summary(self, mental_state: Dict[str, str], patterns: List[str]) -> str:
        if patterns:
            return f"{mental_state['summary']} A noticeable pattern is that {patterns[0].lower()}"
        return mental_state["summary"]

    def comparison_summary(self, before_state: Dict[str, str], after_state: Dict[str, str]) -> str:
        if before_state["label"] == after_state["label"]:
            return f"Both entries read as {after_state['label'].lower()}, but the underlying language signals may still differ."
        return f"The newer entry shifts from {before_state['label'].lower()} toward {after_state['label'].lower()}."


class ConfidenceCalibrator:
    """Internal helper for stable confidence-like values."""

    def normalize_confidence(self, score: float) -> float:
        normalized = 0.60 + (max(0.0, min(1.0, score)) * 0.30)
        return min(0.90, max(0.60, normalized))


class ReflectionPromptGenerator:
    """Generate follow-up journaling prompts based on analysis."""

    def generate_prompts(self, profile: Dict[str, Dict], mental_state: str, thought_patterns: List[str], text_length: int) -> List[str]:
        """Generate 2-3 personalized follow-up prompts."""
        prompts = []

        # Prompt 1: Based on mental state
        if mental_state == "Overthinking":
            prompts.append("Of all the thoughts spinning right now, which one matters most? Write about just that one.")
        elif mental_state == "Stressed":
            prompts.append("What's one small thing you could change or control about this situation right now?")
        elif mental_state == "Reflective":
            prompts.append("What does this insight mean for how you'll act or think going forward?")
        elif mental_state == "Analytical":
            prompts.append("You've thought this through logically. What does your gut tell you to do?")
        else:
            prompts.append("What would help the most right now—action, rest, or just being heard?")

        # Prompt 2: Based on openness/extraversion
        if profile.get("Extraversion", {}).get("score", 0.5) >= 0.60:
            prompts.append("Who could you talk to about this? What do you want them to understand most?")
        else:
            prompts.append("How has this pattern shown up in your life before? Does today feel different?")

        # Prompt 3: Based on text length / depth
        if text_length < 150:
            prompts.append("Go back and pick the sentence that carries the most emotion. Expand it—what details matter there?")
        else:
            prompts.append("Reading back over this, what surprises you most about how you're thinking right now?")

        return prompts[:3]
