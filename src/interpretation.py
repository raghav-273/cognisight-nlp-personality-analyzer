"""
Interpretation layer for Cognisight.

The model remains unchanged, but this module turns its output plus linguistic
features into more useful journaling insights.
"""
from typing import Dict, List, Tuple
import re

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
FEATURE_LABELS.update({
    "emotion_sentiment": "Overall sentiment",
    "emotion_intensity": "Emotional strength",
    "emotion_diversity": "Emotional variety",
    "emotion_polarity": "Positive vs negative balance",
    "emotion_stability": "Emotional stability",
    "emotion_dominant": "Dominant emotion"
})

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
FEATURE_EXPLANATIONS.update({
    "emotion_sentiment": "Captures whether the writing leans positive or negative overall.",
    "emotion_intensity": "Indicates how strongly emotions are expressed.",
    "emotion_diversity": "Higher values suggest multiple emotions are present.",
    "emotion_polarity": "Balance between positive and negative emotional tone.",
    "emotion_stability": "Lower values suggest steadier emotional expression.",
    "emotion_dominant": "The most prominent emotion detected in the text."
})


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
        dominant = features.get("emotion_dominant", None)
        polarity = features.get("emotion_polarity", 0.0)

        if (neuroticism >= 0.74 
            or (intensity >= 0.58 and fluctuation >= 0.18 and tone <= 0.45)
            or (dominant is not None and dominant in [2, 5] and polarity < -0.2)
        ):
            return {
                "label": "Stressed",
                "summary": "The entry feels emotionally heavy, with tension driving much of the thinking rather than clarity.",
            }
        if repetition <= 0.58 and fluctuation >= 0.16  and intensity >= 0.3:
            return {
                "label": "Overthinking",
                "summary": "The writing shows signs of looping thoughts, where ideas repeat without fully resolving.",
            }
        if conscientiousness >= 0.65 and features.get("ling_feat_2", 0.0) >= 0.70:
            return {
                "label": "Analytical",
                "summary": "The entry feels structured and deliberate, with clear effort to reason through the situation step by step.",
            }
        if openness >= 0.64 and tone >= 0.48:
            return {
                "label": "Reflective",
                "summary": "The writing is introspective and meaning-focused, aiming to understand rather than just describe.",
            }
        if (tone >= 0.58 
            and intensity <= 0.36 
            and fluctuation <= 0.12 
            and ((dominant is not None and dominant == 3) or polarity > 0.2)
        ):
            return {
                "label": "Calm",
                "summary": "The tone appears steady and controlled, with emotions present but not overwhelming the thinking.",
            }
        return {
            "label": "Mixed",
            "summary": "The entry blends reflection, emotion, and uncertainty without a single clear state dominating.",
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
        dominant = features.get("emotion_dominant", None)
        polarity = features.get("emotion_polarity", 0.0)

        emotion_map = {
            0: "anger",
            1: "disgust",
            2: "fear",
            3: "joy",
            4: "neutral",
            5: "sadness",
            6: "surprise"
        }

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

        insights = []

        # Intensity insight
        if intensity_label == "High":
            insights.append("Emotions are strongly present and actively shaping how the situation is being processed.")
        elif intensity_label == "Moderate":
            insights.append("Emotions are present but still somewhat contained within the writing.")
        else:
            insights.append("Emotions appear relatively controlled and not dominating the thought process.")

        # Stability insight
        if stability_label == "Fluctuating":
            insights.append("The emotional tone shifts noticeably across the entry, suggesting an unsettled internal state.")
        elif stability_label == "Somewhat variable":
            insights.append("There is some variation in emotional tone, but not enough to feel chaotic.")
        else:
            insights.append("The emotional tone remains steady throughout, without sharp swings.")

        # Tone insight
        if tone_label == "Negative":
            insights.append("The overall emotional tone carries a heavier or more strained quality.")
        elif tone_label == "Positive":
            insights.append("The tone reflects a more constructive or hopeful emotional state.")
        else:
            insights.append("The tone sits in a neutral range without a strong positive or negative pull.")

        # NEW: Dominant emotion insight
        if dominant is not None:
            emotion_name = emotion_map.get(int(dominant), "mixed emotion") if isinstance(dominant, (int, float)) else "mixed emotion"
            insights.append(f"The dominant emotional signal appears to be {emotion_name}, which strongly influences how the situation is being experienced.")

        # NEW: Polarity insight
        if polarity > 0.2:
            insights.append("There is a noticeable positive emotional tilt across the entry.")
        elif polarity < -0.2:
            insights.append("There is a noticeable negative emotional weight across the entry.")

        # Existing neuroticism signal
        if profile.get("Neuroticism", {}).get("score", 0) >= 0.68:
            insights.append("There are also signs of stress-sensitive language patterns in the writing.")

        if intensity_label == "High" and stability_label == "Fluctuating":
            insights.append("Strong and shifting emotions together suggest the experience may feel overwhelming or hard to settle.")
        return {
            "intensity": intensity_label,
            "stability": stability_label,
            "tone": tone_label,
            "insights": insights,
        }

    def thought_patterns(self, features: Dict[str, float]) -> List[str]:
        dominant = features.get("emotion_dominant", None)
        intensity = features.get("ling_feat_6", 0.0)
        patterns: List[str] = []

        if dominant is not None:
            if dominant == 2:  # fear
                patterns.append("The thinking pattern suggests underlying anxiety, with attention drawn toward possible outcomes or uncertainties.")
            elif dominant == 5:  # sadness
                patterns.append("The writing reflects a heavier thought pattern, where ideas may feel slowed down or weighed by emotion.")

        if features.get("ling_feat_25", 1.0) <= 0.58:
            if intensity >= 0.4:
                patterns.append("There are signs of emotionally charged rumination, where thoughts repeat and remain difficult to resolve.")
            else:
                patterns.append("There are signs of repetition, suggesting the thinking may be circling around the same ideas.")
        if features.get("ling_feat_13", 0.0) >= 0.14:
            patterns.append("Frequent questioning suggests uncertainty or active self-interrogation.")
        if features.get("ling_feat_2", 0.0) <= 0.42 and features.get("ling_feat_5", 0.0) >= 0.15:
            patterns.append("Short and unstable phrasing suggests difficulty organizing thoughts, which can feel like mental clutter.")
        if features.get("ling_feat_1", 0.0) >= 0.68 or features.get("ling_feat_24", 0.0) >= 0.48:
            patterns.append("The language shows a broad and reflective range, suggesting the thinking is not stuck on a single narrow loop.")
        if not patterns:
            patterns.append("The thought flow appears reasonably clear without strong looping or chaos.")
        
        return patterns[:4]

    def communication_style(self, profile: Dict[str, Dict], features: Dict[str, float], text: str) -> List[str]:
        style: List[str] = []
        dominant = features.get("emotion_dominant", None)
        informal_markers = features.get("ling_feat_28", 0.0)
        emotional_intensity = features.get("ling_feat_6", 0.0)
        extraversion = profile.get("Extraversion", {}).get("score", 0.5)
        slang_ratio = self._slang_ratio(text)

        if slang_ratio >= 0.03 and emotional_intensity >= 0.5:
            style.append("The tone combines informality with strong emotional expression, making it feel spontaneous and unfiltered.")
        elif slang_ratio >= 0.03 or informal_markers >= 0.06:
            style.append("The tone is informal and conversational, resembling natural speech rather than structured writing.")
        else:
            style.append("The tone leans more formal and deliberate, with controlled and structured phrasing.")

        if emotional_intensity >= 0.50 or extraversion >= 0.62:
            if dominant is not None and dominant == 2:  # fear
                style.append("The writing is expressive, with emotional energy leaning toward tension or concern.")
            elif dominant == 5:  # sadness
                style.append("The writing is expressive, carrying a softer or heavier emotional tone.")
            elif dominant == 3:  # joy
                style.append("The writing is expressive, with a lighter and more positive emotional tone.")
            else:
                style.append("The writing style feels expressive rather than reserved.")
        else:
            style.append("The writing style feels controlled and relatively reserved, with limited outward emotional expression.")

        return style

    def mental_signals(self, profile: Dict[str, Dict], features: Dict[str, float]) -> List[str]:
        signals: List[str] = []

        neuroticism = profile.get("Neuroticism", {}).get("score", 0.0)
        intensity = features.get("ling_feat_6", 0.0)
        fluctuation = features.get("ling_feat_5", 0.0)
        structure = features.get("ling_feat_2", 0.0)
        repetition = features.get("ling_feat_25", 1.0)

        dominant = features.get("emotion_dominant", None)
        polarity = features.get("emotion_polarity", 0.0)

        # Stress / emotional load
        if neuroticism >= 0.68 or intensity >= 0.56:
            if dominant == 2:  # fear
                signals.append("There are signs of anxiety-driven stress, where uncertainty or future concerns are taking up mental space.")
            elif dominant == 5:  # sadness
                signals.append("The entry reflects emotional heaviness, which may be slowing down thinking and making things feel more effortful.")
            else:
                signals.append("The entry contains signs of heightened emotional load, where feelings are strongly influencing the thought process.")

        # Emotional instability
        if fluctuation >= 0.18:
            signals.append("The emotional tone shifts across the entry, suggesting difficulty maintaining a steady internal state.")

        # Cognitive fatigue / clutter
        if structure <= 0.42 and repetition <= 0.60:
            signals.append("There are signs of mental clutter or fatigue, where organizing thoughts clearly may feel harder than usual.")

        # Polarity-based signal
        if polarity < -0.25 and intensity < 0.55:
            signals.append("There is a consistent negative emotional weight, which may make the experience feel more draining over time.")
        elif polarity > 0.25 and intensity < 0.4:
            signals.append("The emotional tone is positive but relatively stable, without signs of strong internal strain.")

        # Fallback
        if not signals:
            signals.append("The entry does not show strong signs of mental strain, and the overall state appears relatively stable.")

        return signals[:4]

    def strengths(self, profile: Dict[str, Dict], features: Dict[str, float]) -> List[str]:
        strengths: List[str] = []

        openness = profile.get("Openness", {}).get("score", 0.0)
        conscientiousness = profile.get("Conscientiousness", {}).get("score", 0.0)
        self_focus = features.get("ling_feat_10", 0.0)
        structure = features.get("ling_feat_2", 0.0)
        intensity = features.get("ling_feat_6", 0.0)
        dominant = features.get("emotion_dominant", None)

        # Reflective depth
        if openness >= 0.62:
            strengths.append("**Reflective curiosity:** You're not just describing what happened—you’re trying to understand *why*, which is the foundation of real self-awareness.")

        # Self-awareness
        if self_focus >= 0.08:
            strengths.append("**Self-awareness:** You’re paying attention to your internal state instead of avoiding it, which is a strong indicator of emotional insight.")

        # Structure
        if conscientiousness >= 0.60 or structure >= 0.70:
            strengths.append("**Structured thinking:** You organize your thoughts clearly, which makes it easier to move from confusion toward clarity.")

        # Emotional honesty (NEW)
        if intensity >= 0.45:
            strengths.append("**Emotional honesty:** You’re allowing real feelings to show up in your writing instead of suppressing them, which is essential for meaningful reflection.")

        # Emotion-specific strength (NEW)
        if dominant == 2:  # fear
            strengths.append("**Awareness of uncertainty:** You’re able to notice and articulate anxious thoughts instead of ignoring them.")
        elif dominant == 5:  # sadness
            strengths.append("**Emotional depth:** You’re engaging with heavier emotions instead of avoiding them, which builds deeper self-understanding.")
        elif dominant == 3:  # joy
            strengths.append("**Positive awareness:** You’re able to recognize and express constructive or uplifting emotional states.")

        # Fallback
        if not strengths:
            strengths.append("**Honesty:** Putting your genuine experience into words—without filtering it—is already a strong step toward self-understanding.")

        return strengths[:3]

    def suggestions(self, profile: Dict[str, Dict], features: Dict[str, float]) -> List[str]:
        suggestions: List[str] = []

        neuroticism = profile.get("Neuroticism", {}).get("score", 0.0)
        repetition = features.get("ling_feat_25", 1.0)
        structure = features.get("ling_feat_2", 0.0)
        questioning = features.get("ling_feat_13", 0.0)
        intensity = features.get("ling_feat_6", 0.0)

        dominant = features.get("emotion_dominant", None)
        polarity = features.get("emotion_polarity", 0.0)

        # Looping thoughts
        if repetition <= 0.58:
            suggestions.append(
                "**Break the loop:** Try separating your thoughts into three parts — what happened, what you felt, and what you need next. Completing each one fully can reduce mental repetition."
            )

        # High emotional load
        if neuroticism >= 0.68 or intensity >= 0.56:
            if dominant == 2:  # fear
                suggestions.append(
                    "**Reduce uncertainty:** Write down the worst-case, best-case, and most likely outcome. This helps your mind move from vague anxiety to something more contained."
                )
            elif dominant == 5:  # sadness
                suggestions.append(
                    "**Lower the weight:** Instead of solving everything, focus on one small action you can take. Progress becomes easier when the emotional load is reduced."
                )
            else:
                suggestions.append(
                    "**Slow the pace:** Write one clear sentence at a time. When thoughts feel intense, slowing down helps clarity return."
                )

        # Lack of structure
        if structure <= 0.42:
            suggestions.append(
                "**Add structure first:** Start with bullet points for key ideas, then expand. Organizing thoughts externally often reduces internal confusion."
            )

        # Too many questions
        if questioning >= 0.14:
            suggestions.append(
                "**Focus your thinking:** Pick the most important question and answer it fully before moving on. Depth usually resolves more than breadth."
            )

        # Negative emotional weight
        if polarity < -0.25 and intensity < 0.55:
            suggestions.append(
                "**Balance the perspective:** Try adding one neutral or slightly positive observation. This doesn’t ignore the problem—it stabilizes how it’s processed."
            )

        # Positive stable state
        if polarity > 0.25 and intensity < 0.4:
            suggestions.append(
                "**Reinforce this state:** Identify what’s working right now and why. Writing it down helps make these patterns repeatable."
            )

        # Fallback
        if not suggestions:
            suggestions.append(
                "**Keep going:** This entry shows clarity and reflection. Continuing this style of writing will naturally strengthen your self-awareness."
            )

        return suggestions[:4]

    def confidence_context(self) -> str:
        return (
            "These insights are derived from patterns in your writing—such as tone, structure, and emotional signals—within this single entry. "
            "They capture how this moment is expressed, not a fixed or complete picture of your personality. "
            "Treat them as reflective guidance rather than a definitive conclusion."
        )

    def _slang_ratio(self, text: str) -> float:
        """
        Estimate how informal / conversational the text is.

        Combines:
        slang terms,contractions,short informal tokens,filler words

        Returns:
            float: ratio (0 to ~1), higher = more informal
        """

        # Core slang / internet language
        slang_terms = {
            "idk","idts", "lol", "lmao", "omg", "bro", "dude",
            "gonna", "wanna", "kinda", "sorta",
            "nah", "yeah", "yep", "nope",
            "btw", "tbh", "imo",
            "pls", "tho", "ok", "okay"
        }

        # Common informal contractions
        contractions = {
            "n't", "'re", "'ve", "'ll", "'d", "'s"
        }

        # Very short informal tokens (chat style)
        short_forms = {"u", "ur", "ya", "tho"}

        # Filler / casual speech markers
        fillers = {"like", "just", "actually", "basically", "literally"}

        # Tokenize properly
        words = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())

        if not words:
            return 0.0

        slang_hits = 0

        for word in words:
            # Direct slang match
            if word in slang_terms:
                slang_hits += 1

            # Contraction detection
            elif any(c in word for c in contractions):
                slang_hits += 0.5  # softer signal

            # Short informal tokens
            elif word in short_forms:
                slang_hits += 1

            # Fillers (weak signal)
            elif word in fillers:
                slang_hits += 0.5

        # Normalize
        return slang_hits / len(words)

class BehavioralInsightGenerator:
    """Generate high-level, human-readable summaries from analysis outputs."""

    def reflection_summary(self, mental_state: Dict[str, str], patterns: List[str]) -> str:
        """
        Blend mental state and dominant thought pattern into a smooth reflection.
        """
        base = mental_state.get("summary", "").strip()

        if not base:
            return ""

        if patterns:
            pattern = patterns[0].strip().lower()

            # Clean punctuation
            if base.endswith("."):
                base = base[:-1]

            return (
                f"{base}, which is reflected in how {pattern}"
            )

        return base

    def comparison_summary(self, before_state: Dict[str, str], after_state: Dict[str, str]) -> str:
        """
        Compare two entries and describe the shift with contextual meaning.
        """
        before_label = before_state.get("label", "previous state")
        after_label = after_state.get("label", "current state")

        before = before_label.lower()
        after = after_label.lower()

        # Same state
        if before == after:
            return (
                f"Both entries reflect a {after} state, "
                f"though the underlying tone or clarity may still vary slightly."
            )

        # Improvement transitions
        improving_from = {"stressed", "overthinking"}
        improving_to = {"calm", "reflective", "analytical"}

        if before in improving_from and after in improving_to:
            return (
                f"There is a clear shift from a more {before} state toward a more stable and structured {after} state, "
                f"suggesting improved clarity and emotional regulation."
            )

        # Deterioration transitions
        worsening_from = {"calm", "reflective", "analytical"}
        worsening_to = {"stressed", "overthinking"}

        if before in worsening_from and after in worsening_to:
            return (
                f"The newer entry moves from a relatively {before} state toward a more strained {after} state, "
                f"indicating increased cognitive or emotional pressure."
            )

        # Neutral shift
        return (
            f"There is a noticeable shift from a {before} state toward a more {after} state, "
            f"suggesting a change in how the situation is being processed."
        )
    
class ConfidenceCalibrator:
    """Calibrate raw model scores into stable, human-friendly confidence values."""

    def normalize_confidence(self, score: float) -> float:
        """
        Convert raw score (0-1) into calibrated confidence (0.60-0.90 range)
        with smoothing and uncertainty handling.
        """
        # Clamp input
        score = max(0.0, min(1.0, score))

        # Smooth curve (reduces harsh jumps near extremes)
        # sigmoid-like without heavy math
        smoothed = score ** 0.85

        # Map to desired range
        normalized = 0.60 + (smoothed * 0.30)

        # Extra damping near uncertainty zone (around 0.5)
        if 0.4 <= score <= 0.6:
            normalized -= 0.02  # slight penalty for ambiguity

        # Final clamp
        return round(min(0.90, max(0.60, normalized)), 3)

class ReflectionPromptGenerator:
    """Generate adaptive, psychologically-aware journaling prompts."""

    def generate_prompts(
        self,
        profile: Dict[str, Dict],
        mental_state: str,
        thought_patterns: List[str],
        text_length: int
    ) -> List[str]:

        prompts = []
        used = set()  # avoid repetition

        def add(prompt: str):
            if prompt not in used:
                prompts.append(prompt)
                used.add(prompt)

        extraversion = profile.get("Extraversion", {}).get("score", 0.5)
        openness = profile.get("Openness", {}).get("score", 0.5)

        # --- 1. Mental state driven prompt (core) ---
        if mental_state == "Overthinking":
            add("If you had to reduce everything you're thinking about to one core issue, what would it be?")
            add("What part of this situation actually requires action, and what part is just looping in your head?")
        elif mental_state == "Stressed":
            add("What is one small thing within your control right now that could reduce the pressure even slightly?")
            add("If nothing changed externally, what internal shift would make this feel easier to handle?")
        elif mental_state == "Reflective":
            add("What insight here feels most important to carry forward into your next decision?")
            add("How might this understanding change how you respond next time?")
        elif mental_state == "Analytical":
            add("You've reasoned this out well—what decision feels right when you step back from analysis?")
            add("If you had to act without more thinking, what would you choose?")
        elif mental_state == "Calm":
            add("What is helping you stay steady right now, and how can you maintain it?")
        else:
            add("What feels most unresolved in this situation right now?")

        # --- 2. Thought pattern driven prompt (adds depth) ---
        if thought_patterns:
            pattern = thought_patterns[0].lower()

            if "rumination" in pattern or "repetition" in pattern:
                add("What new angle or perspective have you not yet considered?")
            elif "uncertainty" in pattern or "questioning" in pattern:
                add("Which question here actually needs an answer, and which can be left open?")
            elif "anxiety" in pattern:
                add("What is the most likely outcome, not just the worst-case one?")
            elif "heavy" in pattern or "slowed" in pattern:
                add("What feels hardest to move forward on right now, and why?")
            elif "reflective" in pattern:
                add("Which part of this reflection feels most true to you?")

        # --- 3. Personality-aware prompt ---
        if extraversion >= 0.6:
            add("Who could you talk to about this, and what would you want them to truly understand?")
        else:
            add("When you've felt something similar before, what helped you move through it?")

        # --- 4. Depth control (based on text length) ---
        if text_length < 120:
            add("Take the most emotionally charged sentence you wrote and expand on it—what’s underneath it?")
        elif text_length > 300:
            add("Looking back at what you've written, what stands out as the most important thread?")

        # --- 5. Optional openness boost ---
        if openness >= 0.65:
            add("Is there a deeper meaning or pattern here that connects to a bigger part of your life?")

        return prompts[:3]