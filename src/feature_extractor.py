"""
Feature extraction for personality analysis

Combines TF-IDF features with linguistic features for comprehensive text representation.
Total: 120+ dimensions combining vocabulary (100) and writing style (20+)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

from src.preprocessing import tokenize_sentences, tokenize_words, get_pos_tags
from src.emotion_extractor import EmotionExtractor

class PersonalityFeatureExtractor:
    """Extract comprehensive personality features from text."""
    
    def __init__(self, max_tfidf_features: int = 100):
        """
        Initialize feature extractor.
        
        Args:
            max_tfidf_features: Maximum TF-IDF features to extract
        """
        self.max_tfidf_features = max_tfidf_features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        self.sia = SentimentIntensityAnalyzer()
        self.is_fitted = False

        self.emotion_extractor = EmotionExtractor()

    def fit_tfidf(self, texts: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on texts.
        
        Args:
            texts: List of text samples
        """
        self.tfidf_vectorizer.fit(texts)
        self.is_fitted = True
    
    def extract_linguistic_features(self, text: str) -> dict:
        """
        Extract comprehensive linguistic features from text.
        
        Features (30 total):
        1. avg_word_length - Writing complexity
        2. lexical_diversity - Unique words / total words
        3. avg_sentence_length - Structure
        4. sentence_length_variance - Language variability
        5. sentiment_score - Overall tone (-1 to 1)
        6. sentiment_variance - Emotional consistency
        7. sentiment_intensity - Absolute sentiment force
        8. emotional_word_ratio - Emotional language
        9. positive_word_ratio - Positivity
        10. negative_word_ratio - Negativity
        11. first_person_ratio - Self-focus
        12. second_person_ratio - Other-focus
        13. third_person_ratio - Objectivity
        14. question_ratio - Inquisitiveness
        15. exclamation_ratio - Expressiveness
        16. punctuation_diversity - Accent variety
        17. uppercase_ratio - Emphasis
        18. digit_ratio - Data/number reference
        19. stopword_ratio - Common words
        20. readability_score - Text complexity
        21. pronoun_variety - Pronoun diversity
        22. complex_word_ratio - Sophisticated language
        23. coordinating_conjunction_ratio - Adds connectivity
        24. subordinating_conjunction_ratio - Adds complexity
        25. word_frequency_entropy - Info content diversity
        26. avg_word_frequency - Language specificity
        27. short_word_ratio - Simple language
        28. long_word_ratio - Complex vocabulary
        29. contraction_ratio - Conversational style
        30. tense_diversity - Temporal complexity
        """
        features = {}
        
        # Tokenize
        words = tokenize_words(text)
        sentences = tokenize_sentences(text)
        
        if not words:
            return {f'ling_feat_{i}': 0.0 for i in range(30)}
        
        # Basic metrics
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / len(words) if words else 0
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Sentiment analysis
        sentiment_scores = [self.sia.polarity_scores(s)['compound'] for s in sentences]
        sentiment_score = np.mean(sentiment_scores) if sentiment_scores else 0
        sentiment_variance = np.std(sentiment_scores) if sentiment_scores else 0
        
        # Emotional words
        emotional_words = {
            'positive': ['good', 'great', 'love', 'excellent', 'happy', 'wonderful', 'amazing'],
            'negative': ['bad', 'hate', 'terrible', 'sad', 'angry', 'awful', 'horrible']
        }
        
        words_lower = [w.lower() for w in words]
        positive_count = sum(1 for w in words_lower if w in emotional_words['positive'])
        negative_count = sum(1 for w in words_lower if w in emotional_words['negative'])
        emotional_count = positive_count + negative_count
        
        emotional_word_ratio = emotional_count / len(words) if words else 0
        positive_word_ratio = positive_count / len(words) if words else 0
        negative_word_ratio = negative_count / len(words) if words else 0
        
        # Pronoun analysis
        pronouns = {
            'first': ['i', 'me', 'my', 'mine', 'we', 'us', 'our'],
            'second': ['you', 'your', 'yours'],
            'third': ['he', 'she', 'it', 'his', 'her', 'its', 'they', 'them', 'their']
        }
        
        first_count = sum(1 for w in words_lower if w in pronouns['first'])
        second_count = sum(1 for w in words_lower if w in pronouns['second'])
        third_count = sum(1 for w in words_lower if w in pronouns['third'])
        
        first_person_ratio = first_count / len(words) if words else 0
        second_person_ratio = second_count / len(words) if words else 0
        third_person_ratio = third_count / len(words) if words else 0
        
        # Punctuation analysis
        question_count = text.count('?')
        exclamation_count = text.count('!')
        question_ratio = question_count / len(sentences) if sentences else 0
        exclamation_ratio = exclamation_count / len(sentences) if sentences else 0
        
        punctuation_chars = set('.,!?;:-')
        punctuation_diversity = len(set(c for c in text if c in punctuation_chars)) / len(punctuation_chars)
        
        # Capitalization
        uppercase_count = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_count / len(text) if text else 0
        
        # Digits
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / len(text) if text else 0
        
        # Stopwords (common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'is', 'to', 'of'}
        stopword_count = sum(1 for w in words_lower if w in stop_words)
        stopword_ratio = stopword_count / len(words) if words else 0
        
        # Flesch-Kincaid readability approximation
        syllable_count = sum(len(w) // 3 for w in words)  # Rough approximation
        readability_score = min(1.0, (syllable_count / max(len(words), 1)) / 3)
        
        # Pronoun variety
        total_pronouns = first_count + second_count + third_count
        pronoun_variety = len(set(w for w in words_lower if w in pronouns['first'] + pronouns['second'] + pronouns['third']))
        pronoun_variety = pronoun_variety / max(total_pronouns, 1) if total_pronouns > 0 else 0
        
        # Complex words
        complex_word_count = sum(1 for w in words if len(w) > 6)
        complex_word_ratio = complex_word_count / len(words) if words else 0
        
        # Sentence length variance
        sentence_lengths = [len(tokenize_words(s)) for s in sentences]
        sentence_length_variance = np.std(sentence_lengths) if sentence_lengths else 0
        
        # Sentiment intensity (absolute value avg)
        sentiment_intensity = np.mean([abs(s) for s in sentiment_scores]) if sentiment_scores else 0
        
        # Conjunctions
        coord_conj = {'and', 'but', 'or', 'nor', 'yet', 'so'}
        subord_conj = {'because', 'since', 'when', 'if', 'although', 'while', 'whereas'}
        
        coord_count = sum(1 for w in words_lower if w in coord_conj)
        subord_count = sum(1 for w in words_lower if w in subord_conj)
        
        coordinating_conjunction_ratio = coord_count / len(words) if words else 0
        subordinating_conjunction_ratio = subord_count / len(words) if words else 0
        
        # Word frequency entropy (using simple word repetition)
        word_counts = {}
        for w in words_lower:
            word_counts[w] = word_counts.get(w, 0) + 1
        
        frequencies = np.array(list(word_counts.values()))
        probabilities = frequencies / frequencies.sum()
        word_frequency_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        word_frequency_entropy = min(1.0, word_frequency_entropy / 10)  # Normalize
        
        # Average word frequency (lower = more varied vocabulary)
        avg_word_frequency = np.mean(frequencies) if len(frequencies) > 0 else 1.0
        avg_word_frequency = min(1.0, 1.0 / avg_word_frequency)  # Invert and normalize
        
        # Short and long words
        short_word_count = sum(1 for w in words if len(w) <= 3)
        long_word_count = sum(1 for w in words if len(w) >= 7)
        
        short_word_ratio = short_word_count / len(words) if words else 0
        long_word_ratio = long_word_count / len(words) if words else 0
        
        # Contractions (informal/conversational)
        contractions = {"'ll", "'ve", "'re", "'d", "'s", "'t", "n't"}
        contraction_count = sum(1 for w in words_lower if any(c in w for c in contractions))
        contraction_ratio = contraction_count / len(words) if words else 0
        
        # Tense diversity (very rough: past tense -ed, present -ing, etc.)
        past_count = sum(1 for w in words_lower if w.endswith('ed'))
        present_count = sum(1 for w in words_lower if w.endswith('ing'))
        
        tense_diversity = len({past_count > 0, present_count > 0, True}) / 3  # Normalize
        
        # Build features dict (30 features)
        features['ling_feat_0'] = avg_word_length / 10  # Normalize
        features['ling_feat_1'] = lexical_diversity
        features['ling_feat_2'] = avg_sentence_length / 20  # Normalize
        features['ling_feat_3'] = sentence_length_variance / 10  # Normalize
        features['ling_feat_4'] = (sentiment_score + 1) / 2  # Convert -1,1 to 0,1
        features['ling_feat_5'] = sentiment_variance
        features['ling_feat_6'] = sentiment_intensity
        features['ling_feat_7'] = emotional_word_ratio
        features['ling_feat_8'] = positive_word_ratio
        features['ling_feat_9'] = negative_word_ratio
        features['ling_feat_10'] = first_person_ratio
        features['ling_feat_11'] = second_person_ratio
        features['ling_feat_12'] = third_person_ratio
        features['ling_feat_13'] = question_ratio
        features['ling_feat_14'] = exclamation_ratio
        features['ling_feat_15'] = punctuation_diversity
        features['ling_feat_16'] = uppercase_ratio
        features['ling_feat_17'] = digit_ratio
        features['ling_feat_18'] = stopword_ratio
        features['ling_feat_19'] = readability_score
        features['ling_feat_20'] = pronoun_variety
        features['ling_feat_21'] = complex_word_ratio
        features['ling_feat_22'] = coordinating_conjunction_ratio
        features['ling_feat_23'] = subordinating_conjunction_ratio
        features['ling_feat_24'] = word_frequency_entropy
        features['ling_feat_25'] = avg_word_frequency
        features['ling_feat_26'] = short_word_ratio
        features['ling_feat_27'] = long_word_ratio
        features['ling_feat_28'] = contraction_ratio
        features['ling_feat_29'] = tense_diversity
        
        return features
    
    def extract_tfidf_features(self, text: str) -> Optional[np.ndarray]:
        """
        Extract TF-IDF features from text.
        
        Args:
            text: Input text
            
        Returns:
            TF-IDF feature vector (100 dimensions)
        """
        if not self.is_fitted:
            return None
        
        try:
            tfidf_vector = self.tfidf_vectorizer.transform([text]).toarray()
            return tfidf_vector[0]
        except:
            return None
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract features for model inference — matches the 130-dim training shape.

        Returns:
            Feature vector: 30 linguistic + 100 TF-IDF = 130 dims (matches saved model).

        Emotion features are computed and cached on self._last_emotion_features
        so the interpreter can use them without a second transformer call.
        They are NOT included in the returned vector — the saved model was trained
        on 130 features only (via extract_batch_features which never included emotion).
        Adding 6 extra dims causes a hard shape-mismatch error at predict() time.
        """
        # Linguistic features — fast, rule-based
        ling_features = self.extract_linguistic_features(text)
        ling_vector = np.array([ling_features[f'ling_feat_{i}'] for i in range(30)])

        # TF-IDF features
        tfidf_vector = self.extract_tfidf_features(text)
        if tfidf_vector is None:
            tfidf_vector = np.zeros(self.max_tfidf_features)

        # Emotion — run transformer ONCE, cache result for interpreter reuse
        emotion_features = self.emotion_extractor.extract(text)

        # Store both sub-components for analyzer.py to read without re-running
        self._last_ling_features = ling_features
        self._last_emotion_features = emotion_features

        # Return ONLY ling + tfidf → 130 dims, matching training shape exactly
        return np.concatenate([ling_vector, tfidf_vector])
    
    def extract_batch_features(self, texts: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract features for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Linguistic features DataFrame and TF-IDF array
        """
        ling_features_list = []
        tfidf_features_list = []
        
        for text in texts:
            # Linguistic
            ling_dict = self.extract_linguistic_features(text)
            ling_features_list.append(ling_dict)
            
            # TF-IDF
            tfidf_vec = self.extract_tfidf_features(text)
            if tfidf_vec is not None:
                tfidf_features_list.append(tfidf_vec)
        
        # Convert to DataFrames/arrays
        ling_df = pd.DataFrame(ling_features_list).fillna(0)
        tfidf_array = np.array(tfidf_features_list) if tfidf_features_list else None
        
        return ling_df, tfidf_array
    
    def get_feature_names(self) -> List[str]:
        # Linguistic features (30)
        ling_names = [f'ling_feat_{i}' for i in range(30)]

        # TF-IDF features (100)
        tfidf_names = [f'tfidf_{i}' for i in range(self.max_tfidf_features)]

        # Emotion features (6)
        emotion_names = [
            "emotion_sentiment",
            "emotion_intensity",
            "emotion_diversity",
            "emotion_polarity",
            "emotion_stability",
            "emotion_dominant"
        ]

        return ling_names + tfidf_names + emotion_names