"""
Text preprocessing utilities for Cognisight
"""

import re
import nltk
from typing import List

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = ' '.join(text.split())
    return text


def tokenize_sentences(text: str) -> List[str]:
    """Tokenize text into sentences."""
    try:
        return nltk.sent_tokenize(text)
    except:
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


def tokenize_words(text: str) -> List[str]:
    """Tokenize text into words."""
    try:
        return nltk.word_tokenize(text.lower())
    except:
        return re.findall(r'\b\w+\b', text.lower())


def get_pos_tags(text: str) -> List[tuple]:
    """Get part-of-speech tags."""
    try:
        words = tokenize_words(text)
        return nltk.pos_tag(words)
    except Exception as e:
        print(f"POS tagging failed: {e}")
        return []
