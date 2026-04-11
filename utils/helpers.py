"""
Utility functions and constants for Cognisight
"""

import os
import re
from pathlib import Path
from typing import Dict, List

# Configuration
CONFIG = {
    'model_dir': './models',
    'dataset_path': './data/mbti_dataset.csv',
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'min_text_length': 50,
}

# Model configurations
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 7,
        'learning_rate': 0.1,
    },
    'mlp': {
        'hidden_layer_sizes': (128, 64, 32),
        'learning_rate_init': 0.001,
        'max_iter': 500,
    }
}

# MBTI to Big Five mapping
MBTI_TO_BIG_FIVE = {
    'E': {'Extraversion': 0.75, 'Neuroticism': -0.1},
    'I': {'Extraversion': 0.25, 'Neuroticism': 0.1},
    'S': {'Openness': 0.25},
    'N': {'Openness': 0.75},
    'T': {'Agreeableness': 0.3, 'Neuroticism': 0.2},
    'F': {'Agreeableness': 0.7, 'Neuroticism': 0.6},
    'J': {'Conscientiousness': 0.8},
    'P': {'Conscientiousness': 0.4},
}

# Personality traits
TRAITS = [
    'Openness',
    'Conscientiousness',
    'Extraversion',
    'Agreeableness',
    'Neuroticism'
]

# Trait descriptions
TRAIT_DESCRIPTIONS = {
    'Openness': 'Curiosity and willingness to try new ideas',
    'Conscientiousness': 'Organization and dependability',
    'Extraversion': 'Sociability and assertiveness',
    'Agreeableness': 'Compassion and cooperation',
    'Neuroticism': 'Emotional sensitivity and stress response',
}


def normalize_text(text: str) -> str:
    """
    Normalize text for processing.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned and normalized text
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove extra punctuation
    text = re.sub(r'[^\w\s.!?,-]', '', text)
    
    return text.strip()


def get_config(key: str = None) -> Dict:
    """
    Get configuration value(s).
    
    Args:
        key: Specific config key, or None for all
        
    Returns:
        Configuration dict or single value
    """
    if key is None:
        return CONFIG
    return CONFIG.get(key)


def ensure_models_dir() -> str:
    """
    Ensure models directory exists.
    
    Returns:
        Path to models directory
    """
    model_dir = Path(CONFIG['model_dir'])
    model_dir.mkdir(exist_ok=True)
    return str(model_dir)


def ensure_data_dir() -> str:
    """
    Ensure data directory exists.
    
    Returns:
        Path to data directory
    """
    data_dir = Path(CONFIG['dataset_path']).parent
    data_dir.mkdir(exist_ok=True)
    return str(data_dir)


def mbti_to_big_five(mbti_type: str) -> Dict[str, float]:
    """
    Convert MBTI type to Big Five trait scores.
    
    Args:
        mbti_type: MBTI type (e.g., 'ENFP')
        
    Returns:
        Dictionary with Big Five trait scores (0-1)
    """
    big_five = {
        'Openness': 0.5,
        'Conscientiousness': 0.5,
        'Extraversion': 0.5,
        'Agreeableness': 0.5,
        'Neuroticism': 0.5,
    }
    
    for letter in mbti_type:
        if letter in MBTI_TO_BIG_FIVE:
            for trait, adjustment in MBTI_TO_BIG_FIVE[letter].items():
                big_five[trait] = big_five[trait] * 0.5 + adjustment * 0.5
    
    # Clamp to 0-1
    for trait in big_five:
        big_five[trait] = max(0.0, min(1.0, big_five[trait]))
    
    return big_five


def score_to_label(score: float) -> str:
    """
    Convert trait score to interpretable label.
    
    Args:
        score: Score between 0 and 1
        
    Returns:
        Descriptive label
    """
    if score < 0.25:
        return 'Very Low'
    elif score < 0.42:
        return 'Low'
    elif score < 0.58:
        return 'Moderate'
    elif score < 0.75:
        return 'High'
    else:
        return 'Very High'


def normalize_confidence(score: float, std: float = 0.1) -> float:
    """
    Normalize confidence score to 60-90% range.
    
    Args:
        score: Raw model score (0-1)
        std: Standard deviation for natural variation
        
    Returns:
        Normalized confidence (0.6-0.9)
    """
    del std

    confidence = 0.60 + (score * 0.30)
    return min(0.90, max(0.60, confidence))
