"""
Model utilities for Cognisight
"""

import pickle
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

from utils import get_config, ensure_models_dir


class ModelManager:
    """Manage model loading and saving."""
    
    @staticmethod
    def save_model(model: Any, name: str, trait: str = 'general') -> str:
        """
        Save a trained model.
        
        Args:
            model: Trained model object
            name: Model name (e.g., 'random_forest', 'xgboost')
            trait: Personality trait (optional)
            
        Returns:
            Path where model was saved
        """
        model_dir = ensure_models_dir()
        filename = f'{trait}_{name}.pkl' if trait != 'general' else f'{name}.pkl'
        filepath = os.path.join(model_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        return filepath
    
    @staticmethod
    def load_model(name: str, trait: str = 'general') -> Optional[Any]:
        """
        Load a trained model.
        
        Args:
            name: Model name
            trait: Personality trait (optional)
            
        Returns:
            Loaded model or None if not found
        """
        model_dir = get_config('model_dir')
        filename = f'{trait}_{name}.pkl' if trait != 'general' else f'{name}.pkl'
        filepath = os.path.join(model_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    @staticmethod
    def list_models() -> Dict[str, list]:
        """
        List all available models.
        
        Returns:
            Dictionary of model types and available models
        """
        model_dir = get_config('model_dir')
        if not os.path.exists(model_dir):
            return {}
        
        models = {}
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl'):
                key = filename.replace('.pkl', '')
                models[key] = True
        
        return models


class ModelFactory:
    """Create model instances with configured parameters."""
    
    @staticmethod
    def create_random_forest() -> RandomForestRegressor:
        """Create Random Forest model with configured parameters."""
        from utils import MODEL_CONFIG
        config = MODEL_CONFIG['random_forest']
        
        return RandomForestRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            random_state=42,
            n_jobs=-1
        )
    
    @staticmethod
    def create_xgboost() -> xgb.XGBRegressor:
        """Create XGBoost model with configured parameters."""
        from utils import MODEL_CONFIG
        config = MODEL_CONFIG['xgboost']
        
        return xgb.XGBRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            random_state=42,
            verbosity=0
        )
    
    @staticmethod
    def create_mlp() -> MLPRegressor:
        """Create MLP neural network with configured parameters."""
        from utils import MODEL_CONFIG
        config = MODEL_CONFIG['mlp']
        
        return MLPRegressor(
            hidden_layer_sizes=config['hidden_layer_sizes'],
            learning_rate_init=config['learning_rate_init'],
            max_iter=config['max_iter'],
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
