"""
Model utilities for Cognisight
Handles persistence of trained ML models (save/load/list).
"""

import pickle
import os
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestRegressor
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
        filepath = Path(model_dir) / filename
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")
            raise e
        
        return str(filepath)
    
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
        filepath = Path(model_dir) / filename
        
        if not filepath.exists():
            return None
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            return None

        return model
    
    @staticmethod
    def list_models() -> list:
        model_dir = Path(get_config('model_dir'))
        if not model_dir.exists():
            return []

        return [
            f.stem
            for f in model_dir.iterdir()
            if f.is_file() and f.suffix == '.pkl'
        ]


class ModelFactory:
    """Factory for creating configured ML models."""

    @staticmethod
    def create_random_forest() -> RandomForestRegressor:
        from utils import MODEL_CONFIG
        config = MODEL_CONFIG["random_forest"]

        return RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            random_state=42,
            n_jobs=-1,
            oob_score=True  # added for better evaluation insight
        )

    @staticmethod
    def create_xgboost() -> xgb.XGBRegressor:
        from utils import MODEL_CONFIG
        config = MODEL_CONFIG["xgboost"]

        return xgb.XGBRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            subsample=config.get("subsample", 0.8),
            colsample_bytree=config.get("colsample_bytree", 0.8),
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )