#!/usr/bin/env python3
"""
Quick training script for Cognisight models

Usage:
    python src/run_training.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train_model_pipeline

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 COGNISIGHT MODEL TRAINING")
    print("="*70 + "\n")
    
    # Ask user which models to train
    print("Choose training mode:")
    print("1. Fast (XGBoost only) - ~1 minute")
    print("2. Full (all models) - ~5 minutes")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    train_all = choice == '2'
    
    # Train
    results = train_model_pipeline(train_all_models=train_all)
    
    # Success message
    print("\n" + "="*70)
    print("✅ Training complete!")
    print("="*70)
    print("\nModels saved to: ./models/")
    print("\nNext, run:")
    print("  streamlit run app.py")
    print("\n" + "="*70)
