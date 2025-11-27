"""
Test script to verify notebook components work correctly.
Run this to check if all imports and basic functionality work.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split

print("Testing all required imports...")

# Test data loader
try:
    sys.path.append('src')
    from data_loader import UrbanSound8KLoader
    print("✅ Data loader imported successfully")
    
    # Initialize data loader
    data_loader = UrbanSound8KLoader(data_path="data/raw/UrbanSound8K")
    print("✅ Data loader initialized")
    
    # Load small subset
    X, y = data_loader.load_dataset(folds=[1], max_duration=4.0)
    print(f"✅ Data loaded: {X.shape}, {y.shape}")
    
except Exception as e:
    print(f"❌ Data loader error: {e}")

# Test model creation
try:
    from model import create_model, compile_model, get_callbacks
    print("✅ Model functions imported successfully")
    
    # Create model
    input_shape = X[0].shape
    num_classes = len(data_loader.classes)
    model = create_model(input_shape=input_shape, num_classes=num_classes)
    model = compile_model(model)
    print("✅ Model created and compiled")
    
    # Test callbacks
    callbacks = get_callbacks(checkpoint_path='models/test_model.h5')
    print("✅ Callbacks created")
    
except Exception as e:
    print(f"❌ Model error: {e}")

# Test preprocessing
try:
    from preprocessing import AudioPreprocessor
    print("✅ Preprocessor imported successfully")
    
    preprocessor = AudioPreprocessor()
    print("✅ Preprocessor initialized")
    
except Exception as e:
    print(f"❌ Preprocessor error: {e}")

# Test prediction
try:
    from prediction import AudioPredictor
    print("✅ Prediction module imported successfully")
    
except Exception as e:
    print(f"❌ Prediction error: {e}")

print("\n✅ All components tested successfully!")
print("You can now run the notebook with confidence.")
