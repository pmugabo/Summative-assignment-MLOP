import os
import numpy as np
import tensorflow as tf
from .data_loader import UrbanSound8KLoader
import librosa
import joblib

class AudioClassifier:
    def __init__(self, model_path='models/best_model.h5'):
        """
        Initialize the audio classifier.
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model = tf.keras.models.load_model(model_path)
        self.data_loader = UrbanSound8KLoader()
        self.classes = self.data_loader.classes
        
    def predict_from_audio(self, audio_path, sr=22050):
        """
        Make predictions on an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            sr (int): Sample rate
            
        Returns:
            dict: Prediction results
        """
        # Load and preprocess audio
        audio, _ = self.data_loader.load_audio_file(audio_path)
        
        # Resample if needed
        if sr != 22050:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        
        # Extract features
        features = self.data_loader.extract_features(audio, sr=22050)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        features = np.expand_dims(features, axis=-1)  # Add channel dimension
        
        # Make prediction
        predictions = self.model.predict(features)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        return {
            'class': self.classes[predicted_class_idx],
            'class_index': int(predicted_class_idx),
            'confidence': confidence,
            'all_predictions': {
                class_name: float(prob) 
                for class_name, prob in zip(self.classes, predictions[0])
            }
        }
    
    def evaluate_on_test_set(self, test_fold=10):
        """
        Evaluate the model on the test set.
        
        Args:
            test_fold (int): Fold number to use as test set
            
        Returns:
            dict: Evaluation metrics
        """
        # Load test data
        X_test, y_test = self.data_loader.load_dataset(folds=[test_fold])
        X_test = np.expand_dims(X_test, -1)  # Add channel dimension
        
        # Evaluate
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist()
        }

def load_model(model_path='models/best_model.h5'):
    """
    Load a trained model.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        AudioClassifier: Loaded model
    """
    return AudioClassifier(model_path)