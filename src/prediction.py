"""
Single data point prediction module for UrbanSound8K audio classification.
This module handles individual audio file predictions using the trained model.
"""

import os
import numpy as np
import librosa
import tensorflow as tf
import cv2
import json
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPredictor:
    """
    Handles single audio file predictions using the trained CNN model.
    """
    
    def __init__(self, model_path: str = "models/best_cnn_model.h5"):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.class_names = [
            "air_conditioner", "car_horn", "children_playing", "dog_bark", 
            "drilling", "engine_idling", "gun_shot", "jackhammer", 
            "siren", "street_music"
        ]
        self.target_shape = (128, 173)
        self.sample_rate = 22050
        self.n_mels = 128
        self.fmax = 8000
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from disk."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Preprocess a single audio file for prediction.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            np.ndarray: Preprocessed audio features ready for model input
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract mel-spectrogram
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels, fmax=self.fmax
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Resize to match training input shape
            if S_dB.shape != self.target_shape:
                S_dB = cv2.resize(S_dB, (self.target_shape[1], self.target_shape[0]))
            
            # Add batch and channel dimensions for CNN input
            S_dB = np.expand_dims(S_dB, axis=-1)  # Add channel dimension
            S_dB = np.expand_dims(S_dB, axis=0)   # Add batch dimension
            
            return S_dB
            
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {str(e)}")
            raise ValueError(f"Failed to preprocess audio: {str(e)}")
    
    def predict(self, audio_path: str, return_probabilities: bool = True) -> Dict:
        """
        Predict the class of a single audio file.
        
        Args:
            audio_path (str): Path to the audio file
            return_probabilities (bool): Whether to return all class probabilities
            
        Returns:
            Dict: Prediction results containing:
                - predicted_class: str - The predicted class name
                - confidence: float - Confidence score (0-1)
                - probabilities: dict - All class probabilities (if requested)
        """
        try:
            # Validate input file
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Preprocess audio
            features = self.preprocess_audio(audio_path)
            
            # Make prediction
            predictions = self.model.predict(features, verbose=0)
            predictions = predictions[0]  # Remove batch dimension
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions)
            confidence = float(np.max(predictions))
            predicted_class = self.class_names[predicted_class_idx]
            
            # Prepare result
            result = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "audio_file": os.path.basename(audio_path)
            }
            
            # Add all probabilities if requested
            if return_probabilities:
                probabilities = {
                    self.class_names[i]: float(predictions[i]) 
                    for i in range(len(self.class_names))
                }
                result["probabilities"] = probabilities
            
            logger.info(f"Prediction successful: {predicted_class} ({confidence:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_batch(self, audio_paths: List[str]) -> List[Dict]:
        """
        Predict multiple audio files.
        
        Args:
            audio_paths (List[str]): List of audio file paths
            
        Returns:
            List[Dict]: List of prediction results
        """
        results = []
        
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict {audio_path}: {str(e)}")
                results.append({
                    "audio_file": os.path.basename(audio_path),
                    "error": str(e)
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model information
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_path": self.model_path,
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "num_parameters": self.model.count_params(),
            "num_classes": len(self.class_names),
            "classes": self.class_names
        }


def predict_single_audio(audio_path: str, model_path: str = "models/best_cnn_model.h5") -> Dict:
    """
    Convenience function to predict a single audio file.
    
    Args:
        audio_path (str): Path to the audio file
        model_path (str): Path to the model file
        
    Returns:
        Dict: Prediction result
    """
    predictor = AudioPredictor(model_path)
    return predictor.predict(audio_path)


def main():
    """
    Example usage of the AudioPredictor.
    """
    # Initialize predictor
    predictor = AudioPredictor()
    
    # Print model info
    print("Model Information:")
    print(json.dumps(predictor.get_model_info(), indent=2))
    
    # Example prediction (replace with actual audio file path)
    audio_path = "../data/raw/UrbanSound8K/audio/fold1/100032-3-0-0.wav"
    
    if os.path.exists(audio_path):
        print(f"\nPredicting audio: {audio_path}")
        result = predictor.predict(audio_path)
        
        print("\nPrediction Results:")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        print("\nAll Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"{class_name}: {prob:.4f}")
    else:
        print(f"Audio file not found: {audio_path}")
        print("Please update the audio_path with a valid audio file.")


if __name__ == "__main__":
    main()
