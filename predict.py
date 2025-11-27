import os
import sys
import argparse
from src.predict import load_model

if __name__ == "__main__":
    # Add src to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser(description='Make predictions using the trained model')
    parser.add_argument('audio_path', type=str, help='Path to the audio file')
    parser.add_argument('--model', type=str, default='models/best_model.h5',
                      help='Path to the trained model')
    
    args = parser.parse_args()
    
    # Load the model
    classifier = load_model(args.model)
    
    # Make prediction
    result = classifier.predict_from_audio(args.audio_path)
    
    # Print results
    print("\nPrediction Results:")
    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("\nClass Probabilities:")
    for class_name, prob in result['all_predictions'].items():
        print(f"{class_name}: {prob:.4f}")