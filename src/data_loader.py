import os
import numpy as np
import pandas as pd
import librosa
import warnings
import cv2

warnings.filterwarnings('ignore')


class UrbanSound8KLoader:
    def __init__(self, data_path='data/raw/UrbanSound8K'):
        """
        Initialize the UrbanSound8K data loader.
        """
        self.data_path = data_path
        self.metadata = pd.read_csv(os.path.join(data_path, 'metadata', 'UrbanSound8K.csv'))
        self.classes = sorted(self.metadata['class'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

        # Print dataset info
        available_folds = sorted(self.metadata['fold'].unique())
        print(f"Available folds: {available_folds}")
        print(f"Number of samples: {len(self.metadata)}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Classes: {self.classes}")

    def get_audio_path(self, row):
        """Return the full path to an audio file based on metadata row."""
        base_path = os.path.abspath(self.data_path)
        return os.path.join(
            base_path,
            'audio',
            f"fold{row['fold']}",
            row['slice_file_name']
        )

    def extract_features(self, file_path, target_shape=(128, 173), max_duration=None, augment=False):
        """Extract Mel-spectrogram features with consistent shape."""
        try:
            # Load audio with librosa
            y, sr = librosa.load(file_path, sr=22050, duration=max_duration)
            
            # Apply data augmentation only if requested (for training)
            if augment:
                if np.random.random() > 0.5:
                    y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
                if np.random.random() > 0.5:
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-1, 1))
                
            # Extract mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Resize if needed
            if S_dB.shape != target_shape:
                S_dB = cv2.resize(S_dB, (target_shape[1], target_shape[0]))
                
            # Add channel dimension
            S_dB = np.expand_dims(S_dB, axis=-1)
            
            # Normalize
            S_dB = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)
            
            return S_dB
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def load_dataset(self, folds=None, max_duration=None):
        """
        Load dataset for specified folds. Skips missing files automatically.
        """
        if folds is None:
            folds = range(1, 7)

        fold_data = self.metadata[self.metadata['fold'].isin(folds)].copy()
        X, y = [], []
        processed_files, missing_files = 0, 0
        max_warnings, warning_count = 5, 0

        print(f"\nLoading data for folds: {list(folds)}")
        print(f"Found {len(fold_data)} samples in metadata")

        for _, row in fold_data.iterrows():
            file_path = self.get_audio_path(row)
            if not os.path.exists(file_path):
                missing_files += 1
                if warning_count < max_warnings:
                    print(f"Warning: File not found - {file_path}")
                    warning_count += 1
                elif warning_count == max_warnings:
                    print("Additional missing files not shown...")
                    warning_count += 1
                continue

            features = self.extract_features(file_path, max_duration=max_duration, augment=False)
            if features is not None:
                X.append(features)
                y.append(self.class_to_idx[row['class']])
                processed_files += 1
                if processed_files % 100 == 0:
                    print(f"Processed {processed_files} files...")

        if missing_files > 0:
            print(f"\nWarning: Could not find {missing_files} files out of {len(fold_data)}")

        if len(X) > 0:
            X = np.array(X)
            y = np.array(y)
            print(f"\nSuccessfully loaded {len(X)} samples")
            print(f"X shape: {X.shape}, y shape: {y.shape}")
            return X, y
        else:
            print("\nNo valid files found for the specified folds!")
            return np.array([]), np.array([])
