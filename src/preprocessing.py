import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """
    Handles audio preprocessing for the UrbanSound8K dataset.
    """
    
    def __init__(self, sample_rate=22050, n_mels=128, fmax=8000, target_shape=(128, 173)):
        """
        Initialize the preprocessor.
        
        Args:
            sample_rate (int): Audio sample rate
            n_mels (int): Number of mel frequency bands
            fmax (int): Maximum frequency for mel spectrogram
            target_shape (tuple): Target shape for spectrograms
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fmax = fmax
        self.target_shape = target_shape
        self.label_encoder = LabelEncoder()
        
    def load_metadata(self, metadata_path):
        """
        Load UrbanSound8K metadata.
        
        Args:
            metadata_path (str): Path to metadata CSV file
            
        Returns:
            pd.DataFrame: Metadata dataframe
        """
        try:
            metadata = pd.read_csv(metadata_path)
            print(f"Loaded metadata with {len(metadata)} entries")
            return metadata
        except Exception as e:
            raise ValueError(f"Error loading metadata: {str(e)}")
    
    def clean_metadata(self, metadata):
        """
        Clean and validate metadata.
        
        Args:
            metadata (pd.DataFrame): Raw metadata
            
        Returns:
            pd.DataFrame: Cleaned metadata
        """
        # Remove entries with missing values
        metadata = metadata.dropna()
        
        # Validate class names
        valid_classes = ['air_conditioner', 'car_horn', 'children_playing', 
                        'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 
                        'jackhammer', 'siren', 'street_music']
        
        metadata = metadata[metadata['class'].isin(valid_classes)]
        
        # Calculate duration
        metadata['duration'] = metadata['end'] - metadata['start']
        
        # Filter out extremely short or long audio files
        metadata = metadata[(metadata['duration'] >= 0.5) & (metadata['duration'] <= 10.0)]
        
        print(f"After cleaning: {len(metadata)} valid entries")
        return metadata
    
    def extract_features(self, audio_path):
        """
        Extract mel-spectrogram features from audio file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            np.ndarray: Mel-spectrogram features with shape (128, 173, 1)
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract mel-spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, fmax=self.fmax)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Resize to target shape
            if S_dB.shape != self.target_shape:
                S_dB = cv2.resize(S_dB, (self.target_shape[1], self.target_shape[0]))
            
            # Add channel dimension for CNN (height, width, channels)
            S_dB = np.expand_dims(S_dB, axis=-1)
            
            return S_dB
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {str(e)}")
            return None
    
    def preprocess_dataset(self, metadata, audio_dir, max_files=None):
        """
        Preprocess entire dataset.
        
        Args:
            metadata (pd.DataFrame): Dataset metadata
            audio_dir (str): Directory containing audio files
            max_files (int): Maximum number of files to process (for testing)
            
        Returns:
            tuple: (features, labels, class_names)
        """
        features = []
        labels = []
        failed_files = []
        
        # Limit files if specified
        if max_files:
            metadata = metadata.head(max_files)
        
        print(f"Processing {len(metadata)} audio files...")
        
        for idx, row in metadata.iterrows():
            # Construct audio file path
            audio_path = os.path.join(audio_dir, f"fold{row['fold']}", row['slice_file_name'])
            
            # Extract features
            feature = self.extract_features(audio_path)
            
            if feature is not None:
                features.append(feature)
                labels.append(row['class'])
            else:
                failed_files.append(audio_path)
            
            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(metadata)} files")
        
        if failed_files:
            print(f"Failed to process {len(failed_files)} files")
        
        # Convert to numpy arrays
        features = np.array(features)
        
        # Add channel dimension for CNN
        features = np.expand_dims(features, axis=-1)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        class_names = self.label_encoder.classes_
        
        print(f"Successfully processed {len(features)} files")
        print(f"Features shape: {features.shape}")
        print(f"Number of classes: {len(class_names)}")
        
        return features, labels_encoded, class_names
    
    def create_train_test_split(self, features, labels, test_size=0.2, random_state=42):
        """
        Split dataset into training and testing sets.
        
        Args:
            features (np.ndarray): Audio features
            labels (np.ndarray): Encoded labels
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            features, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
    
    def visualize_class_distribution(self, metadata):
        """
        Visualize class distribution in the dataset.
        
        Args:
            metadata (pd.DataFrame): Dataset metadata
        """
        plt.figure(figsize=(12, 6))
        sns.countplot(data=metadata, y='class', order=sorted(metadata['class'].unique()))
        plt.title('Class Distribution in UrbanSound8K Dataset')
        plt.xlabel('Count')
        plt.ylabel('Class')
        plt.tight_layout()
        plt.show()
    
    def visualize_duration_distribution(self, metadata):
        """
        Visualize audio duration distribution.
        
        Args:
            metadata (pd.DataFrame): Dataset metadata
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(metadata['duration'], bins=50)
        plt.title('Distribution of Audio Durations')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Count')
        plt.show()
    
    def visualize_sample_spectrograms(self, metadata, audio_dir, n_samples=3):
        """
        Visualize sample spectrograms from different classes.
        
        Args:
            metadata (pd.DataFrame): Dataset metadata
            audio_dir (str): Directory containing audio files
            n_samples (int): Number of samples to visualize
        """
        # Get unique classes
        classes = metadata['class'].unique()[:n_samples]
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(15, 4*n_samples))
        
        for i, class_name in enumerate(classes):
            # Get a sample from this class
            sample = metadata[metadata['class'] == class_name].iloc[0]
            audio_path = os.path.join(audio_dir, f"fold{sample['fold']}", sample['slice_file_name'])
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Plot waveform
            axes[i, 0].plot(y)
            axes[i, 0].set_title(f'Waveform: {class_name}')
            axes[i, 0].set_xlabel('Sample')
            axes[i, 0].set_ylabel('Amplitude')
            
            # Plot spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, fmax=self.fmax)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=self.fmax, ax=axes[i, 1])
            axes[i, 1].set_title(f'Mel-Spectrogram: {class_name}')
            fig.colorbar(img, ax=axes[i, 1], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Example usage of the AudioPreprocessor.
    """
    # Initialize preprocessor
    preprocessor = AudioPreprocessor()
    
    # Load metadata
    metadata_path = "../data/raw/UrbanSound8K/metadata/UrbanSound8K.csv"
    metadata = preprocessor.load_metadata(metadata_path)
    
    # Clean metadata
    metadata = preprocessor.clean_metadata(metadata)
    
    # Visualize data
    preprocessor.visualize_class_distribution(metadata)
    preprocessor.visualize_duration_distribution(metadata)
    
    # Process dataset (limit to 100 files for quick testing)
    audio_dir = "../data/raw/UrbanSound8K/audio"
    features, labels, class_names = preprocessor.preprocess_dataset(
        metadata, audio_dir, max_files=100
    )
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(
        features, labels
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")


if __name__ == "__main__":
    main()
