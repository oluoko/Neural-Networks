"""
This script processes audio files for genre classification using our trained PyTorch model.
Usage: python inference.py <audio_file_path> <model_directory>
"""

import sys
import os
import json
import torch
import librosa
import numpy as np
import scipy.stats as stats
import pickle
from pathlib import Path


# Define the neural network model (same structure as during training)
class MusicGenreClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicGenreClassifier, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# Function to extract features from audio (same as in training)
def extract_features(audio_array, sample_rate):
    # Extract various audio features
    # Mel-frequency cepstral coefficients
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio_array, sr=sample_rate
    )[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_array, sr=sample_rate
    )[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=sample_rate)[
        0
    ]

    # Rhythm features
    tempo, _ = librosa.beat.beat_track(y=audio_array, sr=sample_rate)

    # Zero crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_array)[0]

    # Compute statistics for each feature
    features = {}

    # MFCC stats
    for i in range(mfccs.shape[0]):
        features[f"mfcc{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc{i+1}_std"] = np.std(mfccs[i])
        features[f"mfcc{i+1}_skew"] = stats.skew(mfccs[i])
        features[f"mfcc{i+1}_kurtosis"] = stats.kurtosis(mfccs[i])

    # Other features stats
    for name, feature in [
        ("spectral_centroid", spectral_centroid),
        ("spectral_bandwidth", spectral_bandwidth),
        ("spectral_rolloff", spectral_rolloff),
        ("zero_crossing_rate", zero_crossing_rate),
    ]:
        features[f"{name}_mean"] = np.mean(feature)
        features[f"{name}_std"] = np.std(feature)
        features[f"{name}_skew"] = stats.skew(feature)
        features[f"{name}_kurtosis"] = stats.kurtosis(feature)

    # Add tempo
    features["tempo"] = tempo

    return features


def main():
    if len(sys.argv) != 3:
        print("Usage: python inference.py <audio_file_path> <model_directory>")
        sys.exit(1)

    audio_path = sys.argv[1]
    model_dir = sys.argv[2]

    try:
        # Load the audio file using librosa
        audio_array, sample_rate = librosa.load(audio_path, sr=22050)

        # Extract features
        features = extract_features(audio_array, sample_rate)

        # Convert features to the expected format
        feature_vector = np.array([[value for value in features.values()]])

        # Load the scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        # Scale features
        scaled_features = scaler.transform(feature_vector)

        # Load the label mapping
        label_mapping_path = os.path.join(model_dir, "label_mapping.json")
        with open(label_mapping_path, "r") as mapping_file:
            label_mapping = json.load(mapping_file)

            # Make sure keys are converted to integers for proper indexing
            inverse_mapping = {
                int(idx): str(label) for label, idx in label_mapping.items()
            }

        # Load the model
        model_path = os.path.join(model_dir, "music_genre_classifier.pth")
        input_size = scaled_features.shape[1]
        num_classes = len(label_mapping)

        model = MusicGenreClassifier(input_size, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

        # Make predictions
        with torch.no_grad():
            input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

            # Get the predicted class and confidence
            predicted_idx = torch.argmax(probabilities).item()
            predicted_genre = inverse_mapping[predicted_idx]
            confidence = probabilities[predicted_idx].item()

            # Format and return all genre probabilities
            genre_confidences = []
            for idx, prob in enumerate(probabilities):
                if idx in inverse_mapping:  # Make sure we have a mapping for this index
                    genre_confidences.append(
                        {"genre": inverse_mapping[idx], "confidence": prob.item()}
                    )

            # Create result JSON
            result = {
                "genre": predicted_genre,
                "confidence": confidence,
                "genreConfidences": genre_confidences,
            }

            print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
