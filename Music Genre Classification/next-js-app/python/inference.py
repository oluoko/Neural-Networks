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


# Function to safely compute statistics and handle NaNs
def safe_stats(feature):
    try:
        mean_val = np.mean(feature)
        std_val = np.std(feature)
        # Handle empty arrays or constant values that could cause issues with skew/kurtosis
        if len(feature) < 2 or np.all(feature == feature[0]):
            skew_val = 0
            kurtosis_val = 0
        else:
            skew_val = stats.skew(feature)
            kurtosis_val = stats.kurtosis(feature)

        # Replace NaN values with 0
        if np.isnan(mean_val):
            mean_val = 0
        if np.isnan(std_val):
            std_val = 0
        if np.isnan(skew_val):
            skew_val = 0
        if np.isnan(kurtosis_val):
            kurtosis_val = 0

        return mean_val, std_val, skew_val, kurtosis_val
    except Exception:
        # If any calculation fails, return zeros
        return 0, 0, 0, 0


# Function to extract features from audio (improved for robustness)
def extract_features(audio_array, sample_rate):
    # Extract various audio features
    # Handle empty or very short audio
    if len(audio_array) < sample_rate // 2:  # Less than 0.5 seconds
        print(
            json.dumps({"error": "Audio file too short or corrupted"}), file=sys.stderr
        )
        sys.exit(1)

    # Initialize features dictionary
    features = {}

    try:
        # Mel-frequency cepstral coefficients
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_array, sr=sample_rate
        )[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio_array, sr=sample_rate
        )[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_array, sr=sample_rate
        )[0]

        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=audio_array, sr=sample_rate)

        # Zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_array)[0]

        # Compute statistics for each feature
        # MFCC stats
        for i in range(mfccs.shape[0]):
            mean_val, std_val, skew_val, kurtosis_val = safe_stats(mfccs[i])
            features[f"mfcc{i+1}_mean"] = mean_val
            features[f"mfcc{i+1}_std"] = std_val
            features[f"mfcc{i+1}_skew"] = skew_val
            features[f"mfcc{i+1}_kurtosis"] = kurtosis_val

        # Other features stats
        for name, feature in [
            ("spectral_centroid", spectral_centroid),
            ("spectral_bandwidth", spectral_bandwidth),
            ("spectral_rolloff", spectral_rolloff),
            ("zero_crossing_rate", zero_crossing_rate),
        ]:
            mean_val, std_val, skew_val, kurtosis_val = safe_stats(feature)
            features[f"{name}_mean"] = mean_val
            features[f"{name}_std"] = std_val
            features[f"{name}_skew"] = skew_val
            features[f"{name}_kurtosis"] = kurtosis_val

        # Add tempo
        features["tempo"] = float(tempo) if not np.isnan(tempo) else 0.0

    except Exception as e:
        print(
            json.dumps({"error": f"Feature extraction failed: {str(e)}"}),
            file=sys.stderr,
        )
        sys.exit(1)

    return features


def main():
    if len(sys.argv) != 3:
        print("Usage: python inference.py <audio_file_path> <model_directory>")
        sys.exit(1)

    audio_path = sys.argv[1]
    model_dir = sys.argv[2]

    try:
        # Check if the file exists
        if not os.path.exists(audio_path):
            print(json.dumps({"error": "Audio file not found"}), file=sys.stderr)
            sys.exit(1)

        # Load the audio file using librosa with better error handling
        try:
            # Try to use a different offset if the file is corrupted at the beginning
            audio_array, sample_rate = librosa.load(
                audio_path, sr=22050, offset=0.0, duration=30.0
            )

            # Check if audio was successfully loaded
            if len(audio_array) == 0:
                print(
                    json.dumps({"error": "Failed to load audio file"}), file=sys.stderr
                )
                sys.exit(1)

        except Exception as e:
            # Try alternative loading methods
            try:
                # Try a different offset
                audio_array, sample_rate = librosa.load(
                    audio_path, sr=22050, offset=1.0, duration=30.0
                )
            except Exception:
                print(
                    json.dumps({"error": f"Could not load audio file: {str(e)}"}),
                    file=sys.stderr,
                )
                sys.exit(1)

        # Extract features
        features = extract_features(audio_array, sample_rate)

        # Make sure all feature values are single scalar values, not arrays or sequences
        for key, value in list(features.items()):
            if isinstance(value, (list, np.ndarray)) or np.isnan(value):
                features[key] = 0.0  # Replace problematic values

        # Convert features to the expected format - make sure all values are scalars
        feature_values = [float(value) for value in features.values()]
        feature_vector = np.array([feature_values])

        # Load the scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            print(json.dumps({"error": "Scaler file not found"}), file=sys.stderr)
            sys.exit(1)

        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        # Scale features
        scaled_features = scaler.transform(feature_vector)

        # Load the label mapping
        label_mapping_path = os.path.join(model_dir, "label_mapping.json")
        if not os.path.exists(label_mapping_path):
            print(
                json.dumps({"error": "Label mapping file not found"}), file=sys.stderr
            )
            sys.exit(1)

        with open(label_mapping_path, "r") as mapping_file:
            label_mapping = json.load(mapping_file)

            # Make sure keys are converted to integers for proper indexing
            inverse_mapping = {
                int(idx): str(label) for label, idx in label_mapping.items()
            }

        # Load the model
        model_path = os.path.join(model_dir, "music_genre_classifier.pth")
        if not os.path.exists(model_path):
            print(json.dumps({"error": "Model file not found"}), file=sys.stderr)
            sys.exit(1)

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
                "genreConfidences": sorted(
                    genre_confidences, key=lambda x: x["confidence"], reverse=True
                ),
            }

            print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
