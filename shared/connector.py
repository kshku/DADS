"""
Connector Module
Handles model inference for stutter detection using pre-trained models
Matches the training notebook architecture exactly
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    """CNN model architecture for stutter detection - matches training notebook"""

    def __init__(self, n_mels=64):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # shape: (batch, n_mels, time_frames) -> (batch, 1, n_mels, time_frames)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.fc(x)
        return x


class StutterDetector:
    """Connector class for handling stutter detection using 5 separate binary models"""

    def __init__(self, models_dir=None, detection_threshold=0.5):
        """
        Initialize the stutter detector with pre-trained binary models

        Args:
            models_dir: Path to directory containing .pth model files (default: auto-detect)
            detection_threshold: Probability threshold for positive detection (default: 0.4)
        """
        # Auto-detect models directory relative to this file
        if models_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)  # Go up from shared/ to project root
            models_dir = os.path.join(project_root, "Model", "models", "copy")
            if not os.path.exists(models_dir):
                raise RuntimeError(f"Models directory not found at: {models_dir}")

        self.models_dir = models_dir
        self.detection_threshold = detection_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.model_params = {}  # model_idx -> {'n_mels': int, 'n_fft': int, 'hop_length': int}

        # Default audio parameters
        self.sample_rate = 16000
        self.target_duration = 3.0
        self.target_length = int(self.target_duration * self.sample_rate)

        # Label names and model files
        self.label_dict = {"prolongation": 0, "block": 1, "soundrep": 2, "wordrep": 3, "interjection": 4}

        self.model_files = [
            "prolongation_model_1024_512_128_40.pth",
            "block_model_1024_512_128_40.pth",
            "soundrep_model_1024_512_256_40.pth",
            "wordrep_model_1024_512_64_40.pth",
            "interjection_model_1024_512_128_40.pth",
        ]

        self._load_models()

    def _parse_model_params(self, model_file):
        """
        Parse training parameters from model filename.
        Convention: {type}_model_{n_fft}_{hop_length}_{n_mels}_{epochs}.pth
        """
        name = model_file.replace(".pth", "")
        parts = name.split("_")
        # e.g. prolongation_model_1024_512_128_40
        n_fft = int(parts[2])
        hop_length = int(parts[3])
        n_mels = int(parts[4])
        epochs = int(parts[5])
        return {"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels, "epochs": epochs}

    def _load_models(self):
        """Load all 5 pre-trained binary models"""
        for i, model_file in enumerate(self.model_files):
            model_path = os.path.join(self.models_dir, model_file)

            if not os.path.exists(model_path):
                self.models.append(None)
                continue

            try:
                params = self._parse_model_params(model_file)
                self.model_params[i] = params
                model = Model(n_mels=params["n_mels"])
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models.append(model)
            except Exception as e:
                raise RuntimeError(f"Failed to load {model_file}: {e}")

        loaded_count = sum(1 for m in self.models if m is not None)
        if loaded_count == 0:
            raise RuntimeError("No models could be loaded")

    def pad_or_truncate(self, y):
        """Pad or truncate audio to target length"""
        if len(y) < self.target_length:
            # Pad with zeros (silence) at the end
            pad_width = self.target_length - len(y)
            y = np.pad(y, (0, pad_width))
        elif len(y) > self.target_length:
            # Truncate
            y = y[: self.target_length]
        return y

    def extract_features(self, audio_data, n_mels, n_fft=1024, hop_length=512, normalize=True):
        """
        Extract mel spectrogram features from audio data

        Args:
            audio_data: Audio data as numpy array
            n_mels: Number of mel frequency bands
            n_fft: FFT window size
            hop_length: Hop length
            normalize: Whether to normalize features

        Returns:
            Mel spectrogram in dB scale (n_mels, time_frames)
        """
        y = self.pad_or_truncate(audio_data)

        mels = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        mels_db = librosa.power_to_db(mels, ref=np.max)

        if normalize:
            mean = np.mean(mels_db)
            std = np.std(mels_db) + 1e-10  # Avoid divide by zero
            mels_db = (mels_db - mean) / std

        return mels_db

    def _predict_single_model(self, mels_db, model_idx):
        """
        Run prediction on a single model

        Args:
            mels_db: Mel spectrogram features
            model_idx: Index of the model to use

        Returns:
            (label_name, probability, is_detected)
        """
        model = self.models[model_idx]
        if model is None:
            return None

        label_name = list(self.label_dict.keys())[model_idx]

        with torch.no_grad():
            tensor = torch.tensor(mels_db, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = model(tensor)
            prob = torch.sigmoid(output).item()
            is_detected = prob > self.detection_threshold

        return label_name, prob, is_detected

    def _predict_model(self, audio_data, model_idx, n_mels, n_fft, hop_length):
        """
        Extract features with model-specific params and run prediction

        Args:
            audio_data: Raw audio numpy array
            model_idx: Index of the model
            n_mels: Number of mel bands for this model
            n_fft: FFT size for this model
            hop_length: Hop length for this model

        Returns:
            (label_name, probability, is_detected) or None
        """
        if self.models[model_idx] is None:
            return None
        mels_db = self.extract_features(audio_data, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        return self._predict_single_model(mels_db, model_idx)

    def process_audio_file(self, audio_path, callback=None):
        """
        Process entire audio file by splitting into 3-second chunks and detecting stutters

        Args:
            audio_path: Path to audio file
            callback: Optional callback function(chunk_idx, total_chunks, results)

        Returns:
            Dictionary with detection results for each chunk
        """
        if not any(self.models):
            raise RuntimeError("No models loaded. Cannot perform detection.")

        print(f"\nProcessing audio file: {audio_path}")

        # Load entire audio file
        try:
            y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            raise ValueError(f"Invalid audio file: {audio_path} ({e})")

        total_duration = len(y) / self.sample_rate
        print(f"Total audio duration: {total_duration:.2f}s")

        # Split into 3-second chunks
        chunk_size = self.target_length  # 3 seconds worth of samples
        total_chunks = int(np.ceil(len(y) / chunk_size))
        print(f"Splitting into {total_chunks} chunks of {self.target_duration}s each\n")

        all_results = {}

        for chunk_idx in range(total_chunks):
            start_sample = chunk_idx * chunk_size
            end_sample = min((chunk_idx + 1) * chunk_size, len(y))

            # Extract chunk
            chunk_audio = y[start_sample:end_sample]

            # Pad if last chunk is shorter than 3 seconds
            chunk_audio = self.pad_or_truncate(chunk_audio)

            # Calculate time range
            time_start = chunk_idx * self.target_duration
            time_end = min((chunk_idx + 1) * self.target_duration, total_duration)

            print(f"Chunk {chunk_idx + 1}/{total_chunks} ({time_start:.1f}s - {time_end:.1f}s)")

            # Extract features and run all models
            chunk_results = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                for i in range(5):
                    if self.models[i] is not None:
                        params = self.model_params[i]
                        futures[
                            executor.submit(
                                self._predict_model,
                                chunk_audio,
                                i,
                                params["n_mels"],
                                params["n_fft"],
                                params["hop_length"],
                            )
                        ] = i

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        label_name, prob, is_detected = result
                        chunk_results[label_name] = {"probability": prob, "detected": is_detected}

                        if is_detected:
                            print(f"  ✓ {label_name}: {prob:.3f} - DETECTED")
                        else:
                            print(f"  ○ {label_name}: {prob:.3f}")

            print()  # Empty line between chunks

            # Store chunk results
            all_results[chunk_idx] = {"time_start": time_start, "time_end": time_end, "detections": chunk_results}

            # Call callback if provided (for progress updates)
            if callback:
                callback(chunk_idx, total_chunks, chunk_results)

        print(f"Processed {len(all_results)} chunks successfully\n")
        return all_results

    def process_audio_chunk_realtime(self, audio_data, sr):
        """
        Process a single audio chunk in real-time

        Args:
            audio_data: Audio data as numpy array
            sr: Sample rate

        Returns:
            Dictionary with detection results for all stutter types
        """
        # Resample if needed
        if sr != self.sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)

        # Pad or truncate
        audio_data = self.pad_or_truncate(audio_data)

        # Run predictions with per-model feature extraction
        results = {}
        for i in range(5):
            if self.models[i] is not None:
                params = self.model_params[i]
                result = self._predict_model(audio_data, i, params["n_mels"], params["n_fft"], params["hop_length"])
                if result:
                    label_name, prob, is_detected = result
                    results[label_name] = {"probability": prob, "detected": is_detected}

        return results

    def get_summary(self, results):
        """
        Get summary statistics from detection results

        Args:
            results: Results dictionary from process_audio_file

        Returns:
            Summary dictionary with counts and percentages
        """
        summary = {label: 0 for label in self.label_dict.keys()}
        total_chunks = len(results)

        for chunk_data in results.values():
            for label_name, detection in chunk_data["detections"].items():
                if detection["detected"]:
                    summary[label_name] += 1

        # Calculate percentages
        summary_with_pct = {}
        for label_name, count in summary.items():
            summary_with_pct[label_name] = {
                "count": count,
                "percentage": (count / total_chunks * 100) if total_chunks > 0 else 0,
            }

        return summary_with_pct
