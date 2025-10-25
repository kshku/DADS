"""
Connector Module
Handles model inference for stutter detection using pre-trained models
Matches the training notebook architecture exactly
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


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
    
    def __init__(self, models_dir="Model/models/copy", detection_threshold=0.5):
        """
        Initialize the stutter detector with pre-trained binary models
        
        Args:
            models_dir: Path to directory containing .pth model files
            detection_threshold: Probability threshold for positive detection (default: 0.5)
        """
        self.models_dir = models_dir
        self.detection_threshold = detection_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        
        # Model parameters - MATCH TRAINING EXACTLY
        self.window_size = 1024
        self.hop_length = 512
        self.n_mels = 64
        self.epochs = 40  # From copy folder models
        
        # Label names and model files
        self.label_dict = {
            'prolongation': 0,
            'block': 1,
            'soundrep': 2,
            'wordrep': 3,
            'interjection': 4
        }
        
        self.model_files = [
            'prolongation_model_1024_512_128_40.pth',
            'block_model_1024_512_128_40.pth',
            'soundrep_model_1024_512_256_40.pth',
            'wordrep_model_1024_512_64_40.pth',
            'interjection_model_1024_512_128_40.pth'
        ]
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.target_duration = 3.0
        self.target_length = int(self.target_duration * self.sample_rate)
        
        self._load_models()
    
    def _load_models(self):
        """Load all 5 pre-trained binary models"""
        print(f"Loading stutter detection models from {self.models_dir}...")
        
        for i, model_file in enumerate(self.model_files):
            model_path = os.path.join(self.models_dir, model_file)
            
            if not os.path.exists(model_path):
                print(f"✗ Model file not found: {model_path}")
                self.models.append(None)
                continue
            
            try:
                model = Model(n_mels=self.n_mels)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.models.append(model)
                label_name = list(self.label_dict.keys())[i]
                print(f"✓ Loaded {label_name} model")
            except Exception as e:
                print(f"✗ Failed to load {model_file}: {e}")
                self.models.append(None)
        
        loaded_count = sum(1 for m in self.models if m is not None)
        print(f"Total models loaded: {loaded_count}/5")
    
    def pad_or_truncate(self, y):
        """Pad or truncate audio to target length"""
        if len(y) < self.target_length:
            # Pad with zeros (silence) at the end
            pad_width = self.target_length - len(y)
            y = np.pad(y, (0, pad_width))
        elif len(y) > self.target_length:
            # Truncate
            y = y[:self.target_length]
        return y
    
    def extract_features(self, audio_file, normalize=True):
        """
        Extract mel spectrogram features from audio file
        
        Args:
            audio_file: Path to audio file
            normalize: Whether to normalize features
            
        Returns:
            Mel spectrogram in dB scale (n_mels, time_frames)
        """
        try:
            y, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
        except Exception as e:
            raise ValueError(f"Invalid audio file: {audio_file} ({e})")
        
        y = self.pad_or_truncate(y)
        
        mels = librosa.feature.melspectrogram(
            y=y, 
            sr=self.sample_rate, 
            n_fft=self.window_size, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
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
    
    def process_audio_file(self, audio_path, callback=None):
        """
        Process audio file and detect stutters using all 5 models
        
        Args:
            audio_path: Path to audio file
            callback: Optional callback function(time_start, time_end, results)
            
        Returns:
            Dictionary with detection results
        """
        if not any(self.models):
            raise RuntimeError("No models loaded. Cannot perform detection.")
        
        print(f"\nProcessing audio file: {audio_path}")
        
        # Extract features once
        try:
            mels_db = self.extract_features(audio_path)
        except Exception as e:
            print(f"Error processing audio: {e}")
            return {}
        
        duration = self.target_duration
        print(f"Analyzing audio (~{duration:.1f}s duration)")
        
        results = {}
        
        # Run all models in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._predict_single_model, mels_db, i): i
                for i in range(5) if self.models[i] is not None
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    label_name, prob, is_detected = result
                    results[label_name] = {
                        'probability': prob,
                        'detected': is_detected
                    }
                    
                    if is_detected:
                        print(f"  ✓ {label_name}: {prob:.3f} - DETECTED")
                    else:
                        print(f"  ○ {label_name}: {prob:.3f}")
        
        # Call callback if provided
        if callback:
            callback(0, duration, results)
        
        return {
            0: {
                'time_start': 0,
                'time_end': duration,
                'detections': results
            }
        }
    
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
        
        # Extract features
        mels = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=self.sample_rate, 
            n_fft=self.window_size, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        mels_db = librosa.power_to_db(mels, ref=np.max)
        
        # Normalize
        mean = np.mean(mels_db)
        std = np.std(mels_db) + 1e-10
        mels_db = (mels_db - mean) / std
        
        # Run predictions
        results = {}
        for i in range(5):
            if self.models[i] is not None:
                label_name, prob, is_detected = self._predict_single_model(mels_db, i)
                results[label_name] = {
                    'probability': prob,
                    'detected': is_detected
                }
        
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
            for label_name, detection in chunk_data['detections'].items():
                if detection['detected']:
                    summary[label_name] += 1
        
        # Calculate percentages
        summary_with_pct = {}
        for label_name, count in summary.items():
            summary_with_pct[label_name] = {
                'count': count,
                'percentage': (count / total_chunks * 100) if total_chunks > 0 else 0
            }
        
        return summary_with_pct
