"""
Connector Module
Handles model inference for stutter detection using pre-trained models
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
    
    def __init__(self, models_dir="Model/models", detection_threshold=0.5):
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
            f'prolongation_model_{self.window_size}_{self.hop_length}_{self.n_mels}_{self.epochs}.pth',
            f'block_model_{self.window_size}_{self.hop_length}_{self.n_mels}_{self.epochs}.pth',
            f'soundrep_model_{self.window_size}_{self.hop_length}_{self.n_mels}_{self.epochs}.pth',
            f'wordrep_model_{self.window_size}_{self.hop_length}_{self.n_mels}_{self.epochs}.pth',
            f'interjection_model_{self.window_size}_{self.hop_length}_{self.n_mels}_{self.epochs}.pth'
        ]
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.target_duration = 3.0
        self.target_length = int(self.target_duration * self.sample_rate)
        
        self._load_models()
    
    def _load_model(self):
        """Load the pre-trained multi-label model"""
        print(f"Loading stutter detection model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # Initialize model with correct architecture
            self.model = CNNLSTM(n_mels=self.n_mels, num_classes=5, hidden_dim=128, num_layers=2)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    
    def _preprocess_audio_chunk(self, waveform, orig_sr):
        """
        Convert audio chunk to mel spectrogram
        
        Args:
            waveform: Audio waveform tensor
            orig_sr: Original sample rate
            
        Returns:
            Mel spectrogram tensor ready for model input
        """
        # Resample if needed
        if orig_sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Generate mel spectrogram
        mel = self.mel_spec(waveform)
        mel_db = self.amplitude_to_db(mel)
        
        # Transpose to [time, n_mels]
        mel_db = mel_db.squeeze(0).transpose(0, 1)
        
        # Pad or truncate to max_len
        if mel_db.shape[0] > self.max_len:
            mel_db = mel_db[:self.max_len, :]
        else:
            pad_len = self.max_len - mel_db.shape[0]
            mel_db = torch.nn.functional.pad(mel_db, (0, 0, 0, pad_len))
        
        return mel_db
    
    def _split_audio_into_chunks(self, audio_path):
        """
        Load audio file using torchaudio with torchcodec backend (matches training)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of (chunk_index, waveform, sample_rate) tuples with single chunk
        """
        try:
            # Use torchaudio with torchcodec backend (same as training)
            waveform, sr = torchaudio.load(audio_path, backend="ffmpeg")
        except Exception as e:
            # Fallback to soundfile if torchaudio fails
            print(f"Warning: torchaudio failed, using soundfile: {e}")
            audio_data, sr = sf.read(audio_path)
            waveform = torch.from_numpy(audio_data.T).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        
        total_duration = waveform.shape[1] / sr
        
        # Return entire audio as a single chunk
        return [(0, waveform, sr, 0, total_duration)]

    
    def _predict_chunk(self, mel_input):
        """
        Run prediction on a single chunk with the multi-label model
        
        Args:
            mel_input: Preprocessed mel spectrogram
            
        Returns:
            Dictionary with predictions for all stutter types
        """
        with torch.no_grad():
            mel_input = mel_input.unsqueeze(0).to(self.device)  # Add batch dimension
            outputs = self.model(mel_input)  # Shape: (1, 5)
            probabilities = torch.sigmoid(outputs).squeeze(0)  # Shape: (5,)
            
            results = {}
            for i, label_name in enumerate(self.label_names):
                prob = probabilities[i].item()
                is_detected = prob > self.detection_threshold
                results[label_name] = {
                    'probability': prob,
                    'detected': is_detected
                }
            
        return results

    
    def process_audio_file(self, audio_path, callback=None):
        """
        Process entire audio file and detect stutters using multi-label model
        
        Args:
            audio_path: Path to audio file
            callback: Optional callback function(chunk_idx, time_start, time_end, results)
            
        Returns:
            Dictionary with detection results for each chunk
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Cannot perform detection.")
        
        print(f"\nProcessing audio file: {audio_path}")
        chunks = self._split_audio_into_chunks(audio_path)
        print(f"Processing entire audio as single chunk (~{chunks[0][4]:.1f}s duration)")
        
        all_results = {}
        
        for chunk_idx, waveform, sr, time_start, time_end in chunks:
            print(f"\nAnalyzing audio ({time_start:.1f}s - {time_end:.1f}s)")
            
            # Preprocess the chunk
            mel_input = self._preprocess_audio_chunk(waveform, sr)
            
            # Get predictions for all stutter types
            chunk_results = self._predict_chunk(mel_input)
            
            # Print results
            for label_name, result in chunk_results.items():
                prob = result['probability']
                detected = result['detected']
                if detected:
                    print(f"  ✓ {label_name}: {prob:.3f} - DETECTED")
                else:
                    print(f"  ○ {label_name}: {prob:.3f}")
            
            # Store results for this chunk
            all_results[chunk_idx] = {
                'time_start': time_start,
                'time_end': time_end,
                'detections': chunk_results
            }
            
            # Call callback if provided
            if callback:
                callback(chunk_idx, time_start, time_end, chunk_results)
        
        return all_results

    
    def process_audio_chunk_realtime(self, waveform, sr):
        """
        Process a single audio chunk in real-time
        
        Args:
            waveform: Audio waveform tensor
            sr: Sample rate
            
        Returns:
            Dictionary with detection results for all stutter types
        """
        mel_input = self._preprocess_audio_chunk(waveform, sr)
        results = self._predict_chunk(mel_input)
        return results
    
    def get_summary(self, results):
        """
        Get summary statistics from detection results
        
        Args:
            results: Results dictionary from process_audio_file
            
        Returns:
            Summary dictionary with counts and percentages
        """
        summary = {label_name: 0 for label_name in self.label_names}
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
