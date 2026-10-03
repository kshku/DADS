"""Shared configuration for DADS model training and evaluation.

Single source of truth for the stutter class registry, the SEP-28k label
mapping and the production model specifications. The CNN architecture itself
is imported from ``shared.connector`` so training and inference can never
drift apart.
"""

import os

# --- Audio -----------------------------------------------------------------
SAMPLE_RATE = 16000
TARGET_DURATION = 3.0
TARGET_LENGTH = int(TARGET_DURATION * SAMPLE_RATE)

# --- Classes ---------------------------------------------------------------
# Order matches StutterDetector.label_dict / model slot order in shared/connector.py
STUTTER_CLASSES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]

# SEP-28k label column backing each class
LABEL_COLUMNS = {
    "prolongation": "Prolongation",
    "block": "Block",
    "soundrep": "SoundRep",
    "wordrep": "WordRep",
    "interjection": "Interjection",
}

# Clips carrying these flags are dropped, matching Model/model_train.ipynb
EXCLUDE_FLAGS = ["Music", "NoSpeech"]

# Additional exclusions, off by default so results stay comparable to accuracy.txt
OPTIONAL_EXCLUDE_FLAGS = ["PoorAudioQuality", "Unsure"]

# --- Production models -----------------------------------------------------
# Filename convention: {type}_model_{n_fft}_{hop_length}_{n_mels}_{epochs}.pth
# Values mirror Model/models/copy/ so retraining reproduces the shipped models.
PRODUCTION_MODELS = {
    "prolongation": {"n_fft": 1024, "hop_length": 512, "n_mels": 128, "epochs": 40},
    "block": {"n_fft": 1024, "hop_length": 512, "n_mels": 128, "epochs": 40},
    "soundrep": {"n_fft": 1024, "hop_length": 512, "n_mels": 256, "epochs": 40},
    "wordrep": {"n_fft": 1024, "hop_length": 512, "n_mels": 64, "epochs": 40},
    "interjection": {"n_fft": 1024, "hop_length": 512, "n_mels": 128, "epochs": 40},
}

# Default feature params for a class not present in PRODUCTION_MODELS
DEFAULT_FEATURES = {"n_fft": 1024, "hop_length": 512, "n_mels": 128, "epochs": 40}

# --- Thresholds ------------------------------------------------------------
# Production operating point used by backend/services/detector.py and App/
DEFAULT_THRESHOLD = 0.4

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def portable_path(path: str) -> str:
    """Render a filesystem path so generated reports stay machine-independent.

    Paths inside the project become repo-relative. Anything outside is reduced
    to its final directory name, because datasets commonly live on a user's
    personal machine and an absolute path in a committed report is neither
    reproducible for others nor meaningful once written down.
    """
    if not path:
        return path
    absolute = os.path.abspath(path)
    root = os.path.abspath(PROJECT_ROOT)
    if absolute == root or absolute.startswith(root + os.sep):
        return os.path.relpath(absolute, root)
    parent = os.path.basename(os.path.dirname(absolute))
    name = os.path.basename(absolute)
    return os.path.join("<external>", parent, name) if parent else os.path.join("<external>", name)


DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
LABELS_CSV = os.path.join(DATASET_DIR, "SEP-28k_labels.csv")
WAVS_DIR = os.path.join(DATASET_DIR, "Waves")

# SEP-28k can be consumed two ways: sliced from the full episode WAVs in
# dataset/Waves/ using the CSV's Start/Stop sample offsets, or read from a
# directory of already-cut <Show>_<EpId>_<ClipId>.wav clips. The latter covers
# every show (dataset/Waves currently holds only 70 of 385 episodes), so
# --clips_dir is accepted by ml/train.py and ml/evaluate.py. It is deliberately
# not defaulted here, since the location depends on where SEP-28k happens to be
# on each machine; pass it on the command line (see README).

MODELS_DIR = os.path.join(PROJECT_ROOT, "Model", "models")
PRODUCTION_MODELS_DIR = os.path.join(MODELS_DIR, "copy")
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained")

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports")


def feature_params(class_name: str) -> dict:
    """Return the production feature params (n_fft/hop_length/n_mels/epochs) for a class."""
    return dict(PRODUCTION_MODELS.get(class_name, DEFAULT_FEATURES))


def production_model_path(class_name: str, models_dir: str = None) -> str:
    """Absolute path of the shipped .pth file for a class."""
    params = feature_params(class_name)
    base = models_dir or PRODUCTION_MODELS_DIR
    filename = f"{class_name}_model_{params['n_fft']}_{params['hop_length']}_{params['n_mels']}_{params['epochs']}.pth"
    return os.path.join(base, filename)
