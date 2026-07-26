# DADS — Detection and Analysis of Dysfluencies in Speech

AI-powered stutter detection system. Analyzes speech audio across 5 stutter types using separate CNN models trained on the [SEP28k dataset](https://www.kaggle.com/datasets/ikrbasak/sep-28k), and visualizes results with spectrogram playback.

## Stutter Types

| Type | Description |
|------|-------------|
| **Prolongation** | Sound stretched beyond normal length (e.g., "sss-snake") |
| **Block** | Airflow stops mid-utterance (silent pause with visible effort) |
| **Sound Repetition** | Repeating a single sound (e.g., "b-b-ball") |
| **Word Repetition** | Repeating whole words (e.g., "I-I-I want") |
| **Interjection** | Filler sounds/words (e.g., "um", "uh", "like") |

## Quick Start

### Prerequisites

- Python 3.12
- ffmpeg (for dataset setup)
- CUDA-capable GPU (recommended)

### Installation

```bash
git clone git@github.com:kshku/DADS.git
cd DADS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset Setup

One command downloads audio and extracts clips:

```bash
./setup_dataset.sh
```

This initializes the git submodule, downloads audio via ffmpeg, and extracts 3-second clips. Skip with `--skip-download` / `--skip-extract` flags.

### Run — PyQt5 Desktop App

```bash
python App/run_app.py
```

Full desktop application with recording, playback, PDF passage viewer, and analysis visualization.

### Run — FastAPI Web App

```bash
uvicorn backend.main:app --reload
```

Opens at `http://localhost:8000`. Upload an audio file, hit Analyze, view spectrogram with real-time detection results, and download the report.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    shared/connector.py                        │
│                  StutterDetector + Model                      │
│                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────────┐ │
│  │  PyQt5 App    │  │  FastAPI Web  │  │  Training        │ │
│  │  (App/)       │  │  (backend/)   │  │  Notebooks       │ │
│  └───────┬───────┘  └───────┬───────┘  └──────────────────┘ │
└──────────┼──────────────────┼────────────────────────────────┘
           │                  │
           ▼                  ▼
     Model/models/copy/   Model/models/copy/
     (5 .pth weights)     (5 .pth weights)
```

### Components

| Module | File | Responsibility |
|--------|------|----------------|
| Entry point | `App/run_app.py` | Launches QApplication, creates MainWindow |
| Main window | `App/main_window.py` | Two-page layout (main → analysis), coordinates widgets |
| Audio handler | `App/audio_handler.py` | Recording via QAudioInput, playback via QAudioOutput |
| PDF viewer | `App/pdf_viewer_widget.py` | Renders reference passages for reading during recording |
| Analysis widget | `App/analysis_widget.py` | Spectrogram/waveform plots, stutter results panel, report export |
| Plot canvas | `App/plot_canvas.py` | Matplotlib-based spectrogram and waveform rendering |
| **Detector** | `shared/connector.py` | Model loading, mel spectrogram extraction, inference engine |
| Web backend | `backend/main.py` | FastAPI app, single-page analyzer |
| Web detector | `backend/services/detector.py` | Singleton wrapper around StutterDetector |

### Data Flow

```
Audio Input (WAV file / recording)
    │
    ▼
pad_or_truncate()          → Normalize to 3s (48000 samples @ 16kHz)
    │
    ▼
Split into chunks          → 3-second windows, non-overlapping
    │
    ▼
For each chunk:
    │
    ├─► Model 0 (prolongation)  ──┐
    ├─► Model 1 (block)          │
    ├─► Model 2 (soundrep)       ├── ThreadPoolExecutor (5 workers)
    ├─► Model 3 (wordrep)        │
    └─► Model 4 (interjection)  ──┘
         │
         ▼
    extract_features()
         │
         ├─ librosa.feature.melspectrogram(n_fft, hop_length, n_mels)
         ├─ librosa.power_to_db()
         └─ z-score normalization
         │
         ▼
    Model forward pass       → sigmoid → probability (0-1)
         │
         ▼
    Threshold (0.4)          → detected / not detected
```

## Model Architecture

5 independent binary CNN models, one per stutter type.

```
Input: mel spectrogram (1, n_mels, time_frames)
    │
    ▼
Conv2d(1→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    │
    ▼
Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    │
    ▼
Conv2d(64→128, 3×3) → BatchNorm → ReLU → AdaptiveAvgPool(1×1)
    │
    ▼
Flatten → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1)
    │
    ▼
Sigmoid → P(stutter)
```

### Production Models

Filename convention: `{type}_model_{n_fft}_{hop_length}_{n_mels}_{epochs}.pth`

| Model | File | n_mels | Accuracy |
|-------|------|--------|----------|
| Prolongation | `prolongation_model_1024_512_128_40.pth` | 128 | 0.76 |
| Block | `block_model_1024_512_128_40.pth` | 128 | 0.68 |
| Sound Repetition | `soundrep_model_1024_512_256_40.pth` | 256 | 0.82 |
| Word Repetition | `wordrep_model_1024_512_64_40.pth` | 64 | 0.81 |
| Interjection | `interjection_model_1024_512_128_40.pth` | 128 | 0.71 |

Training parameters (`n_fft`, `hop_length`, `n_mels`, `epochs`) are parsed from filenames at runtime.

## Project Structure

```
DADS/
├── App/                        # PyQt5 desktop application
│   ├── run_app.py              # Entry point
│   ├── main_window.py          # Main window, page navigation
│   ├── audio_handler.py        # Audio recording and playback
│   ├── analysis_widget.py      # Spectrogram, waveform, stutter panel
│   ├── pdf_viewer_widget.py    # PDF passage viewer
│   ├── plot_canvas.py          # Matplotlib canvas
│   └── Passages/               # Reference passages (Rainbow Passage)
├── backend/                    # FastAPI web application
│   ├── main.py                 # FastAPI app entry point
│   ├── services/detector.py    # StutterDetector singleton wrapper
│   ├── routers/analysis.py     # POST /api/analyze (SSE streaming)
│   ├── templates/index.html    # Single-page analyzer UI
│   └── static/                 # CSS + JS
│       ├── css/style.css       # Dark theme
│       └── js/
│           ├── app.js          # Upload, SSE, results, report download
│           └── player.js       # wavesurfer.js + spectrogram
├── shared/                     # Shared inference code
│   └── connector.py            # StutterDetector + Model class
├── Model/
│   ├── models/copy/            # 5 production .pth models + accuracy.txt
│   ├── models/                 # All trained model variants
│   ├── model.ipynb             # CNN architecture + training
│   ├── model_train.ipynb       # CNNLSTM training
│   └── inference.ipynb         # CNNLSTM inference
├── dataset/                    # Git submodule → SEP28k dataset
├── setup_dataset.sh            # Dataset setup (shell wrapper)
├── setup_dataset.py            # Dataset setup (Python script)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker build for web backend
└── pyproject.toml              # Ruff linting config
```

## Development

### Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production — protected, requires PR |
| `dev` | Development — integration branch |
| `feature/*` | New features → PR to `dev` |
| `bugfix/*` | Bug fixes → PR to `dev` |
| `hotfix/*` | Urgent production fixes → PR to `main` |

### Commit Convention

[Conventional Commits](https://www.conventionalcommits.org/): `<type>: <description>`

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`, `perf`

### Linting

```bash
ruff check .
ruff format --check .
```

### CI

GitHub Actions runs on every PR:
- **Lint:** `ruff check` + `ruff format --check`
