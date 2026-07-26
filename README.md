# DADS — Detection and Analysis of Dysfluencies in Speech

AI-powered stutter detection system. Records or uploads speech audio, analyzes it across 5 stutter types using separate CNN models, and visualizes results with waveform/spectrogram playback.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        PyQt5 Desktop App                     │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  PDF Viewer  │  │ Audio Handler│  │  Analysis Widget   │  │
│  │  (passages)  │  │ (recording/  │  │  (spectrogram,     │  │
│  │              │  │  playback)   │  │   waveform,        │  │
│  │              │  │              │  │   stutter panel)    │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘  │
│         │                 │                    │              │
│         │                 │          ┌─────────▼──────────┐   │
│         │                 │          │   StutterDetector   │   │
│         │                 │          │   (connector.py)    │   │
│         │                 │          │                     │   │
│         │                 │          │  ┌───────────────┐  │   │
│         │                 │          │  │  5 CNN Models  │  │   │
│         │                 │          │  │  (per-type)    │  │   │
│         │                 │          │  └───────────────┘  │   │
│         │                 │          └─────────────────────┘   │
└─────────┼─────────────────┼────────────────────┼──────────────┘
          │                 │                    │
          ▼                 ▼                    ▼
    App/Passages/     WAV files          Model/models/copy/
    (reference        (16kHz, mono)      (5 .pth weights)
     passages)
```

### Components

| Module | File | Responsibility |
|--------|------|----------------|
| Entry point | `App/run_app.py` | Launches QApplication, creates MainWindow |
| Main window | `App/main_window.py` | Two-page layout (main → analysis), coordinates widgets |
| Audio handler | `App/audio_handler.py` | Recording via QAudioInput, playback via QAudioOutput, WAV I/O |
| PDF viewer | `App/pdf_viewer_widget.py` | Renders reference passages for reading during recording |
| Analysis widget | `App/analysis_widget.py` | Spectrogram/waveform plots, audio playback controls, stutter results panel, report export |
| Plot canvas | `App/plot_canvas.py` | Matplotlib-based spectrogram and waveform rendering |
| Detector | `App/connector.py` | Model loading, audio preprocessing, mel spectrogram extraction, inference engine |

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
         ├─ pad_or_truncate(audio)
         ├─ librosa.feature.melspectrogram(n_fft, hop_length, n_mels)
         ├─ librosa.power_to_db()
         └─ z-score normalization
         │
         ▼
    Model forward pass       → sigmoid → probability (0-1)
         │
         ▼
    Threshold (0.4)          → detected / not detected
         │
         ▼
    Aggregate across chunks  → max probability per stutter type
```

## Model Architecture

5 independent binary CNN models, one per stutter type. Each trained on the [SEP28k dataset](https://www.kaggle.com/datasets/ikrbasak/sep-28k).

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

Each model's training parameters (`n_fft`, `hop_length`, `n_mels`, `epochs`) are encoded in the filename and parsed at runtime by `StutterDetector._parse_model_params()`.

## Feature Extraction

- **Sample rate:** 16kHz, mono
- **Chunk size:** 3 seconds (48000 samples)
- **Mel spectrogram:** per-model `n_mels` (64/128/256), `n_fft=1024`, `hop_length=512`
- **Scale:** dB (power_to_db with max reference)
- **Normalization:** z-score (mean=0, std=1)

## Project Structure

```
DADS/
├── App/                        # PyQt5 desktop application
│   ├── run_app.py              # Entry point
│   ├── main_window.py          # Main window, page navigation
│   ├── audio_handler.py        # Audio recording (QAudioInput) and playback (QAudioOutput)
│   ├── connector.py            # StutterDetector + Model class (inference engine)
│   ├── analysis_widget.py      # Analysis page: plots, playback controls, stutter panel
│   ├── pdf_viewer_widget.py    # PDF passage viewer
│   ├── plot_canvas.py          # Matplotlib canvas for spectrogram/waveform
│   └── Passages/               # Reference passages (Rainbow Passage PDF)
├── Model/
│   ├── models/copy/            # 5 production .pth models + accuracy.txt
│   ├── models/                 # All trained models (52+ variants)
│   ├── model.ipynb             # Model architecture + training code
│   ├── model_train.ipynb       # CNNLSTM training notebook
│   └── inference.ipynb         # CNNLSTM inference notebook
├── dataset/                    # Git submodule → SEP28k dataset
├── requirements.txt
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.10
- CUDA-capable GPU (recommended for inference speed)

### Installation

```bash
git clone git@github.com:kshku/DADS.git
cd DADS

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Dataset

The dataset is a git submodule. Initialize after cloning:

```bash
git submodule update --init --recursive
```

Audio data (Clips/, Waves/) must be copied manually into `dataset/`.

### Run

```bash
python App/run_app.py
```

## Development

### Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production — protected, requires PR + review |
| `feature/*` | New features (e.g., `feature/backend-api`) |
| `bugfix/*` | Bug fixes |
| `hotfix/*` | Urgent production fixes |
| `docs/*` | Documentation changes |

### Commit Convention

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>
```

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting (no code change) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Build process, dependencies, tooling |
| `ci` | CI/CD configuration |
| `perf` | Performance improvement |

**Examples:**

```
feat: add audio upload endpoint
fix: correct n_mels parsing for wordrep model
docs: update README with backend architecture
chore: add Dockerfile for deployment
```

### Linting

```bash
ruff check .
ruff format --check .
```

### CI

GitHub Actions runs on every PR to `main`:
- **Lint:** `ruff check` + `ruff format --check`
