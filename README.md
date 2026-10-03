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

Opens at `http://localhost:8000`. Upload an audio file (WAV, MP3, WebM, OGG, FLAC), hit Analyze, and view real-time detection results via SSE streaming with a server-side generated spectrogram and waveform playback.

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
| Web analysis | `backend/routers/analysis.py` | POST /api/analyze — SSE streaming + server-side spectrogram generation |

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

### Evaluation and Training (`ml/`)

`ml/` scores the five models on SEP-28k and retrains them with a clean split. Feature
extraction mirrors `shared/connector.py` exactly (mel → `power_to_db(ref=max)` →
per-clip z-score), so checkpoints are drop-in compatible with production inference.

```bash
# Score the shipped production weights
python -m ml.evaluate --class all --split both

# Retrain all five models on a speaker-disjoint split
python -m ml.train --class all --split group --epochs 40
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--class` | `all` | `all`, or one of `prolongation` `block` `soundrep` `wordrep` `interjection` |
| `--split` | `group` (`both` for evaluate) | `random`, `group`, or `both` |
| `--group_by` | `episode` | Group key for `group` splits: `episode` or `show` |
| `--epochs` / `--batch_size` / `--lr` | `40` / `32` / `1e-3` | Match production settings |
| `--loss_type` | `bce` | `bce`, `bce_balanced`, or `focal` |
| `--seed` | `42` | Seeds weight init and shuffling |
| `--threshold` | `0.4` | Operating point; only used for the metrics tables. Live detection reads `Model/registry.json` |
| `--subset` | `test` | Which split to score: `train`, `val`, or `test` |
| `--clips_dir` | `None` | Directory of pre-cut `<Show>_<EpId>_<ClipId>.wav` clips |
| `--cache_dir` | `None` | Persist extracted mel features to reuse across runs |
| `--models_dir` | `Model/models/copy` | Which checkpoints to score |
| `--output_dir` | `reports/` | Where JSON reports and `SUMMARY.md` land |

**Audio source.** By default clips are sliced on the fly from `dataset/Waves/`, the same
operation `setup_dataset.py` performs. That requires the full episode WAVs, which are only
partially downloaded, and it limits evaluation to whichever shows happen to be present. If you
have SEP-28k as pre-cut clips, pass `--clips_dir` instead:

```bash
python -m ml.evaluate --clips_dir /path/to/clips --class all --split both
```

Both layouts were verified to produce features within 0.013 z-units of each other. Header-only
(44-byte) WAVs are skipped, since they decode to zero samples.

**Splits.** `random` is a multi-label stratified random split over clips; it reproduces
`Model/model_train.ipynb` and is comparable to the accuracy figures above, but adjacent clips
come from the same recording, so speaker identity leaks between splits. `group` keeps every
episode — or every show, with `--group_by show` — in exactly one subset. Use `group` for any
number you intend to quote.

**Outputs.** `reports/<split>/<class>_report.json` (metrics at the top level), a confusion-matrix
PNG per class, `multilabel_report.json` for the joint view (exact-match accuracy, Hamming loss,
macro F1/AUROC), plus `reports/summary.json` and `reports/SUMMARY.md` covering both splits side
by side. Retrained checkpoints go to `Model/models/trained/` under a fingerprint filename that
encodes every hyperparameter, so `ml/evaluate.py --models_dir Model/models/trained` picks them up.
Absolute paths are rewritten to project-relative or `<external>/...` form before being written, so
committed reports stay portable.

### Thresholds and the model registry

`Model/registry.json` is the single source of truth for **which checkpoints run** and **what
probability each class is cut at**. A threshold is only meaningful against the weights it was
fitted on, so each entry pins its own `models_dir`, `model_files`, `thresholds`, and a
`calibration` block recording how they were produced. Two entries ship today:

| Entry | Checkpoints | Notes |
|-------|-------------|-------|
| `production` | `Model/models/copy/` | Shipped weights (default) |
| `retrained` | `Model/models/trained/` | Retrained on an episode-disjoint split |

`defaults.detector` picks the active entry; select another with
`StutterDetector(detector="retrained")`.

```bash
# Fit thresholds on the validation split and write them into the registry
python -m ml.calibrate --class all --split group --models_dir Model/models/copy \
    --register production

python -m ml.calibrate --class all --split group --models_dir Model/models/trained \
    --register retrained --output_dir reports/trained

# Inspect without writing
python -m ml.calibrate --class all --models_dir Model/models/copy --dry_run
```

`ml.calibrate` selects each threshold by maximising F1 on **validation only**, then reports what
that choice scores on the held-out test split. Test scores are diagnostic — they are never used to
pick a threshold, so they stay an honest estimate of generalization. Pass `--dry_run` to print
results without touching the registry.

The production weights were fitted on an unseeded random ~70% of every clip, so their test
threshold and metrics overlap training data; the `retrained` entry exists because its validation
split is genuinely held out.

Calibrating is worth it. On the episode-disjoint split, the single shipped threshold of `0.4`
gives the production models a macro F1 of `0.369`; their individually calibrated thresholds lift
that to `0.523` on the same clips.

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
│           └── player.js       # wavesurfer.js waveform player
├── ml/                         # Training, evaluation, calibration pipeline
│   ├── evaluate.py             # Score checkpoints, write reports
│   ├── train.py                # Retrain on a clean split
│   ├── calibrate.py            # Fit thresholds on validation
│   └── registry.py             # Model/checkpoint/threshold registry
├── shared/                     # Shared inference code
│   └── connector.py            # StutterDetector + Model class
├── Model/
│   ├── registry.json           # Active detector: checkpoints + per-class thresholds
│   ├── models/copy/            # 5 production .pth models + accuracy.txt
│   ├── models/trained/         # Retrained .pth models (fingerprint filenames)
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
