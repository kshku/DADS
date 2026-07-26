#!/usr/bin/env python3
"""Dataset setup for DADS.

Downloads audio from SEP-28k and extracts clips for model training.
Run via: ./setup_dataset.sh (preferred) or python3 setup_dataset.py

Requires: ffmpeg, numpy, pandas, scipy
"""

import argparse
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

EPISODES_CSV = os.path.join(DATASET_DIR, "SEP-28k_episodes.csv")
LABELS_CSV = os.path.join(DATASET_DIR, "SEP-28k_labels.csv")
DOWNLOAD_SCRIPT = os.path.join(DATASET_DIR, "download_audio.py")
EXTRACT_SCRIPT = os.path.join(DATASET_DIR, "extract_clips.py")

WAVS_DIR = os.path.join(DATASET_DIR, "Waves")
CLIPS_DIR = os.path.join(DATASET_DIR, "Clips")

MAX_WORKERS = 8


def get_workers():
    """Auto-detect worker count, capped at MAX_WORKERS."""
    nproc = multiprocessing.cpu_count() or 4
    return min(nproc, MAX_WORKERS)


def step_download(workers):
    """Download raw audio files and convert to 16kHz mono WAV."""
    if not os.path.exists(DOWNLOAD_SCRIPT):
        print(f"ERROR: {DOWNLOAD_SCRIPT} not found. Is the submodule initialized?")
        sys.exit(1)

    print(f"  Downloading audio files to {WAVS_DIR}/ ...")
    cmd = [
        sys.executable,
        DOWNLOAD_SCRIPT,
        "--episodes",
        EPISODES_CSV,
        "--wavs",
        WAVS_DIR,
    ]
    subprocess.run(cmd, check=True)


def step_extract(workers):
    """Extract 3-second clips from downloaded WAV files."""
    if not os.path.exists(EXTRACT_SCRIPT):
        print(f"ERROR: {EXTRACT_SCRIPT} not found. Is the submodule initialized?")
        sys.exit(1)

    print(f"  Extracting clips to {CLIPS_DIR}/ ...")
    cmd = [
        sys.executable,
        EXTRACT_SCRIPT,
        "--labels",
        LABELS_CSV,
        "--wavs",
        WAVS_DIR,
        "--clips",
        CLIPS_DIR,
        "--progress",
    ]
    subprocess.run(cmd, check=True)


def step_cleanup():
    """Remove temporary Waves directory after extraction."""
    if os.path.exists(WAVS_DIR):
        print(f"  Cleaning up {WAVS_DIR}/ ...")
        shutil.rmtree(WAVS_DIR)


def main():
    parser = argparse.ArgumentParser(description="Set up DADS dataset")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step (if Waves/ already exists)")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extract step (if Clips/ already exists)")
    parser.add_argument("--skip-cleanup", action="store_true", help="Keep Waves/ after extraction")
    args = parser.parse_args()

    workers = get_workers()
    print(f"Workers: {workers}")

    # Skip if Clips/ already exists
    if os.path.exists(CLIPS_DIR) and os.listdir(CLIPS_DIR):
        print(f"\nClips directory already exists at {CLIPS_DIR}/")
        print("Use --skip-download --skip-extract to force re-download, or delete Clips/ first.")
        return

    start = time.time()

    # Step 1: Download
    if not args.skip_download:
        print("\n[1/3] Downloading audio...")
        step_download(workers)
    else:
        print("\n[1/3] Skipping download (--skip-download)")

    # Step 2: Extract clips
    if not args.skip_extract:
        print("\n[2/3] Extracting clips...")
        step_extract(workers)
    else:
        print("\n[2/3] Skipping extract (--skip-extract)")

    # Step 3: Cleanup
    if not args.skip_cleanup:
        print("\n[3/3] Cleaning up...")
        step_cleanup()
    else:
        print("\n[3/3] Skipping cleanup (--skip-cleanup)")

    elapsed = time.time() - start
    print(f"\nDone! ({elapsed:.1f}s)")
    print(f"Clips available at: {CLIPS_DIR}/")


if __name__ == "__main__":
    main()
