#!/usr/bin/env python3
"""Dataset setup for DADS.

Downloads audio from SEP-28k and extracts 3-second clips for model training.
Replaces the broken download_audio.py and extract_clips.py from the submodule.
Run via: ./setup_dataset.sh (preferred) or python3 setup_dataset.py

Requires: ffmpeg, numpy, pandas, scipy
"""

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

EPISODES_CSV = os.path.join(DATASET_DIR, "SEP-28k_episodes.csv")
LABELS_CSV = os.path.join(DATASET_DIR, "SEP-28k_labels.csv")

WAVS_DIR = os.path.join(DATASET_DIR, "Waves")
CLIPS_DIR = os.path.join(DATASET_DIR, "Clips")

MAX_WORKERS = 8
SAMPLE_RATE = 16000
AUDIO_EXTENSIONS = (".mp3", ".m4a", ".mp4")


def get_workers():
    """Auto-detect worker count, capped at MAX_WORKERS."""
    nproc = multiprocessing.cpu_count() or 4
    return min(nproc, MAX_WORKERS)


# ---------------------------------------------------------------------------
# Step 1: Download — fetch episodes, convert to 16kHz mono WAV
# ---------------------------------------------------------------------------


def _download_one(args):
    """Download a single episode: fetch URL, convert to 16kHz mono WAV."""
    url, show, ep_idx = args
    ext = ""
    for e in AUDIO_EXTENSIONS:
        if e in url:
            ext = e
            break

    show_dir = os.path.join(WAVS_DIR, show)
    os.makedirs(show_dir, exist_ok=True)

    wav_path = os.path.join(show_dir, f"{ep_idx}.wav")
    if os.path.exists(wav_path):
        return True

    audio_path = os.path.join(show_dir, f"{ep_idx}{ext}")
    try:
        subprocess.run(["wget", "-q", "-O", audio_path, url], check=True, timeout=120)
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", wav_path],
            check=True,
            timeout=120,
            capture_output=True,
        )
        os.remove(audio_path)
        return True
    except Exception as e:
        print(f"  WARN: Failed {show}/{ep_idx}: {e}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return False


def step_download(workers):
    """Download raw audio files and convert to 16kHz mono WAV."""
    if not os.path.exists(EPISODES_CSV):
        print(f"ERROR: {EPISODES_CSV} not found. Is the submodule initialized?")
        sys.exit(1)

    df = pd.read_csv(EPISODES_CSV, header=None, names=["show", "episode", "url", "show_id", "ep_idx"])
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    print(f"  Found {len(df)} episodes")

    tasks = [(row.url, row.show_id, int(row.ep_idx)) for _, row in df.iterrows()]

    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="  Downloading", unit="file") as pbar:
            for future in as_completed(futures):
                if not future.result():
                    failed += 1
                pbar.update(1)
    if failed:
        print(f"  {failed} downloads failed")
    print("  Download complete")


# ---------------------------------------------------------------------------
# Step 2: Extract clips — slice 3-second segments from downloaded WAVs
# ---------------------------------------------------------------------------


def step_extract():
    """Extract clips from downloaded WAV files using labels CSV."""
    if not os.path.exists(LABELS_CSV):
        print(f"ERROR: {LABELS_CSV} not found.")
        sys.exit(1)

    data = pd.read_csv(LABELS_CSV, dtype={"EpId": str})
    print(f"  Found {len(data)} clips in SEP-28k_labels.csv")

    loaded_wav = ""
    audio = None

    with tqdm(total=len(data), desc="  Extracting", unit="clip") as pbar:
        for _, row in data.iterrows():
            show = row.Show
            episode = row.EpId.strip()
            clip_idx = row.ClipId
            start = row.Start
            stop = row.Stop

            wav_path = os.path.join(WAVS_DIR, show, f"{episode}.wav")
            clip_dir = os.path.join(CLIPS_DIR, show, episode)
            clip_path = os.path.join(clip_dir, f"{show}_{episode}_{clip_idx}.wav")

            if not os.path.exists(wav_path):
                pbar.update(1)
                continue

            if wav_path != loaded_wav:
                sr, audio = wavfile.read(wav_path)
                assert sr == SAMPLE_RATE, f"Expected 16kHz, got {sr}Hz for {wav_path}"
                loaded_wav = wav_path

            os.makedirs(clip_dir, exist_ok=True)

            clip = audio[start:stop]
            wavfile.write(clip_path, sr, clip)
            pbar.update(1)

    print("  Extraction complete")


# ---------------------------------------------------------------------------
# Step 3: Cleanup
# ---------------------------------------------------------------------------


def step_cleanup():
    """Remove temporary Waves directory after extraction."""
    if os.path.exists(WAVS_DIR):
        print(f"  Cleaning up {WAVS_DIR}/ ...")
        shutil.rmtree(WAVS_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        print("Delete Clips/ first to re-run.")
        return

    start = time.time()

    # Step 1: Download
    if not args.skip_download:
        print("\n[1/3] Downloading audio...")
        step_download(workers)
    else:
        print("\n[1/3] Skipping download")

    # Step 2: Extract clips
    if not args.skip_extract:
        print("\n[2/3] Extracting clips...")
        step_extract()
    else:
        print("\n[2/3] Skipping extract")

    # Step 3: Cleanup
    if not args.skip_cleanup:
        print("\n[3/3] Cleaning up...")
        step_cleanup()
    else:
        print("\n[3/3] Skipping cleanup")

    elapsed = time.time() - start
    print(f"\nDone! ({elapsed:.1f}s)")
    print(f"Clips available at: {CLIPS_DIR}/")


if __name__ == "__main__":
    main()
