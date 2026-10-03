"""SEP-28k data loading and splitting for DADS training and evaluation.

Audio comes from one of two layouts:

* episode WAVs (``dataset/Waves/<show_id>/<episode>.wav``), sliced on the fly
  with the ``Start``/``Stop`` sample offsets in ``SEP-28k_labels.csv`` — the same
  operation ``setup_dataset.py`` performs, so this path needs no pre-extracted
  ``dataset/Clips`` directory;
* a directory of already-cut ``<Show>_<EpId>_<ClipId>.wav`` clips, passed as
  ``clips_dir``/``--clips_dir``. Faster, and the only option when the full
  episodes have not been downloaded. Both layouts were verified to produce
  features within 0.013 z-units of each other.

Two split strategies are provided:

``random``
    Multi-label stratified random split over clips. Reproduces the methodology
    of ``Model/model_train.ipynb`` and is comparable to ``accuracy.txt``, but
    clips are adjacent slices of the same recording, so speaker identity leaks
    between splits and the numbers are optimistic.

``group``
    Episode- (or show-) disjoint split. No recording appears in more than one
    subset, so the score reflects generalisation to unseen speakers.
"""

import hashlib
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset

from ml.config import (
    EXCLUDE_FLAGS,
    LABEL_COLUMNS,
    LABELS_CSV,
    SAMPLE_RATE,
    STUTTER_CLASSES,
    TARGET_LENGTH,
    WAVS_DIR,
)

# A canonical WAV header is 44 bytes; anything at or below it holds no audio.
WAV_HEADER_BYTES = 44

# Above this many groups, bucket filling falls back to a greedy pass: an exact
# subset search is only tractable while 2**n stays small.
EXACT_GROUP_SEARCH_LIMIT = 16


def _subsets_within_budget(order, sizes, budget):
    """Yield every subset of ``order`` whose clip total does not exceed ``budget``.

    Greedy filling overshoots badly when groups are large relative to the budget
    — a single show is a third of SEP-28k — so small group sets are searched
    exhaustively for the subset closest to the budget instead.
    """
    limit = max(1, int(budget))
    for mask in range(1, 1 << len(order)):
        chosen = [g for i, g in enumerate(order) if mask & (1 << i)]
        if sum(sizes[g] for g in chosen) <= limit:
            yield chosen


@dataclass
class Split:
    """Positional indices into a clip table for one split strategy."""

    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    group_by: Optional[str] = None
    meta: Dict = field(default_factory=dict)

    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "group_by": self.group_by,
            "sizes": self.sizes(),
            **self.meta,
        }


# ---------------------------------------------------------------------------
# Clip table
# ---------------------------------------------------------------------------


def load_clip_table(
    labels_csv: str = LABELS_CSV,
    waves_dir: str = WAVS_DIR,
    exclude_flags: Optional[List[str]] = None,
    extra_exclude_flags: Optional[List[str]] = None,
    require_audio: bool = True,
    clips_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load SEP-28k labels as a clip table with binary labels and resolved paths.

    SEP-28k ships as full episode WAVs that must be sliced with the CSV's
    Start/Stop sample offsets. ``clips_dir`` points at an already-cut mirror
    (one ``<Show>_<EpId>_<ClipId>.wav`` per row) and takes priority when set,
    since DADS's own ``dataset/Waves/`` only has 70 of 385 episodes downloaded.

    Args:
        labels_csv: Path to SEP-28k_labels.csv.
        waves_dir: Directory holding ``<show_id>/<episode>.wav``.
        exclude_flags: Columns that must be 0 for a clip to be kept. Defaults to
            ``config.EXCLUDE_FLAGS`` (Music, NoSpeech), matching the training
            notebook.
        extra_exclude_flags: Optional additional columns to drop, e.g.
            ``["PoorAudioQuality"]``.
        require_audio: Drop clips whose audio is missing or empty.
        clips_dir: Directory of pre-cut clip WAVs. Defaults to
            ``config.SEP28K_CLIPS_DIR`` when it exists, else falls back to
            slicing ``waves_dir``.

    Returns:
        DataFrame with one row per clip, sorted by (show_id, episode, start) so
        sequential iteration hits the per-worker WAV cache. Columns include
        ``show_id``, ``episode``, ``clip_id``, ``start``, ``stop``, ``wav_path``,
        a ``group`` key, one binary column per stutter class, and ``clip_id_str``.
    """
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}. Run ./setup_dataset.sh")

    df = pd.read_csv(labels_csv, dtype={"EpId": str})
    df.columns = [c.strip() for c in df.columns]
    for col in ("EpId", "ClipId", "Start", "Stop"):
        df[col] = df[col].astype(str).str.strip()

    total_clips = len(df)

    dropped = {}
    for flag in list(exclude_flags if exclude_flags is not None else EXCLUDE_FLAGS) + list(extra_exclude_flags or []):
        if flag not in df.columns:
            continue
        mask = df[flag].astype(float) > 0
        dropped[flag] = int(mask.sum())
        df = df[~mask]

    if clips_dir is not None:
        clips_dir = os.path.abspath(os.path.expanduser(clips_dir))
    use_clips = clips_dir is not None and os.path.isdir(clips_dir)
    if clips_dir is not None and not use_clips:
        raise FileNotFoundError(f"clips_dir does not exist: {clips_dir}")

    # Map the long show title to the show_id used on disk (Waves/<show_id>/).
    # Only needed when slicing episodes: the pre-cut mirror is named by the
    # show title, which is what the clip_id_str column always holds.
    show_ids = _show_ids(waves_dir) if (require_audio and not use_clips) else None

    df["show"] = df["Show"].astype(str).str.strip()
    df["show_id"] = df["show"]
    if show_ids is not None:
        df["show_id"] = df["show_id"].map(lambda s: show_ids.get(s, s))

    df["episode"] = df["EpId"]
    df["clip_id"] = df["ClipId"].astype(int)
    df["start"] = df["Start"].astype(int)
    df["stop"] = df["Stop"].astype(int)
    df["clip_id_str"] = [f"{s}_{e}_{c}" for s, e, c in zip(df["show"], df["episode"], df["clip_id"])]
    if use_clips:
        df["wav_path"] = [os.path.join(clips_dir, f"{c}.wav") for c in df["clip_id_str"]]
    else:
        df["wav_path"] = [os.path.join(waves_dir, show, f"{ep}.wav") for show, ep in zip(df["show_id"], df["episode"])]

    for class_name in STUTTER_CLASSES:
        df[class_name] = (df[LABEL_COLUMNS[class_name]].astype(float) > 0).astype(np.int64)

    if require_audio:
        missing = ~df["wav_path"].map(os.path.exists)
        if missing.any():
            by_show = df[missing].groupby("show_id").size().to_dict()
            df = df[~missing]
            dropped["missing_audio"] = {k: int(v) for k, v in sorted(by_show.items())}

        # A 44-byte WAV is a header with no audio; reading it yields a zero-length
        # array and librosa raises. Some SEP-28k mirrors ship a few hundred.
        sizes = df["wav_path"].map(lambda p: os.path.getsize(p) if os.path.exists(p) else 0)
        empty = sizes <= WAV_HEADER_BYTES
        if empty.any():
            by_show = df[empty].groupby("show_id").size().to_dict()
            df = df[~empty]
            dropped["empty_audio"] = {k: int(v) for k, v in sorted(by_show.items())}

    df = df.sort_values(["show_id", "episode", "start"]).reset_index(drop=True)

    table = df.copy()
    table.attrs["dropped"] = dropped
    table.attrs["total_clips"] = total_clips
    table.attrs["audio_source"] = "clips" if use_clips else "waves"

    source = clips_dir if use_clips else waves_dir
    print(f"  Audio source: {source} ({'pre-cut clips' if use_clips else 'sliced from episodes'})")
    print(f"  Clips available: {len(table)} / {total_clips} in {labels_csv}")
    for key, value in dropped.items():
        if isinstance(value, dict):
            detail = ", ".join(f"{k}={v}" for k, v in value.items())
            print(f"    skipped {key}: {detail}")
        else:
            print(f"    skipped {key}>0: {value}")
    return table


def _show_ids(waves_dir: str) -> Dict[str, str]:
    """Map show title -> show_id using the episodes CSV and the Waves directory."""
    episodes_csv = os.path.join(os.path.dirname(waves_dir), "SEP-28k_episodes.csv")
    if not os.path.exists(episodes_csv):
        return {}
    eps = pd.read_csv(episodes_csv, header=None, names=["show", "episode", "url", "show_id", "ep_idx"])
    eps["show"] = eps["show"].astype(str).str.strip()
    eps["show_id"] = eps["show_id"].astype(str).str.strip()
    return dict(zip(eps["show"], eps["show_id"]))


def label_matrix(table: pd.DataFrame) -> np.ndarray:
    """Return the (N, 5) binary label matrix in STUTTER_CLASSES order."""
    return table[STUTTER_CLASSES].to_numpy(dtype=np.int64)


def group_keys(table: pd.DataFrame, group_by: str = "episode") -> np.ndarray:
    """Return the grouping key per clip.

    Args:
        group_by: "episode" for ``<show_id>|<episode>`` or "show" for ``<show_id>``.
    """
    if group_by == "episode":
        return (table["show_id"] + "|" + table["episode"]).to_numpy()
    if group_by == "show":
        return table["show_id"].to_numpy()
    raise ValueError(f"group_by must be 'episode' or 'show', got {group_by!r}")


def label_distribution(table: pd.DataFrame, indices: np.ndarray = None) -> Dict[str, float]:
    """Positive rate per stutter class, for a subset of rows (default: all)."""
    sub = table if indices is None else table.iloc[np.asarray(indices)]
    n = max(len(sub), 1)
    return {c: round(float(sub[c].sum()) / n, 4) for c in STUTTER_CLASSES}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def build_random_split(
    table: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Split:
    """Multi-label stratified random clip split (70/15/15 by default).

    Uses ``iterative_train_test_split`` exactly like ``Model/model_train.ipynb``,
    but with an explicit seed so the split is reproducible.

    Warning:
        Clips are adjacent slices of the same recording, so this split leaks
        speaker identity between subsets. Scores are optimistic.
    """
    from skmultilearn.model_selection import iterative_train_test_split

    y = label_matrix(table).astype(np.float64)
    rows = np.arange(len(table)).reshape(-1, 1)

    # ``test_size`` is the fraction held out, so the first call reserves
    # val_size + test_size for the second stage and leaves the rest for training.
    x_train, y_train, x_temp, y_temp = iterative_train_test_split(rows, y, test_size=val_size + test_size)
    x_val, y_val, x_test, y_test = iterative_train_test_split(
        x_temp, y_temp, test_size=val_size / (val_size + test_size)
    )

    train = x_train.flatten()
    val = x_val.flatten()
    test = x_test.flatten()

    return Split(
        name="random",
        train=np.sort(train),
        val=np.sort(val),
        test=np.sort(test),
        group_by=None,
        meta={
            "seed": seed,
            "test_size": test_size,
            "val_size": val_size,
            "strategy": "multi-label stratified random (iterative_train_test_split)",
            "leakage": "speaker identity leaks between splits — scores are optimistic",
            "label_distribution": {
                "train": label_distribution(table, train),
                "val": label_distribution(table, val),
                "test": label_distribution(table, test),
            },
        },
    )


def build_group_split(
    table: pd.DataFrame,
    group_by: str = "episode",
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Split:
    """Group-disjoint split: each recording is assigned to exactly one subset.

    Groups are shuffled with a seed, then whole groups are handed to the test
    and val buckets, each targeting its requested share of clips. When the group
    count is small (a show-disjoint split has one group per show) a bucket is
    filled by the subset whose total clip count lands closest to the budget,
    since a single show can be a large fraction of the dataset. With many groups
    (episodes) a greedy fill in decreasing-size order is both cheaper and
    accurate enough.
    """
    groups = group_keys(table, group_by)
    unique = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)

    sizes = {g: int(np.sum(groups == g)) for g in unique}
    total = len(table)

    # Shuffled order doubles as the deterministic tie-breaker below.
    tiebreak = {g: i for i, g in enumerate(unique)}
    largest_first = sorted(unique, key=lambda g: (-sizes[g], tiebreak[g]))
    exact_search = len(unique) <= EXACT_GROUP_SEARCH_LIMIT

    def take(order, budget):
        """Assign whole groups until ``budget`` clips are covered."""
        if budget <= 0:
            return [], 0
        if exact_search:
            best, best_gap = [], None
            for subset in _subsets_within_budget(order, sizes, budget):
                used = sum(sizes[g] for g in subset)
                gap = abs(used - budget)
                key = (gap, min(tiebreak[g] for g in subset))
                if best_gap is None or key < best_gap:
                    best, best_gap = list(subset), key
            if best:
                return best, sum(sizes[g] for g in best)
        picked, used = [], 0
        for g in order:
            if used >= budget:
                break
            picked.append(g)
            used += sizes[g]
        return picked, used

    test_groups, n_test = take(largest_first, total * test_size)
    remaining = [g for g in largest_first if g not in set(test_groups)]
    val_groups, _ = take(remaining, total * val_size)
    train_groups = [g for g in remaining if g not in set(val_groups)]

    def rows_for(gs):
        mask = np.isin(groups, np.asarray(gs))
        return np.sort(np.flatnonzero(mask))

    return Split(
        name=f"group_{group_by}",
        train=rows_for(train_groups),
        val=rows_for(val_groups),
        test=rows_for(test_groups),
        group_by=group_by,
        meta={
            "seed": seed,
            "test_size": test_size,
            "val_size": val_size,
            "num_groups": len(unique),
            "strategy": f"group-disjoint by {group_by}",
            "test_groups": sorted(str(g) for g in test_groups),
            "val_groups": sorted(str(g) for g in val_groups),
            "label_distribution": {
                "train": label_distribution(table, rows_for(train_groups)),
                "val": label_distribution(table, rows_for(val_groups)),
                "test": label_distribution(table, rows_for(test_groups)),
            },
        },
    )


def build_split(table: pd.DataFrame, name: str, group_by: str = "episode", **kwargs) -> Split:
    """Dispatch to the split builder for ``name`` ("random", "group", "group_episode", "group_show")."""
    if name == "random":
        return build_random_split(table, **kwargs)
    if name in ("group", "group_episode"):
        return build_group_split(table, group_by="episode", **kwargs)
    if name == "group_show":
        return build_group_split(table, group_by="show", **kwargs)
    raise ValueError(f"Unknown split {name!r}; use random, group, group_episode or group_show")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class _WavCache:
    """LRU cache of decoded episode WAVs.

    Clips are contiguous slices of an episode, and the table is sorted by
    (show_id, episode, start), so iteration touches the same handful of
    episodes over and over. DataLoader workers receive batches round-robin, so
    each worker strides through the table — hence an LRU sized to the worker
    count rather than a single slot.
    """

    def __init__(self, maxsize: int = 16):
        self.maxsize = maxsize
        self._entries: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, path: str, sample_rate: int) -> np.ndarray:
        if path in self._entries:
            self._entries.move_to_end(path)
            return self._entries[path]

        sr, audio = wavfile.read(path)
        if sr != sample_rate:
            raise ValueError(f"Expected {sample_rate}Hz audio, got {sr}Hz for {path}")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32) / 32768.0

        self._entries[path] = audio
        if len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
        return audio


class SEP28kClipDataset(Dataset):
    """3-second SEP-28k clips as z-scored mel spectrograms, one per item.

    Feature extraction mirrors ``StutterDetector.extract_features``
    (librosa mel -> power_to_db(ref=max) -> per-clip z-score) so checkpoints
    stay compatible with the production inference path.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        indices: np.ndarray,
        n_mels: int,
        n_fft: int = 1024,
        hop_length: int = 512,
        sample_rate: int = SAMPLE_RATE,
        target_length: int = TARGET_LENGTH,
        slice_episodes: Optional[bool] = None,
    ):
        self.table = table
        self.indices = np.asarray(indices)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.labels = label_matrix(table)
        self._cache = _WavCache()
        # Pre-cut clip files already *are* the clip, so the CSV's Start/Stop
        # sample offsets (which index into the episode) must not be applied.
        if slice_episodes is None:
            slice_episodes = table.attrs.get("audio_source", "waves") != "clips"
        self.slice_episodes = bool(slice_episodes)

    def __len__(self) -> int:
        return len(self.indices)

    def _features(self, audio: np.ndarray) -> np.ndarray:
        import librosa

        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        elif len(audio) > self.target_length:
            audio = audio[: self.target_length]

        mels = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        mels_db = librosa.power_to_db(mels, ref=np.max)
        mean = np.mean(mels_db)
        std = np.std(mels_db) + 1e-10
        return ((mels_db - mean) / std).astype(np.float32)

    def __getitem__(self, i: int):
        row = int(self.indices[i])
        rec = self.table.iloc[row]
        audio = self._cache.get(rec["wav_path"], self.sample_rate)
        if self.slice_episodes:
            audio = audio[rec["start"] : rec["stop"]]
        features = self._features(audio)
        labels = self.labels[row].astype(np.float32)
        return torch.from_numpy(features), torch.from_numpy(labels), row


def n_frames_for(hop_length: int, target_length: int = TARGET_LENGTH) -> int:
    """Number of mel frames for a fixed-length clip (librosa centres the analysis window)."""
    return 1 + target_length // hop_length


class FeatureCache:
    """All mel features for a clip subset, materialised once in a single process.

    Episode WAVs average ~82 MB here, so decode-on-the-fly is dominated by disk
    reads: measured on this dataset at 129 clips/s with one process (one warm
    LRU) versus 19 clips/s with 8 DataLoader workers, each holding a private
    cache copy and thrashing the page cache. Extracting once into a contiguous
    array turns every later epoch into a pure memory read — a full split is
    only a few hundred MB.

    Pass ``cache_dir`` to persist the array and reuse it across runs.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        indices: np.ndarray,
        n_mels: int,
        n_fft: int = 1024,
        hop_length: int = 512,
        sample_rate: int = SAMPLE_RATE,
        target_length: int = TARGET_LENGTH,
        cache_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.rows = np.asarray(indices, dtype=np.int64)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.shape = (len(self.rows), n_mels, n_frames_for(hop_length, target_length))

        path = self._cache_path(cache_dir)
        if path and os.path.exists(path):
            loaded = np.load(path)
            if loaded.shape == self.shape:
                self.features = loaded
                if verbose:
                    print(f"  features (n_mels={n_mels}): reused {self.features.nbytes / 1e6:.0f} MB from {path}")
                return
            if verbose:
                print(f"  features (n_mels={n_mels}): stale cache {loaded.shape} != {self.shape}, recomputing")

        started = time.time()
        self.features = self._extract(table)
        if verbose:
            rate = len(self.rows) / max(time.time() - started, 1e-9)
            print(
                f"  features (n_mels={n_mels}): {len(self.rows)} clips at {rate:.0f}/s "
                f"({self.features.nbytes / 1e6:.0f} MB, {time.time() - started:.1f}s)"
            )
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, self.features)

    def _cache_path(self, cache_dir: Optional[str]) -> Optional[str]:
        if not cache_dir:
            return None
        signature = hashlib.sha256(
            json.dumps(
                {
                    "rows": self.rows.tolist(),
                    "n_mels": self.n_mels,
                    "n_fft": self.n_fft,
                    "hop_length": self.hop_length,
                    "sample_rate": self.sample_rate,
                    "target_length": self.target_length,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]
        return os.path.join(cache_dir, f"features_nm{self.n_mels}_nf{self.n_fft}_h{self.hop_length}_{signature}.npy")

    def _extract(self, table: pd.DataFrame) -> np.ndarray:
        """Decode every clip once, in table order so the WAV LRU stays warm."""
        source = SEP28kClipDataset(
            table,
            self.rows,
            self.n_mels,
            self.n_fft,
            self.hop_length,
            sample_rate=self.sample_rate,
            target_length=self.target_length,
        )
        out = np.zeros(self.shape, dtype=np.float32)
        for i in range(len(self.rows)):
            out[i] = source[i][0].numpy()
        return out

    def dataset(self, table: pd.DataFrame) -> "CachedClipDataset":
        """Wrap the materialised features in a Dataset."""
        return CachedClipDataset(self, table)


class CachedClipDataset(Dataset):
    """Dataset view over a ``FeatureCache`` — no disk access, no workers needed."""

    def __init__(self, cache: FeatureCache, table: pd.DataFrame):
        self.cache = cache
        self.labels = label_matrix(table)

    def __len__(self) -> int:
        return len(self.cache.rows)

    def __getitem__(self, i: int):
        row = int(self.cache.rows[i])
        return (
            torch.from_numpy(self.cache.features[i]),
            torch.from_numpy(self.labels[row].astype(np.float32)),
            row,
        )
