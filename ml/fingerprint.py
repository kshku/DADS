"""Checkpoint fingerprints for DADS models.

Training encodes every relevant hyperparameter into the checkpoint filename
(a "fingerprint") so a weight file can always be traced back to the exact
configuration that produced it, and so evaluation can recover the feature
parameters without consulting a side file. This module is the single source of
truth for both directions.

Two filename formats are supported:

fingerprint format (written by ``ml.train``)::

    prolongation_e40_b32_lr1e-3_n1024h512m128_bce_s42_group_1a2b3c4d.pth

legacy production format (shipped in ``Model/models/copy``)::

    prolongation_model_1024_512_128_40.pth
"""

import hashlib
import json
import os
import re
from typing import Dict, Union

from ml.config import STUTTER_CLASSES

# Hyperparameters that must match for a checkpoint to be reused for training
RESUME_KEYS = [
    "class_name",
    "n_fft",
    "hop_length",
    "n_mels",
    "epochs",
    "batch_size",
    "lr",
    "loss_type",
    "seed",
    "split",
]

# Parameters that change the data but not the model shape; covered by the hash
DATA_KEYS = ["group_by", "test_size", "val_size", "max_samples"]

FINGERPRINT_FMT = (
    "{class_name}_e{epochs}_b{batch_size}_lr{lr}_n{n_fft}h{hop_length}m{n_mels}_{loss_type}_s{seed}_{split}_{hash}"
)

_FINGERPRINT_PATTERN = re.compile(
    r"^(?P<class_name>\w+)"
    r"_e(?P<epochs>\d+)"
    r"_b(?P<batch_size>\d+)"
    r"_lr(?P<lr>[\d.eE+-]+)"
    r"_n(?P<n_fft>\d+)"
    r"h(?P<hop_length>\d+)"
    r"m(?P<n_mels>\d+)"
    r"_(?P<loss_type>\w+)"
    r"_s(?P<seed>\d+)"
    r"_(?P<split>\w+)"
    r"_(?P<hash>[0-9a-f]{8})$"
)

LEGACY_PATTERN = re.compile(
    r"^(?P<class_name>\w+?)_model_(?P<n_fft>\d+)_(?P<hop_length>\d+)_(?P<n_mels>\d+)_(?P<epochs>\d+)$"
)


def _as_dict(values: Union[Dict, object]) -> Dict:
    """Accept either a mapping or an argparse.Namespace-like object."""
    if isinstance(values, dict):
        return dict(values)
    return {k: v for k, v in vars(values).items() if not k.startswith("_")}


def _fmt_fp(v) -> str:
    """Format a value for embedding in a filename (floats shortened, no padded exponents)."""
    if isinstance(v, float):
        s = f"{v:.10g}"
        return re.sub(r"e([+-])0(\d)", r"e\1\2", s)
    return str(v)


def config_hash(values: Union[Dict, object]) -> str:
    """Stable 8-hex-char digest of the full configuration.

    Covers every key in ``RESUME_KEYS`` and ``DATA_KEYS``, so any change that
    affects the resulting model yields a different checkpoint name.
    """
    d = _as_dict(values)
    payload = {k: d.get(k) for k in RESUME_KEYS + DATA_KEYS}
    payload = {k: _fmt_fp(v) for k, v in payload.items()}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:8]


def fingerprint(values: Union[Dict, object]) -> str:
    """Build the fingerprint string for a configuration (no extension)."""
    d = _as_dict(values)
    merged = dict(d)
    merged["hash"] = config_hash(d)
    missing = [k for k in RESUME_KEYS if k not in merged]
    if missing:
        raise ValueError(f"Missing fingerprint keys: {missing}")
    return FINGERPRINT_FMT.format(**{k: _fmt_fp(merged[k]) for k in RESUME_KEYS} | {"hash": merged["hash"]})


def parse_fingerprint(fp: str) -> Dict:
    """Parse a fingerprint string back into its parameters."""
    fp = fp[:-4] if fp.endswith(".pth") else fp
    m = _FINGERPRINT_PATTERN.match(fp)
    if not m:
        raise ValueError(f"Cannot parse fingerprint: {fp}")
    d = m.groupdict()
    for k in ("epochs", "batch_size", "n_fft", "hop_length", "n_mels", "seed"):
        d[k] = int(d[k])
    for k in ("lr",):
        d[k] = float(d[k])
    d["hash"] = d["hash"]
    d["format"] = "fingerprint"
    return d


def fingerprint_filename(values: Union[Dict, object]) -> str:
    """Filename (including ``.pth``) for a configuration."""
    return f"{fingerprint(values)}.pth"


def parse_model_file(model_path: str) -> Dict:
    """Recover the model spec from any supported checkpoint filename.

    Returns a dict with ``class_name``, ``n_fft``, ``hop_length``, ``n_mels``,
    ``epochs`` and ``format`` (``"fingerprint"`` or ``"legacy"``). Raises
    ValueError when the name matches neither format.
    """
    stem = os.path.basename(model_path)
    if stem.endswith(".pth"):
        stem = stem[:-4]

    m = _FINGERPRINT_PATTERN.match(stem)
    if m:
        return parse_fingerprint(stem)

    m = LEGACY_PATTERN.match(stem)
    if m:
        d = m.groupdict()
        for k in ("n_fft", "hop_length", "n_mels", "epochs"):
            d[k] = int(d[k])
        d["format"] = "legacy"
        d["path"] = model_path
        return d

    raise ValueError(
        f"Unrecognised checkpoint filename: {os.path.basename(model_path)}. "
        "Expected {class}_e{epochs}_b{batch}_lr{lr}_n{n_fft}h{hop}m{n_mels}_{loss}_s{seed}_{split}_{hash}.pth "
        "or the legacy {class}_model_{n_fft}_{hop}_{n_mels}_{epochs}.pth."
    )


def find_checkpoint(class_name: str, models_dir: str, prefer: str = "fingerprint") -> str:
    """Locate a checkpoint for a class inside a directory.

    Args:
        class_name: One of STUTTER_CLASSES.
        models_dir: Directory to search (non-recursive).
        prefer: "fingerprint" to prefer fingerprint-named files, "legacy" to
            prefer the legacy convention.

    Returns:
        Absolute path of the chosen checkpoint.

    Raises:
        FileNotFoundError: when no matching checkpoint exists.
    """
    if class_name not in STUTTER_CLASSES:
        raise ValueError(f"Unknown class {class_name!r}; expected one of {STUTTER_CLASSES}")

    matches = []
    for name in sorted(os.listdir(models_dir)):
        if not name.endswith(".pth"):
            continue
        try:
            spec = parse_model_file(os.path.join(models_dir, name))
        except ValueError:
            continue
        if spec["class_name"] == class_name:
            matches.append((spec["format"], os.path.join(models_dir, name)))

    if not matches:
        raise FileNotFoundError(
            f"No {class_name} checkpoint in {models_dir}. Train one with "
            f"`python -m ml.train --class {class_name}` or point --models_dir at Model/models/copy."
        )

    matches.sort(key=lambda t: (t[0] != prefer, t[1]))
    return matches[0][1]


def describe_model(spec: Dict) -> str:
    """One-line human description of a parsed model spec, for reports."""
    parts = [
        f"n_fft={spec['n_fft']}",
        f"hop_length={spec['hop_length']}",
        f"n_mels={spec['n_mels']}",
        f"epochs={spec['epochs']}",
    ]
    if spec.get("format") == "fingerprint":
        parts += [f"batch_size={spec.get('batch_size')}", f"lr={spec.get('lr')}", f"seed={spec.get('seed')}"]
    return ", ".join(parts)
