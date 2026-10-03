"""Model construction, checkpoint loading and batched inference for DADS.

The CNN architecture comes from ``shared.connector`` — the same module the
PyQt app and the FastAPI backend use — so a checkpoint trained here is loadable
by the production inference path with no conversion step.
"""

import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml import fingerprint as fp
from ml.config import STUTTER_CLASSES
from shared.connector import Model


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    """Resolve a device string ("auto", "cpu", "cuda", "cuda:1", ...)."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_model(n_mels: int) -> Model:
    """Instantiate the production CNN for a mel configuration."""
    return Model(n_mels=n_mels)


def load_checkpoint(model_path: str, device: torch.device) -> Tuple[Model, Dict]:
    """Load a checkpoint written in either supported filename format.

    Returns:
        ``(model, spec)`` where spec comes from ``fingerprint.parse_model_file``.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    spec = fp.parse_model_file(model_path)
    model = build_model(spec["n_mels"])
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, spec


def predict_scores(
    model: Model,
    dataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 8,
    class_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one binary model over a dataset.

    Args:
        model: Loaded CNN in eval mode.
        dataset: A ``SEP28kClipDataset``.
        class_index: Column of the label matrix to return. Defaults to 0.

    Returns:
        ``(y_true, y_scores, rows)`` — 1-D arrays aligned to ``rows``, the
        clip-table indices the predictions came from.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    idx = STUTTER_CLASSES.index(class_index) if isinstance(class_index, str) else (class_index or 0)

    all_true, all_scores, all_rows = [], [], []
    with torch.no_grad():
        for features, labels, rows in loader:
            logits = model(features.to(device, non_blocking=True))
            probs = torch.sigmoid(logits).flatten().cpu().numpy()
            all_scores.append(probs)
            all_true.append(labels[:, idx].numpy())
            all_rows.append(rows.numpy())

    if not all_rows:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=int)

    return (
        np.concatenate(all_true).astype(int),
        np.concatenate(all_scores).astype(float),
        np.concatenate(all_rows).astype(int),
    )


def pos_weight_for(labels: np.ndarray, device: torch.device, clip: float = 10.0) -> torch.Tensor:
    """Positive/negative ratio for ``BCEWithLogitsLoss``, capped at ``clip``."""
    labels = np.asarray(labels).astype(float)
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(min(n_neg / n_pos, clip), device=device, dtype=torch.float32)
