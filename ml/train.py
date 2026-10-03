"""Train the five DADS binary stutter classifiers on a reproducible split.

Ported from ``Model/model_train.ipynb`` and parameterised so the resulting
checkpoints stay drop-in compatible with the shipped models: the architecture is
``shared.connector.Model`` and the feature parameters default to the production
values recorded in ``Model/models/copy``.

Examples::

    # All five models, episode-disjoint split, production hyperparameters
    python -m ml.train --class all --split group

    # Smoke test: one model, few samples, 2 epochs
    python -m ml.train --class block --split random --max_samples 500 --epochs 2
"""

import argparse
import hashlib
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml import fingerprint as fp
from ml import metrics as M
from ml.config import (
    DEFAULT_THRESHOLD,
    LABELS_CSV,
    STUTTER_CLASSES,
    TRAINED_MODELS_DIR,
    WAVS_DIR,
    feature_params,
    portable_path,
)
from ml.data import FeatureCache, build_split, label_distribution, load_clip_table
from ml.modeling import build_model, get_device, pos_weight_for, set_seed

# Per-run memo so classes sharing an n_mels (prolongation/block/interjection all
# use 128) extract features once for the whole training pass.
_FEATURE_CACHES: Dict[tuple, FeatureCache] = {}


def get_feature_cache(table, indices, n_mels: int, n_fft: int, hop: int, cache_dir: str = None) -> FeatureCache:
    """Return a ``FeatureCache`` for a subset, reusing it across classes."""
    indices = np.asarray(indices, dtype=np.int64)
    key = (
        n_mels,
        n_fft,
        hop,
        cache_dir,
        hashlib.sha256(indices.tobytes()).hexdigest()[:16],
    )
    if key not in _FEATURE_CACHES:
        _FEATURE_CACHES[key] = FeatureCache(table, indices, n_mels, n_fft, hop, cache_dir=cache_dir)
    return _FEATURE_CACHES[key]


class FocalLoss(nn.Module):
    """Binary focal loss on raw logits.

    The usual ``FocalLoss`` targets multiclass ``cross_entropy``/``softmax``;
    the DADS heads are single-logit binary, so this variant wraps
    ``binary_cross_entropy_with_logits`` and down-weights easy examples by
    ``(1 - p_t) ** gamma``.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        return (((1 - p_t) ** self.gamma) * bce).mean()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DADS stutter classifiers.")
    parser.add_argument(
        "--class",
        dest="class_name",
        default="all",
        choices=STUTTER_CLASSES + ["all"],
        help="Stutter class to train, or 'all'.",
    )
    parser.add_argument(
        "--split",
        default="random",
        choices=["random", "group"],
        help="random = stratified clip split; group = recording-disjoint.",
    )
    parser.add_argument(
        "--group_by", default="episode", choices=["episode", "show"], help="Grouping unit for --split group."
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--loss_type",
        default="bce",
        choices=["bce", "bce_balanced", "focal"],
        help="bce_balanced weights positives by the neg/pos ratio.",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience in epochs.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers. Features are pre-extracted into RAM, so 0 is fastest.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N")
    parser.add_argument("--n_fft", type=int, default=None, help="Override production n_fft.")
    parser.add_argument("--hop_length", type=int, default=None, help="Override production hop_length.")
    parser.add_argument("--n_mels", type=int, default=None, help="Override production n_mels.")
    parser.add_argument("--output_dir", default=TRAINED_MODELS_DIR)
    parser.add_argument("--cache_dir", default=None, help="Persist extracted mel features here to reuse across runs.")
    parser.add_argument("--labels_csv", default=LABELS_CSV)
    parser.add_argument(
        "--waves_dir", default=WAVS_DIR, help="Directory of <show_id>/<episode>.wav files to slice clips from."
    )
    parser.add_argument(
        "--clips_dir",
        default=None,
        help="Directory of already-cut <Show>_<EpId>_<ClipId>.wav clips. "
        "Use this when the full episode WAVs are not downloaded; "
        "default is to slice --waves_dir.",
    )
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap clips per subset — smoke tests only, invalidates the metrics.",
    )
    parser.add_argument(
        "--exclude_extra", default="", help="Comma-separated extra label columns to drop, e.g. PoorAudioQuality,Unsure."
    )
    parser.add_argument("--no_curves", action="store_true", help="Skip training curve PNGs.")
    return parser.parse_args(argv)


def _make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # Without this, PyTorch respawns every worker for each new iterator,
        # i.e. once per epoch per loader.
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=device.type == "cuda",
    )


def _build_criterion(args: argparse.Namespace, pos_weight: torch.Tensor = None):
    """Return a ``fn(logits, targets) -> loss`` for the requested loss."""
    if args.loss_type == "focal":
        module = FocalLoss(gamma=args.focal_gamma)
        return lambda logits, targets: module(logits, targets)
    module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return lambda logits, targets: module(logits, targets)


def train_one_epoch(model, loader, criterion, optimizer, device, class_index: int) -> float:
    model.train()
    total, n_batches = 0.0, 0
    for features, labels, _ in loader:
        features = features.to(device, non_blocking=True)
        targets = labels[:, class_index].to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(features).flatten(), targets)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def run_eval(model, loader, criterion, device, class_index: int, threshold: float) -> Dict:
    """Evaluate one model over a loader, returning loss, metrics and raw scores."""
    model.eval()
    losses, scores, trues = [], [], []
    for features, labels, _ in loader:
        features = features.to(device, non_blocking=True)
        targets = labels[:, class_index].to(device, non_blocking=True)
        logits = model(features).flatten()
        losses.append(float(criterion(logits, targets).item()))
        scores.append(torch.sigmoid(logits).cpu().numpy())
        trues.append(targets.cpu().numpy())

    y_true = np.concatenate(trues).astype(int) if trues else np.array([], dtype=int)
    y_scores = np.concatenate(scores).astype(float) if scores else np.array([], dtype=float)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "metrics": M.compute_binary_metrics(y_true, y_scores, threshold=threshold),
        "y_true": y_true,
        "y_scores": y_scores,
    }


def _save_curves(history: List[Dict], path: str, class_name: str, split_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", ms=3, label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], marker="o", ms=3, label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["val_f1"] for h in history], marker="o", ms=3, label="val F1")
    ax2.plot(epochs, [h["val_auroc"] for h in history], marker="o", ms=3, label="val AUROC")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(f"{class_name} - {split_name} split")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def train_class(args: argparse.Namespace, class_name: str, table, split, device: torch.device) -> Dict:
    """Train one binary model and return its training history and test metrics."""
    params = feature_params(class_name)
    params["epochs"] = args.epochs
    if args.n_fft:
        params["n_fft"] = args.n_fft
    if args.hop_length:
        params["hop_length"] = args.hop_length
    if args.n_mels:
        params["n_mels"] = args.n_mels

    config = {
        **params,
        "class_name": class_name,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "loss_type": args.loss_type,
        "seed": args.seed,
        "split": split.name,
        "group_by": args.group_by if args.split == "group" else None,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "max_samples": args.max_samples,
    }

    set_seed(args.seed)
    filename = fp.fingerprint_filename(config)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, filename)

    print(f"\n{'=' * 72}")
    print(
        f"  {class_name.upper()}   n_mels={params['n_mels']}  n_fft={params['n_fft']}  "
        f"hop={params['hop_length']}  epochs={args.epochs}"
    )
    print(f"  split={split.name}  train={len(split.train)}  val={len(split.val)}  test={len(split.test)}")
    print(f"  checkpoint -> {filename}")
    print(f"{'=' * 72}")

    class_index = STUTTER_CLASSES.index(class_name)
    n_mels, n_fft, hop = params["n_mels"], params["n_fft"], params["hop_length"]

    # Materialise mel features once per class instead of per epoch. Feature
    # extraction is identical for every class sharing an n_mels, so the cache is
    # keyed on n_mels and reused across the three 128-band models.
    train_cache = get_feature_cache(table, split.train, n_mels, n_fft, hop, args.cache_dir)
    val_cache = get_feature_cache(table, split.val, n_mels, n_fft, hop, args.cache_dir)
    test_cache = get_feature_cache(table, split.test, n_mels, n_fft, hop, args.cache_dir)

    train_loader = _make_loader(train_cache.dataset(table), args.batch_size, True, args.num_workers, device)
    val_loader = _make_loader(val_cache.dataset(table), args.batch_size, False, args.num_workers, device)
    test_loader = _make_loader(test_cache.dataset(table), args.batch_size, False, args.num_workers, device)

    pos_weight = None
    if args.loss_type == "bce_balanced":
        pos_weight = pos_weight_for(table[STUTTER_CLASSES].to_numpy()[split.train, class_index], device)
        print(f"  pos_weight: {float(pos_weight):.3f}")

    model = build_model(n_mels).to(device)
    criterion = _build_criterion(args, pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: List[Dict] = []
    best_f1, best_epoch, best_state, stale = -1.0, -1, None, 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, class_index)
        val = run_eval(model, val_loader, criterion, device, class_index, DEFAULT_THRESHOLD)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val["loss"], 5),
                "val_f1": val["metrics"]["f1"],
                "val_auroc": val["metrics"]["auroc"],
                "val_precision": val["metrics"]["precision"],
                "val_recall": val["metrics"]["recall"],
            }
        )

        flag = ""
        if val["metrics"]["f1"] > best_f1:
            best_f1 = val["metrics"]["f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            flag = "  <- best"
        else:
            stale += 1

        print(
            f"  epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  "
            f"val_F1={val['metrics']['f1']:.4f}  val_AUROC={val['metrics']['auroc']:.4f}{flag}"
        )

        if stale >= args.patience:
            print(f"  early stopping: no val F1 improvement for {args.patience} epochs")
            break

    elapsed = time.time() - started

    if best_state is not None:
        torch.save(best_state, checkpoint_path)
        model.load_state_dict(best_state)
        model.to(device)

    test = run_eval(model, test_loader, criterion, device, class_index, DEFAULT_THRESHOLD)
    sweep = M.run_threshold_sweep(test["y_true"], test["y_scores"])

    M.print_binary_report(test["metrics"], f"{class_name} (test, split={split.name})")
    print(f"  Best epoch {best_epoch}  |  trained in {elapsed / 60:.1f} min")
    print(f"  Checkpoint saved: {checkpoint_path}")

    curves_path = os.path.splitext(checkpoint_path)[0] + "_curves.png"
    if not args.no_curves and history:
        _save_curves(history, curves_path, class_name, split.name)

    history_path = os.path.splitext(checkpoint_path)[0] + "_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return {
        "class_name": class_name,
        "checkpoint": checkpoint_path,
        "fingerprint": fp.fingerprint(config),
        "config": config,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "train_minutes": round(elapsed / 60, 2),
        "test_metrics": test["metrics"],
        "threshold_sweep": {k: v for k, v in sweep.items() if k != "sweep"},
        "curves": None if args.no_curves else curves_path,
    }


def main(argv=None) -> Dict:
    args = parse_args(argv)
    device = get_device(args.device)

    print("\nDADS training")
    print(f"  device: {device}")
    print(f"  split:  {args.split}" + (f" (grouped by {args.group_by})" if args.split == "group" else ""))
    print(f"  loss:   {args.loss_type}   lr={args.lr}   batch={args.batch_size}   seed={args.seed}")

    extra = [f.strip() for f in args.exclude_extra.split(",") if f.strip()]
    table = load_clip_table(
        labels_csv=args.labels_csv,
        waves_dir=args.waves_dir,
        clips_dir=args.clips_dir,
        extra_exclude_flags=extra,
    )

    split = build_split(
        table,
        name=args.split,
        group_by=args.group_by,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    print(f"  subsets: {split.sizes()}")

    if args.max_samples:
        rng = np.random.RandomState(args.seed)
        for name in ("train", "val", "test"):
            idx = getattr(split, name)
            if len(idx) > args.max_samples:
                setattr(split, name, np.sort(rng.choice(idx, args.max_samples, replace=False)))
        split.meta["label_distribution"] = {
            name: label_distribution(table, getattr(split, name)) for name in ("train", "val", "test")
        }
        print(f"  WARNING: capped to {args.max_samples} clips per subset — metrics are not meaningful")

    classes = STUTTER_CLASSES if args.class_name == "all" else [args.class_name]
    results = [train_class(args, name, table, split, device) for name in classes]

    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "split": split.to_dict(),
                "dropped": table.attrs.get("dropped", {}),
                "args": {
                    k: portable_path(v) if isinstance(v, str) and os.path.isabs(v) else v for k, v in vars(args).items()
                },
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n{'=' * 72}")
    print(f"  Test summary  (split={split.name}, threshold={DEFAULT_THRESHOLD})")
    print(f"{'=' * 72}")
    for r in results:
        t = r["test_metrics"]
        print(
            f"  {r['class_name']:>13s}  F1={t['f1']:.4f}  AUROC={t['auroc']:.4f}  "
            f"Acc={t['accuracy']:.4f}  (best epoch {r['best_epoch']})"
        )
    print(f"\n  Summary written: {summary_path}")
    return {"results": results, "split": split}


if __name__ == "__main__":
    main()
