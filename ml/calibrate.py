"""Calibrate per-class decision thresholds and record them in the registry.

Thresholds are fitted on the **validation** split only, then reported on the
held-out test split. Fitting on test and quoting the resulting test score bakes
the test set into the operating point, which is the most common way a reported
F1 stops meaning anything.

The sweep maximises F1 by default. Youden's J is reported alongside so the
precision/recall trade-off at the chosen point is visible rather than implied.

Usage:
    python -m ml.calibrate --clips_dir /path/to/clips
    python -m ml.calibrate --models_dir Model/models/trained --register retrained
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from ml import fingerprint as fp
from ml import registry as reg
from ml.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_THRESHOLD,
    LABELS_CSV,
    PRODUCTION_MODELS_DIR,
    STUTTER_CLASSES,
    WAVS_DIR,
)
from ml.data import build_split, load_clip_table
from ml.metrics import compute_binary_metrics, find_optimal_threshold, run_threshold_sweep
from ml.modeling import get_device, load_checkpoint, predict_scores
from ml.train import get_feature_cache


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate decision thresholds on the validation split.")
    parser.add_argument("--models_dir", default=PRODUCTION_MODELS_DIR, help="Directory holding the checkpoints.")
    parser.add_argument("--register", default=None, help="Registry entry name to write (default: models_dir name).")
    parser.add_argument(
        "--split",
        default="group",
        choices=["random", "group", "group_episode", "group_show"],
        help="Split to calibrate on.",
    )
    parser.add_argument("--group_by", default="episode", choices=["episode", "show"])
    parser.add_argument("--class", dest="classes", default="all", help="'all' or comma-separated class names.")
    parser.add_argument("--metric", default="f1", choices=["f1", "youden"], help="Objective to maximise.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap clips per subset — smoke tests only.")
    parser.add_argument("--labels_csv", default=LABELS_CSV)
    parser.add_argument("--waves_dir", default=WAVS_DIR)
    parser.add_argument("--clips_dir", default=None, help="Pre-cut clip directory; see ml.evaluate --clips_dir.")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--registry", default=reg.REGISTRY_PATH, help="Registry file to read and update.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--exclude_extra",
        default="",
        help="Comma-separated extra label columns to drop, e.g. PoorAudioQuality,Unsure.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print what would change without writing.")
    return parser.parse_args(argv)


def _classes(spec: str) -> List[str]:
    if spec == "all":
        return list(STUTTER_CLASSES)
    names = [c.strip() for c in spec.split(",") if c.strip()]
    unknown = [c for c in names if c not in STUTTER_CLASSES]
    if unknown:
        raise SystemExit(f"Unknown class(es): {unknown}. Choose from {STUTTER_CLASSES}.")
    return names


def _subset(split, subset: str, max_samples: Optional[int], seed: int) -> np.ndarray:
    indices = getattr(split, subset)
    if max_samples and len(indices) > max_samples:
        rng = np.random.RandomState(seed)
        indices = np.sort(rng.choice(indices, max_samples, replace=False))
    return indices


def _score_subset(args, table, rows, specs: Dict[str, Dict], device) -> Dict[str, Dict]:
    """Score every class on one subset, sharing feature caches across classes.

    ``get_feature_cache`` memoises on (n_mels, n_fft, hop, rows), so the three
    128-band models share a single mel extraction.
    """
    scored = {}
    for class_name, spec in specs.items():
        cache = get_feature_cache(table, rows, spec["n_mels"], spec["n_fft"], spec["hop_length"], args.cache_dir)
        model, _ = load_checkpoint(spec["path"], device)
        y_true, y_scores, _ = predict_scores(
            model,
            cache.dataset(table),
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            class_index=STUTTER_CLASSES.index(class_name),
        )
        scored[class_name] = (y_true, y_scores)
    return scored


def _f1_at(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> float:
    return compute_binary_metrics(y_true, y_scores, threshold=threshold)["f1"]


def main(argv=None) -> Dict:
    args = parse_args(argv)
    device = get_device(args.device)
    classes = _classes(args.classes)
    entry_name = args.register or os.path.basename(os.path.normpath(args.models_dir))

    print("\nDADS threshold calibration")
    print(f"  device:     {device}")
    print(f"  models_dir: {args.models_dir}")
    print(f"  registry:   {entry_name}")
    print(f"  split:      {args.split}   objective: {args.metric}")
    print("  fitted on:  val      reported on: test")

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
    print(f"  sizes:      {split.sizes()}")

    val_rows = _subset(split, "val", args.max_samples, args.seed)
    test_rows = _subset(split, "test", args.max_samples, args.seed)

    specs = {}
    for class_name in classes:
        path = fp.find_checkpoint(class_name, args.models_dir)
        specs[class_name] = {**fp.parse_model_file(path), "path": path}

    print(f"\n  scoring val (n={len(val_rows)}) and test (n={len(test_rows)})...")
    val_scores = _score_subset(args, table, val_rows, specs, device)
    test_scores = _score_subset(args, table, test_rows, specs, device)

    results: Dict[str, Dict] = {}
    for class_name in classes:
        y_val, s_val = val_scores[class_name]
        y_test, s_test = test_scores[class_name]
        threshold, val_best = find_optimal_threshold(y_val, s_val, metric=args.metric)
        val_sweep = run_threshold_sweep(y_val, s_val)
        test_sweep = run_threshold_sweep(y_test, s_test)
        at_chosen = compute_binary_metrics(y_test, s_test, threshold=threshold)
        test_auroc = at_chosen["auroc"]

        results[class_name] = {
            "threshold": threshold,
            "fitted_on": "val",
            "objective": args.metric,
            "checkpoint": os.path.relpath(specs[class_name]["path"]),
            "val": {"auroc": val_sweep["auroc"], "best_score": val_best},
            "test": {
                "auroc": test_sweep["auroc"],
                "auprc": at_chosen["auprc"],
                "f1_at_threshold": at_chosen["f1"],
                "f1_at_0.4": _f1_at(y_test, s_test, DEFAULT_THRESHOLD),
                "precision": at_chosen["precision"],
                "recall": at_chosen["recall"],
                "specificity": at_chosen["specificity"],
                "accuracy": at_chosen["accuracy"],
                "youden_threshold": test_sweep["best_youden"]["threshold"],
            },
        }
        gain = results[class_name]["test"]["f1_at_threshold"] - results[class_name]["test"]["f1_at_0.4"]
        print(
            f"  {class_name:13s} t*={threshold:.2f}  "
            f"test F1@t*={at_chosen['f1']:.4f}  F1@0.4={results[class_name]['test']['f1_at_0.4']:.4f}  "
            f"({gain:+.4f})  prec={at_chosen['precision']:.3f} rec={at_chosen['recall']:.3f}  "
            f"AUROC={test_auroc:.4f}"
        )

    registry = reg.load_registry(args.registry)
    if args.dry_run:
        print("\n  --dry_run: registry not written")
    else:
        entry = reg.ensure_entry(registry, entry_name, args.models_dir)
        entry["model_files"] = {c: reg.relative_path(specs[c]["path"]) for c in classes}
        for class_name, info in results.items():
            reg.set_thresholds(registry, class_name, info["threshold"], entry_name)
        entry["calibration"] = {
            "split": args.split,
            "group_by": args.group_by if args.split.startswith("group") else None,
            "seed": args.seed,
            "objective": args.metric,
            "fitted_on": "val",
            "n_val": int(len(val_rows)),
            "n_test": int(len(test_rows)),
        }
        reg.save_registry(registry, args.registry)
        print(f"\n  Registry updated: {args.registry} [{entry_name}]")
        print(reg.describe(registry, entry_name))

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"calibration_{entry_name}.json")
    with open(out, "w") as f:
        json.dump({"entry": entry_name, "split": args.split, "results": results}, f, indent=2)
    print(f"\n  Calibration report: {out}")
    return results


if __name__ == "__main__":
    main()
