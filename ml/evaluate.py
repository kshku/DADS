"""Evaluate the DADS binary stutter classifiers and write metrics reports.

Ported from the reference evaluation module, adapted to DADS's
five independent binary CNNs and to SEP-28k's clip-offset labels.

Because the original notebook's train/test split was an unseeded random split
that was never saved, this script defines its own splits and reports them side
by side:

``random``
    Multi-label stratified clip split — comparable to ``accuracy.txt``.
``group``
    Episode-disjoint split — no recording in both subsets.

Examples::

    # Both splits over all five production models
    python -m ml.evaluate

    # One model, group split, misclassified dump
    python -m ml.evaluate --class prolongation --split group --save_misclassified

    # Smoke test on 200 clips
    python -m ml.evaluate --max_samples 200
"""

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

from ml import fingerprint as fp
from ml import metrics as M
from ml.config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_THRESHOLD,
    LABELS_CSV,
    PRODUCTION_MODELS_DIR,
    STUTTER_CLASSES,
    WAVS_DIR,
    portable_path,
)
from ml.data import build_split, load_clip_table
from ml.modeling import get_device, load_checkpoint, predict_scores
from ml.train import get_feature_cache

CONTAMINATION_LEGACY = (
    "Production weights were trained on an unseeded random ~70% of every clip "
    "(Model/model_train.ipynb), so their test subset overlaps training data. "
    "Treat these numbers as optimistic and retrain with `python -m ml.train` for clean scores."
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DADS stutter classifiers.")
    parser.add_argument(
        "--class",
        dest="class_name",
        default="all",
        choices=STUTTER_CLASSES + ["all"],
        help="Class to evaluate, or 'all'.",
    )
    parser.add_argument(
        "--split",
        default="both",
        choices=["random", "group", "both"],
        help="Which split(s) to score. 'both' reports them side by side.",
    )
    parser.add_argument(
        "--group_by", default="episode", choices=["episode", "show"], help="Grouping unit for the group split."
    )
    parser.add_argument(
        "--subset", default="test", choices=["train", "val", "test", "all"], help="Which subset of the split to score."
    )
    parser.add_argument(
        "--models_dir", default=PRODUCTION_MODELS_DIR, help="Directory holding one checkpoint per class."
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="Decision threshold (production default is 0.4)."
    )
    parser.add_argument(
        "--sweep_thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sweep t in [0.1, 0.9] and report optimal thresholds.",
    )
    parser.add_argument("--save_misclassified", action="store_true", help="Write misclassified clips per class.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers. Features are pre-extracted into RAM, so 0 is fastest.",
    )
    parser.add_argument("--cache_dir", default=None, help="Persist extracted mel features here to reuse across runs.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap clips per subset — smoke tests only, metrics are not meaningful.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument(
        "--exclude_extra", default="", help="Comma-separated extra label columns to drop, e.g. PoorAudioQuality,Unsure."
    )
    return parser.parse_args(argv)


def _subset_indices(split, subset: str) -> np.ndarray:
    if subset == "all":
        return np.sort(np.concatenate([split.train, split.val, split.test]))
    return getattr(split, subset)


def evaluate_split(args: argparse.Namespace, split_name: str, table, device) -> Dict:
    """Score every requested class on one split and build its report block."""
    split = build_split(
        table,
        name=split_name,
        group_by=args.group_by,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    subset = _subset_indices(split, args.subset)
    if args.max_samples and len(subset) > args.max_samples:
        rng = np.random.RandomState(args.seed)
        subset = np.sort(rng.choice(subset, args.max_samples, replace=False))
        split.meta["capped_at"] = args.max_samples

    print(f"\n{'#' * 72}")
    print(
        f"# SPLIT: {split.name}   subset={args.subset}   n={len(subset)}   "
        f"(train={len(split.train)} val={len(split.val)} test={len(split.test)})"
    )
    print(f"{'#' * 72}")

    classes = STUTTER_CLASSES if args.class_name == "all" else [args.class_name]
    classes = [c for c in classes if _has_checkpoint(args.models_dir, c)]

    position = {int(row): i for i, row in enumerate(subset)}
    n_classes = len(STUTTER_CLASSES)
    scores_matrix = np.full((len(subset), n_classes), np.nan)
    labels_matrix = np.zeros((len(subset), n_classes), dtype=int)

    per_class: Dict[str, Dict] = {}
    details: Dict[str, Dict] = {}
    contaminated: List[str] = []

    for class_name in classes:
        class_index = STUTTER_CLASSES.index(class_name)
        model_path = fp.find_checkpoint(class_name, args.models_dir)
        model, spec = load_checkpoint(model_path, device)

        if spec.get("format") == "legacy" and class_name not in contaminated:
            contaminated.append(class_name)

        # Features depend only on the mel config, so the three 128-band models
        # share a single extraction pass.
        cache = get_feature_cache(table, subset, spec["n_mels"], spec["n_fft"], spec["hop_length"], args.cache_dir)

        y_true, y_scores, rows = predict_scores(
            model,
            cache.dataset(table),
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            class_index=class_index,
        )
        cols = np.array([position[int(r)] for r in rows])
        scores_matrix[cols, class_index] = y_scores
        labels_matrix[cols, class_index] = y_true

        binary = M.compute_binary_metrics(y_true, y_scores, threshold=args.threshold)
        per_class[class_name] = binary

        M.print_binary_report(binary, class_name)
        print(f"  checkpoint: {os.path.basename(model_path)}  [{fp.describe_model(spec)}]")

        cm = M.confusion_matrix(y_true, (y_scores >= args.threshold).astype(int))
        M.print_confusion_matrix(cm)

        out_dir = os.path.join(args.output_dir, split.name)
        cm_path = os.path.join(out_dir, f"{class_name}_confusion_matrix.png")
        M.save_confusion_matrix_plot(
            cm, ["Not Present", "Present"], cm_path, title=f"{class_name} - {split.name} split"
        )

        # Metrics are hoisted to the top level of the entry so the saved report
        # is directly consumable.
        entry = {
            "class_name": class_name,
            "model_path": portable_path(model_path),
            "model": spec,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_png": cm_path,
            "threshold_sweep": None,
            "misclassified_json": None,
            **binary,
        }

        if args.sweep_thresholds:
            sweep = M.run_threshold_sweep(y_true, y_scores)
            entry["threshold_sweep"] = sweep
            print(
                f"  best F1 t={sweep['best_f1']['threshold']:.2f} (F1={sweep['best_f1']['f1']:.4f})   "
                f"best spec t={sweep['best_specificity']['threshold']:.2f} "
                f"(spec={sweep['best_specificity']['specificity']:.4f})   "
                f"best Youden t={sweep['best_youden']['threshold']:.2f}"
            )

        if args.save_misclassified:
            wrong = (y_scores >= args.threshold).astype(int) != y_true
            misclassified = [
                {
                    "clip": table.iloc[int(row)]["clip_id_str"],
                    "show_id": table.iloc[int(row)]["show_id"],
                    "episode": table.iloc[int(row)]["episode"],
                    "true": int(t),
                    "predicted": int(p),
                    "score": round(float(s), 6),
                }
                for row, t, p, s in zip(
                    rows[wrong], y_true[wrong], (y_scores >= args.threshold)[wrong], y_scores[wrong]
                )
            ]
            mis_path = os.path.join(out_dir, f"{class_name}_misclassified.json")
            os.makedirs(out_dir, exist_ok=True)
            with open(mis_path, "w") as f:
                json.dump(misclassified, f, indent=2)
            entry["misclassified_json"] = mis_path
            entry["num_misclassified"] = len(misclassified)
            print(f"  misclassified: {len(misclassified)} -> {mis_path}")

        details[class_name] = entry

    multilabel = None
    multilabel_path = None
    if len(classes) == n_classes and not np.isnan(scores_matrix).any():
        multilabel = M.compute_multilabel_metrics(labels_matrix, scores_matrix, threshold=args.threshold)
        M.print_multilabel_report(multilabel)
        multilabel_path = os.path.join(args.output_dir, split.name, "multilabel_report.json")

    block = {
        "split": split.to_dict(),
        "threshold": args.threshold,
        "subset": args.subset,
        "per_class": per_class,
        "details": details,
        "multilabel": multilabel,
        "multilabel_report": multilabel_path,
        "contamination": CONTAMINATION_LEGACY if contaminated else None,
        "contaminated_classes": contaminated,
    }
    return block


def _has_checkpoint(models_dir: str, class_name: str) -> bool:
    try:
        fp.find_checkpoint(class_name, models_dir)
        return True
    except FileNotFoundError:
        return False


def build_comparison(blocks: List[Dict]) -> Dict:
    """Per-class metric deltas between the first and second scored split.

    Makes the cost of speaker leakage explicit: the random split reuses
    recordings the model was fit on, the group split does not.
    """
    if len(blocks) < 2:
        return {}

    names = [b["split"]["name"] for b in blocks]
    per_class = {}
    for class_name in STUTTER_CLASSES:
        first, second = blocks[0]["per_class"].get(class_name), blocks[1]["per_class"].get(class_name)
        if not first or not second:
            continue
        per_class[class_name] = {
            "f1": {
                "optimistic": first["f1"],
                "conservative": second["f1"],
                "delta": round(first["f1"] - second["f1"], 4),
            },
            "auroc": {
                "optimistic": first["auroc"],
                "conservative": second["auroc"],
                "delta": round(first["auroc"] - second["auroc"], 4),
            },
            "accuracy": {
                "optimistic": first["accuracy"],
                "conservative": second["accuracy"],
                "delta": round(first["accuracy"] - second["accuracy"], 4),
            },
        }

    return {
        "splits": names,
        "note": f"delta = {names[0]} minus {names[1]}; a large positive delta indicates speaker leakage",
        "per_class": per_class,
    }


def format_comparison_table(comparison: Dict) -> str:
    """Markdown table for ``build_comparison`` output."""
    if not comparison:
        return ""
    optimistic, conservative = comparison["splits"]
    lines = [
        f"Positive delta means the {optimistic} split scores higher, i.e. the gap attributable to speaker leakage.\n",
        f"| Model | F1 ({optimistic}) | F1 ({conservative}) | ΔF1 | AUROC ({optimistic}) | "
        f"AUROC ({conservative}) | ΔAUROC |",
        "|---|---|---|---|---|---|---|",
    ]
    for class_name, values in comparison["per_class"].items():
        lines.append(
            f"| {class_name} | {values['f1']['optimistic']:.4f} | {values['f1']['conservative']:.4f} | "
            f"{values['f1']['delta']:+.4f} | {values['auroc']['optimistic']:.4f} | "
            f"{values['auroc']['conservative']:.4f} | {values['auroc']['delta']:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> Dict:
    args = parse_args(argv)
    device = get_device(args.device)

    print("\nDADS evaluation")
    print(f"  device:     {device}")
    print(f"  models_dir: {args.models_dir}")
    print(f"  threshold:  {args.threshold}")
    print(f"  subset:     {args.subset}")

    extra = [f.strip() for f in args.exclude_extra.split(",") if f.strip()]
    table = load_clip_table(
        labels_csv=args.labels_csv,
        waves_dir=args.waves_dir,
        clips_dir=args.clips_dir,
        extra_exclude_flags=extra,
    )

    split_names = ["random", "group"] if args.split == "both" else [args.split]
    blocks = []
    for name in split_names:
        block = evaluate_split(args, name, table, device)
        if block["contamination"]:
            print(f"\n  WARNING: {block['contamination']}")
        blocks.append(block)

    os.makedirs(args.output_dir, exist_ok=True)
    comparison = build_comparison(blocks)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "labels_csv": portable_path(args.labels_csv),
            "waves_dir": portable_path(args.waves_dir),
            "clips_used": len(table),
            "clips_total": table.attrs.get("total_clips"),
            "dropped": table.attrs.get("dropped", {}),
        },
        "models_dir": args.models_dir,
        "threshold": args.threshold,
        "args": {k: portable_path(v) if isinstance(v, str) and os.path.isabs(v) else v for k, v in vars(args).items()},
        "comparison": comparison,
        "splits": blocks,
    }

    summary_json = os.path.join(args.output_dir, "summary.json")
    M.save_report(summary, summary_json)

    for block in blocks:
        block_dir = os.path.join(args.output_dir, block["split"]["name"])
        os.makedirs(block_dir, exist_ok=True)
        for class_name, entry in block["details"].items():
            M.save_report(entry, os.path.join(block_dir, f"{class_name}_report.json"))
        if block.get("multilabel"):
            M.save_report(block["multilabel"], block["multilabel_report"])

    for block in blocks:
        print(f"\n{block['split']['name']} split")
        print("-" * 72)
        print(M.format_summary_table(block["per_class"], block["threshold"]))

    if comparison:
        print("\nSplit comparison (speaker leakage)")
        print("-" * 72)
        print(format_comparison_table(comparison))

    summary_md = M.write_summary(summary, args.output_dir)
    print(f"\n  Reports:   {summary_json}")
    print(f"  Summary:   {summary_md}")
    return summary


if __name__ == "__main__":
    main()
