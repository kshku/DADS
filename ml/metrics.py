"""Evaluation metrics for the DADS binary stutter classifiers.

Ported from the reference evaluation module (pure NumPy, no sklearn
dependency) and trimmed to what DADS needs: five independent binary models plus
a joint multi-label view across them. Localization metrics were dropped because
the DADS models classify a whole 3-second clip rather than a frame.
"""

import json
import os
from typing import Dict, List

import numpy as np

from ml.config import STUTTER_CLASSES

THRESHOLD_SWEEP = np.round(np.arange(0.1, 0.91, 0.05), 4)
"""Canonical threshold grid for sweep-based optimal-threshold search.

Rounding the grid keeps selected thresholds exact (``0.25`` rather than
``0.25000000000000006``), so registry and report values stay readable.

Shared by ``find_optimal_threshold`` and the printed sweep so the reported
optimum is always a row of the printed table.
"""


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate ``y`` over ``x`` with the trapezoidal rule.

    numpy < 2.0 exposes ``np.trapz``; numpy >= 2.0 renamed it to
    ``np.trapezoid``. This helper keeps the metrics working across both.
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


# ---------------------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------------------


def compute_binary_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Comprehensive binary classification metrics for one stutter model.

    Args:
        y_true: Ground truth binary labels (0/1), shape (N,).
        y_scores: Predicted probabilities, shape (N,).
        threshold: Decision threshold for the P/R/F1 rows.

    Returns:
        Threshold-independent ``auroc``/``auprc`` plus threshold-dependent
        precision/recall/f1/specificity/accuracy and the raw confusion counts.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    auroc = compute_auroc(y_true, y_scores)
    auprc = compute_auprc(y_true, y_scores)

    y_pred = (y_scores >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "threshold": float(threshold),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "false_positive_rate": round(fpr, 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
        "accuracy": round(accuracy, 4),
        "support": int(y_true.sum()),
        "num_samples": int(len(y_true)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area under the ROC curve via the trapezoidal rule."""
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    sorted_true = y_true[np.argsort(-y_scores)]

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = fp = 0

    for label in sorted_true:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    return abs(_trapezoid(tpr_list, fpr_list))


def compute_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area under the precision-recall curve (more informative when imbalanced)."""
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0

    sorted_true = y_true[np.argsort(-y_scores)]

    tp = fp = 0
    prec_list = [1.0]
    rec_list = [0.0]

    for label in sorted_true:
        if label == 1:
            tp += 1
        else:
            fp += 1
        prec_list.append(tp / (tp + fp))
        rec_list.append(tp / n_pos)

    return abs(_trapezoid(prec_list, rec_list))


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray, metric: str = "f1") -> tuple:
    """Return ``(best_threshold, best_value)`` on the canonical threshold grid.

    Args:
        metric: One of "f1", "specificity", "recall", "youden".
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    best_thresh = 0.5
    best_val = -float("inf")

    for thresh in THRESHOLD_SWEEP:
        y_pred = (y_scores >= thresh).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        values = {"f1": f1, "specificity": specificity, "recall": recall, "youden": recall + specificity - 1.0}
        val = values.get(metric, f1)

        if val > best_val:
            best_val = val
            best_thresh = float(thresh)

    return best_thresh, float(best_val)


def run_threshold_sweep(y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
    """Full threshold table plus the optimal point for each selection metric.

    ``auroc``/``auprc`` are included because they do not depend on the threshold,
    so a caller sweeping for one has no other way to see them.
    """
    sweep = []
    for t in THRESHOLD_SWEEP:
        m = compute_binary_metrics(y_true, y_scores, threshold=t)
        sweep.append(
            {
                "threshold": float(t),
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "specificity": m["specificity"],
                "accuracy": m["accuracy"],
            }
        )

    reference = compute_binary_metrics(y_true, y_scores)
    return {
        "auroc": reference["auroc"],
        "auprc": reference["auprc"],
        "sweep": sweep,
        "best_f1": dict(zip(("threshold", "f1"), find_optimal_threshold(y_true, y_scores, "f1"))),
        "best_specificity": dict(
            zip(("threshold", "specificity"), find_optimal_threshold(y_true, y_scores, "specificity"))
        ),
        "best_youden": dict(zip(("threshold", "youden"), find_optimal_threshold(y_true, y_scores, "youden"))),
    }


# ---------------------------------------------------------------------------
# Joint multi-label view across the five models
# ---------------------------------------------------------------------------


def compute_multilabel_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.4) -> Dict:
    """Joint metrics treating the five classifiers as one multi-label predictor.

    Args:
        y_true: (N, C) binary ground truth.
        y_scores: (N, C) predicted probabilities.
        threshold: Decision threshold applied to every column.

    Returns:
        Exact-match accuracy, Hamming loss, macro F1/AUROC and per-class
        AUROC — the "all five at once" view that ``Model/model_train.ipynb``
        reports after training.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_scores = y_scores.reshape(-1, 1)

    y_pred = (y_scores >= threshold).astype(int)
    n_samples, n_classes = y_true.shape

    exact_match = float(np.mean(np.all(y_true == y_pred, axis=1))) if n_samples else 0.0
    hamming_loss = float(np.mean(y_true != y_pred)) if n_samples else 0.0

    per_class = {}
    f1s, aurocs = [], []
    names = list(STUTTER_CLASSES[:n_classes]) + [f"class_{i}" for i in range(len(STUTTER_CLASSES), n_classes)]
    for c in range(n_classes):
        m = compute_binary_metrics(y_true[:, c], y_scores[:, c], threshold=threshold)
        per_class[names[c]] = m
        f1s.append(m["f1"])
        aurocs.append(m["auroc"])

    return {
        "threshold": float(threshold),
        "num_samples": int(n_samples),
        "num_classes": int(n_classes),
        "exact_match_accuracy": round(exact_match, 4),
        "hamming_loss": round(hamming_loss, 4),
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "macro_auroc": round(float(np.mean(aurocs)), 4) if aurocs else 0.0,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """Confusion matrix where entry ``[i, j]`` counts true class i predicted as j."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Render a confusion matrix to a PNG file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def save_report(report: dict, output_path: str) -> None:
    """Write an evaluation report as JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=float)


def print_binary_report(metrics: Dict, class_name: str = "") -> None:
    """Print a human-readable binary classifier report."""
    header = f"  {class_name.upper()}" if class_name else "  Binary Report"
    print(f"\n{header}")
    print("  " + "=" * 62)
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")
    print(f"  Threshold:   {metrics['threshold']:.2f}")
    print("  " + "-" * 62)
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Balanced acc:{metrics['balanced_accuracy']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Support:     {metrics['support']} / {metrics['num_samples']}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")


def print_confusion_matrix(cm: np.ndarray) -> None:
    """Print a 2x2 confusion matrix as text."""
    print("                    pred_neg  pred_pos")
    print(f"  true_neg          {cm[0, 0]:>6d}    {cm[0, 1]:>6d}")
    print(f"  true_pos          {cm[1, 0]:>6d}    {cm[1, 1]:>6d}")


def print_multilabel_report(metrics: Dict) -> None:
    """Print the joint five-model view."""
    print("\n  Joint multi-label view (all models together)")
    print("  " + "=" * 62)
    print(f"  Exact-match accuracy: {metrics['exact_match_accuracy']:.4f}")
    print(f"  Hamming loss:         {metrics['hamming_loss']:.4f}")
    print(f"  Macro F1:             {metrics['macro_f1']:.4f}")
    print(f"  Macro AUROC:          {metrics['macro_auroc']:.4f}")


def format_summary_table(per_class: Dict[str, Dict], threshold: float) -> str:
    """Markdown table of the per-model results, ordered by STUTTER_CLASSES."""
    header = (
        "| Model | AUROC | AUPRC | Precision | Recall | F1 | Specificity | Balanced Acc | "
        "Accuracy | Positives | n |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for name in STUTTER_CLASSES:
        m = per_class.get(name)
        if not m:
            continue
        rows.append(
            f"| {name} | {m['auroc']:.4f} | {m['auprc']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m['specificity']:.4f} | "
            f"{m['balanced_accuracy']:.4f} | {m['accuracy']:.4f} | {m['support']} | {m['num_samples']} |"
        )
    if rows:
        macro_f1 = float(np.mean([per_class[n]["f1"] for n in per_class]))
        macro_auroc = float(np.mean([per_class[n]["auroc"] for n in per_class]))
        rows.append(f"| **macro** | {macro_auroc:.4f} | — | — | — | **{macro_f1:.4f}** | — | — | — | — | — |")
    return f"Threshold: {threshold:.2f}\n\n" + header + "\n".join(rows) + "\n"


def _optimal_threshold_table(details: Dict[str, Dict]) -> str:
    """Per-class best achievable F1 and the threshold that reaches it.

    Models here are trained with plain BCE and end up poorly calibrated, so the
    fixed 0.4 operating point is often well away from the best one. This table
    separates ranking quality from calibration.
    """
    if not details:
        return ""
    rows = ["| Model | F1 @ 0.4 | Best F1 | Best threshold | ΔF1 |", "|---|---|---|---|---|"]
    for class_name, detail in details.items():
        sweep = (detail.get("threshold_sweep") or {}).get("best_f1") or {}
        best_f1 = sweep.get("f1")
        best_threshold = sweep.get("threshold")
        if best_f1 is None:
            continue
        at_default = detail.get("f1", 0.0)
        rows.append(
            f"| {class_name} | {at_default:.4f} | {best_f1:.4f} | {best_threshold:.2f} | {best_f1 - at_default:+.4f} |"
        )
    return "\n".join(rows) + "\n\n" if len(rows) > 2 else ""


def write_summary(summary: dict, output_dir: str) -> str:
    """Write SUMMARY.md next to the JSON reports and return its path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "SUMMARY.md")

    lines = ["# DADS model evaluation", ""]
    if summary.get("generated_at"):
        lines += [f"Generated: {summary['generated_at']}", ""]
    if summary.get("dataset"):
        lines += [f"Dataset: {summary['dataset']}", ""]
    if summary.get("threshold") is not None:
        lines += [f"Decision threshold: {summary['threshold']}", ""]

    for block in summary.get("splits", []):
        split_meta = block.get("split", {})
        lines += [f"## Split: {split_meta.get('name', '?')}", ""]
        if split_meta:
            lines += ["```", str(split_meta), "```", ""]
        if block.get("contamination"):
            lines += [f"> **{block['contamination']}**", ""]
        lines.append(format_summary_table(block["per_class"], block.get("threshold", summary.get("threshold", 0.4))))
        lines.append(_optimal_threshold_table(block["details"]))
        if block.get("multilabel"):
            ml = block["multilabel"]
            lines += [
                f"Joint view — exact-match accuracy: **{ml['exact_match_accuracy']:.4f}**, "
                f"Hamming loss: **{ml['hamming_loss']:.4f}**",
                "",
            ]

    comparison = summary.get("comparison")
    if comparison:
        names = comparison["splits"]
        lines += ["## Split comparison", "", comparison["note"], ""]
        lines += [
            f"| Model | F1 ({names[0]}) | F1 ({names[1]}) | dF1 | AUROC ({names[0]}) | AUROC ({names[1]}) | dAUROC |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, values in comparison["per_class"].items():
            lines.append(
                f"| {name} | {values['f1']['optimistic']:.4f} | {values['f1']['conservative']:.4f} | "
                f"{values['f1']['delta']:+.4f} | {values['auroc']['optimistic']:.4f} | "
                f"{values['auroc']['conservative']:.4f} | {values['auroc']['delta']:+.4f} |"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
