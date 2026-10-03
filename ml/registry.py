"""Model registry: checkpoint paths and decision thresholds for each detector.

A registry entry pairs the five checkpoints of one detector with the per-class
decision thresholds tuned for those checkpoints. Keeping them together matters:
a threshold only means anything relative to the weights it was calibrated
against, so the two must never be mixed across entries.

``shared.connector.StutterDetector`` reads this file at inference time and
``python -m ml.calibrate`` writes to it.
"""

import json
import os
from typing import Dict, Optional

from ml.config import PROJECT_ROOT, STUTTER_CLASSES

REGISTRY_PATH = os.path.join(PROJECT_ROOT, "Model", "registry.json")

# Used when a class is absent from the registry, so a partial registry still runs.
FALLBACK_THRESHOLD = 0.4

_THRESHOLD_CACHE: Dict[str, Dict[str, float]] = {}


def load_registry(path: str = REGISTRY_PATH) -> Dict:
    """Read the registry, returning an empty skeleton when the file is absent."""
    if not os.path.exists(path):
        return {"defaults": {"detector": "production"}, "detectors": {}}
    with open(path) as f:
        return json.load(f)


def save_registry(registry: Dict, path: str = REGISTRY_PATH) -> str:
    """Write the registry, creating its directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")
    return path


def get_detector_name(registry: Dict, name: Optional[str] = None) -> str:
    """Resolve which detector entry to use: explicit name, else the default."""
    if name:
        return name
    return registry.get("defaults", {}).get("detector", "production")


def get_entry(registry: Dict, name: Optional[str] = None) -> Dict:
    """Return one detector entry, raising if it is not registered."""
    detector = get_detector_name(registry, name)
    entry = registry.get("detectors", {}).get(detector)
    if entry is None:
        available = sorted(registry.get("detectors", {}))
        raise KeyError(f"Detector '{detector}' is not in the registry. Available: {available}")
    return entry


def resolve_path(path: str, root: str = PROJECT_ROOT) -> str:
    """Resolve a registry-relative path against the project root."""
    return path if os.path.isabs(path) else os.path.join(root, path)


def relative_path(path: str, root: str = PROJECT_ROOT) -> str:
    """Express a path relative to the project root, so the registry stays portable."""
    absolute = os.path.abspath(path)
    if os.path.commonpath([absolute, os.path.abspath(root)]) == os.path.abspath(root):
        return os.path.relpath(absolute, root)
    return absolute


def ensure_entry(registry: Dict, name: str, models_dir: str) -> Dict:
    """Fetch a detector entry, creating a skeleton if it does not exist yet."""
    return registry.setdefault("detectors", {}).setdefault(
        name, {"models_dir": relative_path(models_dir), "thresholds": {}}
    )


def model_path(registry: Dict, class_name: str, name: Optional[str] = None) -> str:
    """Absolute checkpoint path for one class of one detector."""
    entry = get_entry(registry, name)
    files = entry.get("model_files", {})
    if class_name in files:
        return resolve_path(files[class_name])
    # Fall back to convention-named files inside the entry's models_dir.
    return resolve_path(os.path.join(entry.get("models_dir", ""), ""))


def thresholds(registry: Dict, name: Optional[str] = None) -> Dict[str, float]:
    """Per-class decision thresholds, defaulted for any class not registered."""
    entry = get_entry(registry, name)
    configured = entry.get("thresholds", {})
    return {c: float(configured.get(c, FALLBACK_THRESHOLD)) for c in STUTTER_CLASSES}


def default_thresholds(detector: Optional[str] = None) -> Dict[str, float]:
    """Thresholds for the active registry entry, cached per process.

    The UI re-reads these inside per-chunk loops, and re-reading the JSON each
    time would be wasteful; the file is small and effectively static at runtime.
    """
    cache_key = detector or "<default>"
    if cache_key not in _THRESHOLD_CACHE:
        _THRESHOLD_CACHE[cache_key] = thresholds(load_registry(), detector)
    return _THRESHOLD_CACHE[cache_key]


def set_thresholds(registry: Dict, class_name: str, value: float, name: Optional[str] = None) -> None:
    """Record one class's threshold, creating the entry skeleton if needed."""
    detector = get_detector_name(registry, name)
    entry = registry.setdefault("detectors", {}).setdefault(detector, {})
    entry.setdefault("thresholds", {})[class_name] = round(float(value), 4)


def describe(registry: Dict, name: Optional[str] = None) -> str:
    """Human-readable dump of one detector entry, for the CLI."""
    detector = get_detector_name(registry, name)
    try:
        entry = get_entry(registry, detector)
    except KeyError:
        return f"no detector registered as '{detector}'"
    lines = [f"detector: {detector}", f"  models_dir: {entry.get('models_dir', '(per-class files)')}"]
    files = entry.get("model_files", {})
    for class_name in STUTTER_CLASSES:
        threshold = entry.get("thresholds", {}).get(class_name)
        shown = f"{threshold}" if threshold is not None else "(unset)"
        path = files.get(class_name, "")
        lines.append(f"  {class_name:13s} threshold={shown:8s} {os.path.basename(path)}")
    if entry.get("calibration"):
        lines.append(f"  calibrated: {entry['calibration']}")
    return "\n".join(lines)
