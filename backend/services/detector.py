"""Service wrapper for StutterDetector from shared/connector.py"""

import os
import sys

# Add project root to path so we can import shared.connector
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

from shared.connector import StutterDetector  # noqa: E402

_detector = None


def get_detector() -> StutterDetector:
    """Get or initialize the singleton StutterDetector."""
    global _detector
    if _detector is None:
        _detector = StutterDetector(detection_threshold=0.4)
    return _detector
