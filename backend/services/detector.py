"""Service wrapper for StutterDetector from App/connector.py"""

import os
import sys

# Add App/ to path so we can import connector
APP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "App")
sys.path.insert(0, os.path.abspath(APP_DIR))

from connector import StutterDetector  # noqa: E402

_detector = None


def get_detector() -> StutterDetector:
    """Get or initialize the singleton StutterDetector."""
    global _detector
    if _detector is None:
        _detector = StutterDetector(detection_threshold=0.4)
    return _detector
