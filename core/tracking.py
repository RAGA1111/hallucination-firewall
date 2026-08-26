"""
Append aggregate metrics over time (eval / pipeline runs).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


def append_tracking_row(path: str, row: dict[str, Any]) -> None:
    """Append one JSON object per line."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {"ts": time.time(), **row}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Tracking append failed (%s): %s", path, e)
