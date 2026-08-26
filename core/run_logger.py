"""
Append structured JSON lines for each pipeline run (audit / demos).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


def append_pipeline_run(log_dir: str, payload: dict[str, Any]) -> str | None:
    """
    Write one JSON object as a line to logs/runs/<date>.jsonl.

    Returns path written, or None on failure.
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
        day = time.strftime("%Y-%m-%d")
        path = os.path.join(log_dir, f"runs-{day}.jsonl")
        row = {"run_id": str(uuid.uuid4()), "ts": time.time(), **payload}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path
    except Exception as e:
        logger.warning("Structured run log failed: %s", e)
        return None
