"""Append-only JSONL diagnostics that cannot affect tracker decisions."""

import json
from pathlib import Path

import numpy as np


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class JsonlDiagnostics:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8", newline="\n")

    def write(self, payload):
        self._stream.write(json.dumps(
            _json_safe(payload), ensure_ascii=False, separators=(",", ":")) + "\n")
        self._stream.flush()

    def close(self):
        if not self._stream.closed:
            self._stream.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
