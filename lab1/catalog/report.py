"""Запись прогонов в lab1/results."""

import json
from pathlib import Path


def write_report(filename: str, payload: dict) -> Path:
    directory = Path(__file__).resolve().parents[1] / "results"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
