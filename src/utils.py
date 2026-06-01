"""Utility helpers shared across the M3 predictive maintenance project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RANDOM_STATE = 42
PLOT_STYLE = "seaborn-v0_8"
COLOR_PALETTE = {
    "bg": "#0B132B",
    "panel": "#1C2541",
    "accent": "#FF9F1C",
    "success": "#2EC4B6",
    "danger": "#E71D36",
    "info": "#5BC0EB",
    "neutral": "#E0E1DD",
}
SENSOR_COLUMNS = ["volt", "rotate", "pressure", "vibration"]
ROLLING_WINDOWS = [3, 6, 24]
LAG_STEPS = [1, 2, 6, 24]
FAILURE_HORIZON_HOURS = 24
CRITICAL_RUL_THRESHOLD = 20.0


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def format_metric(value: float) -> str:
    """Format numeric metrics with four decimals."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{value:.4f}"


def save_json(payload: dict[str, Any], path: Path | str) -> None:
    """Persist a JSON payload with deterministic formatting."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: Path | str) -> None:
    """Persist a dataframe as CSV."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute an asymmetric NASA-style score for RUL prediction."""

    errors = np.asarray(y_pred) - np.asarray(y_true)
    penalties = np.where(
        errors < 0,
        np.exp(np.abs(errors) / 13.0) - 1.0,
        np.exp(np.abs(errors) / 10.0) - 1.0,
    )
    return float(np.mean(penalties))


def flatten_classification_report(report: dict[str, Any]) -> dict[str, float]:
    """Flatten a sklearn classification report for JSON export."""

    flattened: dict[str, float] = {}
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flattened[f"{key}_{sub_key}"] = float(sub_value)
        else:
            flattened[key] = float(value)
    return flattened
