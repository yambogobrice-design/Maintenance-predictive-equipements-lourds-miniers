"""Evaluation helpers for classification and RUL regression."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    average_precision_score,
)

from .utils import flatten_classification_report, nasa_score


def evaluate_classifier(model_artifact: Any, test_df: pd.DataFrame, feature_columns: list[str]) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a classifier on the chronologically held-out test set."""

    X_test = test_df[feature_columns]
    y_test = test_df["failure_within_24h"].to_numpy()
    probabilities = model_artifact.pipeline.predict_proba(X_test)[:, 1]
    threshold = model_artifact.threshold if model_artifact.threshold is not None else 0.5
    predictions = (probabilities >= threshold).astype(int)

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = flatten_classification_report(report)
    metrics.update(
        {
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "precision_failure": float(precision_score(y_test, predictions, zero_division=0)),
            "recall_failure": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_failure": float(f1_score(y_test, predictions, zero_division=0)),
            "threshold": float(threshold),
        }
    )

    cm = confusion_matrix(y_test, predictions)
    prediction_frame = test_df[["datetime", "machineID", "rul_hours", "failure_within_24h"]].copy()
    prediction_frame["predicted_failure_probability"] = probabilities
    prediction_frame["predicted_failure_label"] = predictions
    prediction_frame["cm_tn"] = cm[0, 0]
    prediction_frame["cm_fp"] = cm[0, 1]
    prediction_frame["cm_fn"] = cm[1, 0]
    prediction_frame["cm_tp"] = cm[1, 1]
    return metrics, prediction_frame


def evaluate_regressor(model_artifact: Any, test_df: pd.DataFrame, feature_columns: list[str]) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate an RUL regressor on the held-out test set."""

    rul_test = test_df[test_df["has_known_rul"] == 1].copy()
    X_test = rul_test[feature_columns]
    y_test = rul_test["rul_hours"].to_numpy()
    predictions = model_artifact.pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "nasa_score": float(nasa_score(y_test, predictions)),
        "critical_zone_mae": float(
            mean_absolute_error(
                y_test[y_test < 20],
                predictions[y_test < 20],
            )
        )
        if np.any(y_test < 20)
        else float("nan"),
    }
    prediction_frame = rul_test[["datetime", "machineID", "rul_hours"]].copy()
    prediction_frame["predicted_rul"] = predictions
    prediction_frame["residual"] = prediction_frame["predicted_rul"] - prediction_frame["rul_hours"]
    return metrics, prediction_frame
