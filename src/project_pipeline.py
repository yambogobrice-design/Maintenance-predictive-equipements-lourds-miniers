"""End-to-end project orchestration used by notebooks and the dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import build_demo_machine_snapshot, prepare_raw_datasets
from .evaluation import evaluate_classifier, evaluate_regressor
from .feature_engineering import build_feature_table, get_feature_columns, temporal_train_test_split
from .models import reduce_training_frame, save_model_artifacts, train_classifiers, train_rul_models
from .utils import ensure_directory, save_dataframe, save_json


def _pick_best_classifier(metrics: dict[str, dict[str, float]]) -> str:
    """Select the most useful alerting model from exported classification metrics."""

    candidates = [name for name in metrics if name != "dummy"]
    if not candidates:
        return next(iter(metrics))
    return max(
        candidates,
        key=lambda name: (
            metrics[name].get("f1_failure", 0.0),
            metrics[name].get("recall_failure", 0.0),
            metrics[name].get("average_precision", 0.0),
        ),
    )


def _pick_best_regressor(metrics: dict[str, dict[str, float]]) -> str:
    """Select the RUL model with the lowest MAE, excluding the dummy baseline."""

    candidates = [name for name in metrics if name != "dummy_rul"]
    if not candidates:
        return next(iter(metrics))
    return min(candidates, key=lambda name: metrics[name].get("mae", float("inf")))


def run_full_training(project_root: str | Path) -> dict[str, object]:
    """Execute the full M3 pipeline and persist the main artifacts."""

    project_root = Path(project_root)
    bundle = prepare_raw_datasets(project_root)
    features = build_feature_table(bundle, project_root)
    train_df, test_df = temporal_train_test_split(features)
    feature_columns = get_feature_columns(features)

    classifier_artifacts = train_classifiers(train_df)
    regressor_artifacts = train_rul_models(train_df)

    model_dir = ensure_directory(project_root / "models")
    save_model_artifacts({**classifier_artifacts, **regressor_artifacts}, model_dir)

    metrics_dir = ensure_directory(project_root / "data" / "processed")
    classifier_metrics = {}
    regressor_metrics = {}
    prediction_frames: list[pd.DataFrame] = []

    for name, artifact in classifier_artifacts.items():
        metrics, preds = evaluate_classifier(artifact, test_df, feature_columns)
        classifier_metrics[name] = metrics
        prediction_frames.append(preds.assign(model_name=name))

    for name, artifact in regressor_artifacts.items():
        metrics, preds = evaluate_regressor(artifact, test_df, feature_columns)
        regressor_metrics[name] = metrics
        prediction_frames.append(preds.assign(model_name=name))

    best_classifier_name = _pick_best_classifier(classifier_metrics)
    best_regressor_name = _pick_best_regressor(regressor_metrics)
    best_classifier = classifier_artifacts[best_classifier_name]
    best_regressor = regressor_artifacts[best_regressor_name]

    combined_predictions = features.copy()
    all_features = combined_predictions[feature_columns]
    combined_predictions["predicted_failure_probability"] = best_classifier.pipeline.predict_proba(all_features)[:, 1]
    threshold = best_classifier.threshold if best_classifier.threshold is not None else 0.5
    combined_predictions["predicted_failure_label"] = (
        combined_predictions["predicted_failure_probability"] >= threshold
    ).astype(int)
    combined_predictions["predicted_rul"] = best_regressor.pipeline.predict(all_features)
    combined_predictions["predicted_rul"] = combined_predictions["predicted_rul"].clip(lower=0)
    combined_predictions["selected_classifier"] = best_classifier_name
    combined_predictions["selected_regressor"] = best_regressor_name
    combined_predictions["alert_threshold"] = float(threshold)

    save_dataframe(combined_predictions, metrics_dir / "fleet_predictions.csv")
    save_dataframe(build_demo_machine_snapshot(combined_predictions), metrics_dir / "fleet_snapshot.csv")
    save_json({"classification": classifier_metrics, "regression": regressor_metrics}, metrics_dir / "metrics_summary.json")
    save_dataframe(pd.concat(prediction_frames, ignore_index=True, sort=False), metrics_dir / "all_predictions.csv")

    return {
        "bundle": bundle,
        "features": features,
        "train_df": train_df,
        "test_df": test_df,
        "feature_columns": feature_columns,
        "classifier_artifacts": classifier_artifacts,
        "regressor_artifacts": regressor_artifacts,
        "classifier_metrics": classifier_metrics,
        "regressor_metrics": regressor_metrics,
        "combined_predictions": combined_predictions,
    }
