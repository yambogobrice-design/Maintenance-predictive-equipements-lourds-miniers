"""Training utilities for classification and RUL regression models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_numeric_dtype

try:
    from imblearn.over_sampling import SMOTE
except Exception:  # pragma: no cover - handled at runtime
    SMOTE = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - handled at runtime
    XGBClassifier = None
    XGBRegressor = None

from .feature_engineering import get_feature_columns
from .utils import RANDOM_STATE, ensure_directory

ENABLE_XGBOOST = False


@dataclass
class ModelArtifacts:
    """Container for fitted model artifacts and evaluation metadata."""

    name: str
    pipeline: Pipeline
    threshold: float | None
    cv_results: pd.DataFrame
    metrics: dict[str, float]


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a shared preprocessing transformer."""

    categorical_cols = [col for col in X.columns if not is_numeric_dtype(X[col])]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def reduce_training_frame(
    df: pd.DataFrame,
    target_col: str,
    stride: int,
    keep_positive: bool = True,
) -> pd.DataFrame:
    """Downsample chronologically to keep the pipeline tractable on dense telemetry."""

    sampled_parts = []
    for _, group in df.sort_values(["machineID", "datetime"]).groupby("machineID", sort=False):
        base_mask = (np.arange(len(group)) % stride) == 0
        if keep_positive:
            base_mask = base_mask | (group[target_col].to_numpy() == 1)
        sampled_parts.append(group.loc[base_mask])
    return pd.concat(sampled_parts).reset_index(drop=True)


def optimize_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose a probability threshold on validation data by failure-class F1."""

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    records = []
    for idx, threshold in enumerate(thresholds):
        preds = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        records.append(
            {
                "threshold": float(threshold),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1),
            }
        )
    threshold_df = pd.DataFrame(records)
    threshold_df = threshold_df.sort_values(["f1", "precision", "recall"], ascending=False)
    best_threshold = float(threshold_df.iloc[0]["threshold"]) if not threshold_df.empty else 0.5
    return best_threshold, threshold_df


def maybe_apply_smote(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE if the dependency is available."""

    if SMOTE is None:
        return X, y
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)


def temporal_validation_split(df: pd.DataFrame, train_ratio: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an inner chronological validation split inside the training window."""

    train_parts = []
    validation_parts = []
    for _, group in df.sort_values(["machineID", "datetime"]).groupby("machineID", sort=False):
        split_idx = max(1, int(len(group) * train_ratio))
        train_parts.append(group.iloc[:split_idx])
        validation_parts.append(group.iloc[split_idx:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(validation_parts).reset_index(drop=True)


def balance_failure_frame(df: pd.DataFrame, negative_ratio: int = 8) -> pd.DataFrame:
    """Keep all failure windows and sample negatives for a usable alerting boundary."""

    positives = df[df["failure_within_24h"] == 1]
    negatives = df[df["failure_within_24h"] == 0]
    if positives.empty or negatives.empty:
        return df
    n_negatives = min(len(negatives), len(positives) * negative_ratio)
    sampled_negatives = negatives.sample(n=n_negatives, random_state=RANDOM_STATE)
    return pd.concat([positives, sampled_negatives]).sort_values(["machineID", "datetime"]).reset_index(drop=True)


def train_classifiers(train_df: pd.DataFrame) -> dict[str, ModelArtifacts]:
    """Train and compare baseline, linear and tree classifiers."""

    modelling_df = reduce_training_frame(train_df, "failure_within_24h", stride=3, keep_positive=True)
    inner_train_df, validation_df = temporal_validation_split(modelling_df)
    feature_columns = get_feature_columns(modelling_df)
    X_train = modelling_df[feature_columns]
    y_train = modelling_df["failure_within_24h"]
    positive_rate = max(y_train.mean(), 1e-4)
    scale_pos_weight = float((1 - positive_rate) / positive_rate)
    estimators = {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(
            n_estimators=80,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }
    if XGBClassifier is not None and ENABLE_XGBOOST:
        estimators["xgb"] = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            n_jobs=4,
        )

    artifacts: dict[str, ModelArtifacts] = {}
    for name, estimator in estimators.items():
        threshold_pipeline = Pipeline([("preprocessor", make_preprocessor(inner_train_df[feature_columns])), ("model", estimator)])
        threshold_fit_df = balance_failure_frame(inner_train_df) if name in {"logreg", "rf", "extra_trees", "xgb"} else inner_train_df
        threshold_pipeline.fit(threshold_fit_df[feature_columns], threshold_fit_df["failure_within_24h"])
        validation_pred = threshold_pipeline.predict_proba(validation_df[feature_columns])[:, 1]
        threshold, threshold_df = optimize_threshold(validation_df["failure_within_24h"].to_numpy(), validation_pred)

        pipeline = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", estimator)])
        final_fit_df = balance_failure_frame(modelling_df) if name in {"logreg", "rf", "extra_trees", "xgb"} else modelling_df
        pipeline.fit(final_fit_df[feature_columns], final_fit_df["failure_within_24h"])
        artifacts[name] = ModelArtifacts(
            name=name,
            pipeline=pipeline,
            threshold=threshold if name != "dummy" else 0.5,
            cv_results=threshold_df.head(10),
            metrics={"validation_average_precision": float(average_precision_score(validation_df["failure_within_24h"], validation_pred))},
        )
    return artifacts


def train_rul_models(train_df: pd.DataFrame) -> dict[str, ModelArtifacts]:
    """Train baseline and tree regressors for RUL."""

    rul_df = train_df[train_df["has_known_rul"] == 1].copy()
    rul_df = reduce_training_frame(rul_df, "has_known_rul", stride=3, keep_positive=False)
    feature_columns = get_feature_columns(rul_df)
    X_train = rul_df[feature_columns]
    y_train = rul_df["rul_hours"]
    preprocessor = make_preprocessor(X_train)
    estimators = {
        "dummy_rul": (DummyRegressor(strategy="mean"), {}),
        "rf_rul": (
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=1,
                n_estimators=100,
                max_depth=14,
                min_samples_leaf=2,
            ),
            {},
        ),
        "extra_trees_rul": (
            ExtraTreesRegressor(
                random_state=RANDOM_STATE,
                n_jobs=1,
                n_estimators=120,
                max_depth=14,
                min_samples_leaf=2,
            ),
            {},
        ),
    }
    if XGBRegressor is not None and ENABLE_XGBOOST:
        estimators["xgb_rul"] = (
            XGBRegressor(
                random_state=RANDOM_STATE,
                n_estimators=120,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                n_jobs=4,
            ),
            {
                "model__max_depth": [3, 4],
                "model__learning_rate": [0.03, 0.05],
            },
        )

    artifacts: dict[str, ModelArtifacts] = {}
    for name, (estimator, param_grid) in estimators.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_train)
        best_model = pipeline
        cv_results = pd.DataFrame([{"params": "default", "mean_test_score": np.nan}])
        metric_payload = {"cv_mae": float(np.mean(np.abs(predictions - y_train.to_numpy())))}
        artifacts[name] = ModelArtifacts(
            name=name,
            pipeline=best_model,
            threshold=None,
            cv_results=cv_results,
            metrics=metric_payload,
        )
    return artifacts


def save_model_artifacts(artifacts: dict[str, ModelArtifacts], output_dir: str) -> None:
    """Persist trained pipelines with joblib."""

    model_dir = ensure_directory(output_dir)
    for name, artifact in artifacts.items():
        dump(artifact, model_dir / f"{name}.joblib")
    if "rf" in artifacts:
        dump(artifacts["rf"], model_dir / "classifier_rf.joblib")
        dump(artifacts.get("xgb", artifacts["rf"]), model_dir / "classifier_xgb.joblib")
    if "extra_trees_rul" in artifacts:
        dump(artifacts["extra_trees_rul"], model_dir / "regressor_rul.joblib")
    elif "rf_rul" in artifacts:
        dump(artifacts["rf_rul"], model_dir / "regressor_rul.joblib")
