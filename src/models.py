"""Training utilities for classification and RUL regression models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR

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

    categorical_cols = [col for col in X.columns if str(X[col].dtype) in {"category", "object"}]
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
    """Choose a probability threshold that favors recall while maximizing F1."""

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    records = []
    for idx, threshold in enumerate(thresholds):
        preds = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        weighted_score = 0.7 * recall[idx] + 0.3 * precision[idx]
        records.append(
            {
                "threshold": float(threshold),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1),
                "weighted_score": float(weighted_score),
            }
        )
    threshold_df = pd.DataFrame(records)
    threshold_df = threshold_df.sort_values(["f1", "recall", "precision"], ascending=False)
    best_threshold = float(threshold_df.iloc[0]["threshold"]) if not threshold_df.empty else 0.5
    return best_threshold, threshold_df


def maybe_apply_smote(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE if the dependency is available."""

    if SMOTE is None:
        return X, y
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)


def train_classifiers(train_df: pd.DataFrame) -> dict[str, ModelArtifacts]:
    """Train and compare baseline, Random Forest and XGBoost classifiers."""

    modelling_df = reduce_training_frame(train_df, "failure_within_24h", stride=6, keep_positive=True)
    feature_columns = get_feature_columns(modelling_df)
    X_train = modelling_df[feature_columns]
    y_train = modelling_df["failure_within_24h"]
    preprocessor = make_preprocessor(X_train)
    positive_rate = max(y_train.mean(), 1e-4)
    scale_pos_weight = float((1 - positive_rate) / positive_rate)
    estimators = {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(
            n_estimators=30,
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
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        if name in {"logreg", "rf", "xgb"} and name != "dummy":
            try:
                X_balanced, y_balanced = maybe_apply_smote(X_train, y_train)
            except Exception:
                X_balanced, y_balanced = X_train, y_train
        else:
            X_balanced, y_balanced = X_train, y_train

        pipeline.fit(X_balanced, y_balanced)
        oof_pred = pipeline.predict_proba(X_train)[:, 1]
        pipeline.fit(X_train, y_train)
        threshold, threshold_df = optimize_threshold(y_train.to_numpy(), oof_pred)
        artifacts[name] = ModelArtifacts(
            name=name,
            pipeline=pipeline,
            threshold=threshold if name != "dummy" else 0.5,
            cv_results=threshold_df.head(10),
            metrics={"cv_average_precision": float(average_precision_score(y_train, oof_pred))},
        )
    return artifacts


def train_rul_models(train_df: pd.DataFrame) -> dict[str, ModelArtifacts]:
    """Train Random Forest, XGBoost and SVR regressors for RUL."""

    rul_df = train_df[train_df["has_known_rul"] == 1].copy()
    rul_df = reduce_training_frame(rul_df, "has_known_rul", stride=6, keep_positive=False)
    feature_columns = get_feature_columns(rul_df)
    X_train = rul_df[feature_columns]
    y_train = rul_df["rul_hours"]
    preprocessor = make_preprocessor(X_train)
    estimators = {
        "dummy_rul": (DummyRegressor(strategy="mean"), {}),
        "rf_rul": (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, n_estimators=40, max_depth=8), {}),
        "svr_rul": (SVR(kernel="rbf", C=5, epsilon=0.1, gamma="scale"), {}),
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
    if "rf" in artifacts:
        dump(artifacts["rf"], model_dir / "classifier_rf.joblib")
        dump(artifacts.get("xgb", artifacts["rf"]), model_dir / "classifier_xgb.joblib")
    if "rf_rul" in artifacts:
        dump(artifacts["rf_rul"], model_dir / "regressor_rul.joblib")
