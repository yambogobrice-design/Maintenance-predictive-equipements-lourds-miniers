"""Feature engineering pipeline for predictive maintenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import DatasetBundle
from .utils import FAILURE_HORIZON_HOURS, LAG_STEPS, ROLLING_WINDOWS, SENSOR_COLUMNS, ensure_directory, save_dataframe


def _add_error_counts(telemetry: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    error_counts = (
        errors.assign(error_flag=1)
        .groupby(["machineID", "datetime"], as_index=False)["error_flag"]
        .sum()
        .rename(columns={"error_flag": "error_count"})
    )
    merged = telemetry.merge(error_counts, on=["machineID", "datetime"], how="left")
    merged["error_count"] = merged["error_count"].fillna(0)
    merged["error_count_cumulative"] = merged.groupby("machineID")["error_count"].cumsum()
    return merged


def _add_failure_targets(df: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    failures = failures.sort_values(["machineID", "datetime"]).copy()
    failure_times = failures.groupby("machineID")["datetime"].agg(list).to_dict()
    failure_types = failures.groupby("machineID")["failure"].agg(list).to_dict()

    next_failure_times = []
    next_failure_types = []
    rul_hours = []

    for machine_id, group in df.groupby("machineID", sort=False):
        group_times = group["datetime"].to_numpy(dtype="datetime64[ns]")
        machine_failures = np.array(failure_times.get(machine_id, []), dtype="datetime64[ns]")
        machine_types = np.array(failure_types.get(machine_id, []), dtype=object)
        if machine_failures.size == 0:
            next_failure_times.extend([pd.NaT] * len(group))
            next_failure_types.extend(["none"] * len(group))
            rul_hours.extend([np.nan] * len(group))
            continue

        idxs = np.searchsorted(machine_failures, group_times, side="left")
        next_times = []
        next_types = []
        next_rul = []
        for idx, current_time in zip(idxs, group_times):
            if idx >= len(machine_failures):
                next_times.append(pd.NaT)
                next_types.append("none")
                next_rul.append(np.nan)
            else:
                next_time = machine_failures[idx]
                next_times.append(pd.Timestamp(next_time))
                next_types.append(machine_types[idx])
                next_rul.append((next_time - current_time) / np.timedelta64(1, "h"))
        next_failure_times.extend(next_times)
        next_failure_types.extend(next_types)
        rul_hours.extend(next_rul)

    df = df.copy()
    df["next_failure_datetime"] = next_failure_times
    df["next_failure_type"] = next_failure_types
    df["rul_hours"] = rul_hours
    df["failure_within_24h"] = ((df["rul_hours"] >= 0) & (df["rul_hours"] <= FAILURE_HORIZON_HOURS)).astype(int)
    return df


def _add_time_since_last_failure(df: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    failures = failures.sort_values(["machineID", "datetime"]).copy()
    last_failure_times = []
    failure_lookup = failures.groupby("machineID")["datetime"].agg(list).to_dict()

    for machine_id, group in df.groupby("machineID", sort=False):
        machine_failures = np.array(failure_lookup.get(machine_id, []), dtype="datetime64[ns]")
        group_times = group["datetime"].to_numpy(dtype="datetime64[ns]")
        idxs = np.searchsorted(machine_failures, group_times, side="right") - 1
        values = []
        for idx, current_time in zip(idxs, group_times):
            if idx < 0:
                values.append(np.nan)
            else:
                values.append((current_time - machine_failures[idx]) / np.timedelta64(1, "h"))
        last_failure_times.extend(values)
    df = df.copy()
    df["hours_since_last_failure"] = last_failure_times
    return df


def _add_time_since_last_maintenance(df: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    maintenance = maintenance.sort_values(["machineID", "datetime"]).copy()
    maintenance_lookup = maintenance.groupby("machineID")["datetime"].agg(list).to_dict()
    values = []

    for machine_id, group in df.groupby("machineID", sort=False):
        machine_maintenance = np.array(maintenance_lookup.get(machine_id, []), dtype="datetime64[ns]")
        group_times = group["datetime"].to_numpy(dtype="datetime64[ns]")
        idxs = np.searchsorted(machine_maintenance, group_times, side="right") - 1
        for idx, current_time in zip(idxs, group_times):
            if idx < 0:
                values.append(np.nan)
            else:
                values.append((current_time - machine_maintenance[idx]) / np.timedelta64(1, "h"))
    df = df.copy()
    df["hours_since_last_maintenance"] = values
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machineID", "datetime"]).copy()
    start_by_machine = df.groupby("machineID")["datetime"].transform("min")
    df["operating_hours"] = (df["datetime"] - start_by_machine).dt.total_seconds() / 3600.0
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machineID", "datetime"]).copy()
    grouped = df.groupby("machineID", sort=False)

    for sensor in SENSOR_COLUMNS:
        for window in ROLLING_WINDOWS:
            df[f"{sensor}_rolling_mean_{window}h"] = grouped[sensor].transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
            )
            df[f"{sensor}_rolling_std_{window}h"] = grouped[sensor].transform(
                lambda s, w=window: s.rolling(window=w, min_periods=2).std()
            ).fillna(0.0)
        for lag in LAG_STEPS:
            df[f"{sensor}_lag_{lag}"] = grouped[sensor].shift(lag)
        df[f"{sensor}_diff_1"] = grouped[sensor].diff().fillna(0.0)
    return df


def _standardize_by_machine(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        mean = df.groupby("machineID")[column].transform("mean")
        std = df.groupby("machineID")[column].transform("std").replace(0, 1.0).fillna(1.0)
        df[f"{column}_zscore"] = (df[column] - mean) / std
    return df


def _add_ai4i_reference_features(df: pd.DataFrame, ai4i: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "ai4i_failure_rate": float(ai4i["Machine failure"].mean()),
        "ai4i_hdf_rate": float(ai4i["HDF"].mean()),
        "ai4i_osf_rate": float(ai4i["OSF"].mean()),
        "ai4i_torque_mean": float(ai4i["Torque [Nm]"].mean()),
        "ai4i_speed_mean": float(ai4i["Rotational speed [rpm]"].mean()),
    }
    enriched = df.copy()
    for key, value in summary.items():
        enriched[key] = value
    return enriched


def build_feature_table(bundle: DatasetBundle, project_root: Path | str) -> pd.DataFrame:
    """Create the master feature table and save it in processed data."""

    telemetry = bundle.telemetry.merge(bundle.machines, on="machineID", how="left")
    telemetry = _add_error_counts(telemetry, bundle.errors)
    telemetry = _add_failure_targets(telemetry, bundle.failures)
    telemetry = _add_time_since_last_failure(telemetry, bundle.failures)
    telemetry = _add_time_since_last_maintenance(telemetry, bundle.maintenance)
    telemetry = _add_calendar_features(telemetry)
    telemetry = _add_temporal_features(telemetry)
    telemetry = _standardize_by_machine(telemetry, SENSOR_COLUMNS)
    telemetry = _add_ai4i_reference_features(telemetry, bundle.ai4i)

    telemetry["model"] = telemetry["model"].astype("category")
    telemetry["has_known_rul"] = telemetry["rul_hours"].notna().astype(int)
    telemetry["rul_hours"] = telemetry["rul_hours"].clip(lower=0)

    processed_dir = ensure_directory(Path(project_root) / "data" / "processed")
    save_dataframe(telemetry, processed_dir / "feature_table.csv")
    return telemetry


def temporal_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a strict per-machine chronological split to avoid leakage."""

    train_parts = []
    test_parts = []
    for _, group in df.sort_values(["machineID", "datetime"]).groupby("machineID", sort=False):
        split_idx = max(1, int(len(group) * train_ratio))
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model-ready feature columns."""

    excluded = {
        "datetime",
        "next_failure_datetime",
        "next_failure_type",
        "failure_within_24h",
        "rul_hours",
        "has_known_rul",
        "predicted_failure_probability",
        "predicted_failure_label",
        "predicted_rul",
    }
    return [column for column in df.columns if column not in excluded]
