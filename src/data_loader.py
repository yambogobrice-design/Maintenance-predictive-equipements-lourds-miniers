"""Load Azure and AI4I predictive maintenance datasets with synthetic fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import RANDOM_STATE, SENSOR_COLUMNS, ensure_directory, save_dataframe


@dataclass
class DatasetBundle:
    """Container for raw source tables and their provenance metadata."""

    telemetry: pd.DataFrame
    failures: pd.DataFrame
    errors: pd.DataFrame
    maintenance: pd.DataFrame
    machines: pd.DataFrame
    ai4i: pd.DataFrame
    metadata: dict[str, str]


AZURE_FILES = {
    "telemetry": "PdM_telemetry.csv",
    "failures": "PdM_failures.csv",
    "errors": "PdM_errors.csv",
    "maintenance": "PdM_maint.csv",
    "machines": "PdM_machines.csv",
}
MAX_MACHINES = 20
MAX_DAYS = 90


def _parse_datetime(df: pd.DataFrame, column: str = "datetime") -> pd.DataFrame:
    if column in df.columns:
        df = df.copy()
        df[column] = pd.to_datetime(df[column])
    return df


def _load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _compact_azure_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Reduce the Azure dataset size for a lightweight academic pipeline."""

    telemetry = _parse_datetime(tables["telemetry"])
    selected_machines = sorted(telemetry["machineID"].unique())[:MAX_MACHINES]
    start_time = telemetry["datetime"].min()
    end_time = start_time + pd.Timedelta(days=MAX_DAYS)
    telemetry = telemetry[
        telemetry["machineID"].isin(selected_machines)
        & (telemetry["datetime"] < end_time)
    ].copy()

    compacted = {"telemetry": telemetry}
    for key in ["failures", "errors", "maintenance"]:
        frame = _parse_datetime(tables[key])
        compacted[key] = frame[
            frame["machineID"].isin(selected_machines)
            & (frame["datetime"] < end_time)
        ].copy()
    compacted["machines"] = tables["machines"][tables["machines"]["machineID"].isin(selected_machines)].copy()
    return compacted


def generate_synthetic_azure_data(
    n_machines: int = 40,
    n_hours: int = 24 * 180,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate Azure-like telemetry, failures, errors, maintenance and machines tables."""

    rng = np.random.default_rng(seed)
    datetimes = pd.date_range("2015-01-01 00:00:00", periods=n_hours, freq="h")
    machine_ids = np.arange(1, n_machines + 1)

    machine_rows = []
    telemetry_rows = []
    failures_rows = []
    error_rows = []
    maint_rows = []
    components = ["comp1", "comp2", "comp3", "comp4"]
    models = ["model1", "model2", "model3", "model4"]

    for machine_id in machine_ids:
        age = int(rng.integers(2, 25))
        model = models[(machine_id - 1) % len(models)]
        machine_rows.append({"machineID": machine_id, "model": model, "age": age})

        health = 1.0
        last_failure_idx = -1
        for idx, dt in enumerate(datetimes):
            degradation = max(idx - last_failure_idx, 1) / 24.0
            health = min(1.8, 0.6 + degradation / 180.0 + age / 60.0)
            volt = 165 + rng.normal(0, 9) + 10 * health
            rotate = 410 - 22 * health + rng.normal(0, 11)
            pressure = 101 + 8 * health + rng.normal(0, 6)
            vibration = 37 + 9 * health + rng.normal(0, 4)
            telemetry_rows.append(
                {
                    "datetime": dt,
                    "machineID": machine_id,
                    "volt": float(volt),
                    "rotate": float(rotate),
                    "pressure": float(pressure),
                    "vibration": float(vibration),
                }
            )

            risk = min(0.08, 0.0008 + 0.006 * max(health - 1.0, 0))
            if rng.random() < risk:
                failure = rng.choice(components, p=[0.28, 0.25, 0.22, 0.25])
                failures_rows.append({"datetime": dt, "machineID": machine_id, "failure": failure})
                maint_rows.append({"datetime": dt, "machineID": machine_id, "comp": failure})
                last_failure_idx = idx
                health = 0.6
            elif rng.random() < min(0.03, 0.004 + 0.015 * max(health - 0.9, 0)):
                error_rows.append(
                    {
                        "datetime": dt,
                        "machineID": machine_id,
                        "errorID": f"error{int(rng.integers(1, 6))}",
                    }
                )
            elif rng.random() < 0.0025:
                maint_rows.append(
                    {
                        "datetime": dt,
                        "machineID": machine_id,
                        "comp": rng.choice(components),
                    }
                )

    return (
        pd.DataFrame(telemetry_rows),
        pd.DataFrame(failures_rows),
        pd.DataFrame(error_rows),
        pd.DataFrame(maint_rows),
        pd.DataFrame(machine_rows),
    )


def generate_synthetic_ai4i_data(n_rows: int = 10000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate an AI4I-like dataset with realistic failure mechanisms."""

    rng = np.random.default_rng(seed)
    udi = np.arange(1, n_rows + 1)
    product_id = [f"M{idx:05d}" for idx in udi]
    machine_type = rng.choice(["L", "M", "H"], size=n_rows, p=[0.5, 0.3, 0.2])
    air_temp = rng.normal(300, 2.0, n_rows)
    process_temp = air_temp + rng.normal(10, 1.2, n_rows)
    speed = np.clip(rng.normal(1500, 220, n_rows), 1100, 2900)
    torque = np.clip(rng.normal(40, 10, n_rows), 5, 80)
    tool_wear = np.clip(rng.normal(120, 70, n_rows), 0, 260)

    hdf = ((torque > 55) & (speed < 1450)).astype(int)
    pwf = ((torque * speed / 1000 > 110) & (torque > 50)).astype(int)
    osf = ((tool_wear > 210) & (speed > 1800)).astype(int)
    twf = ((tool_wear > 180) & (rng.random(n_rows) < 0.12)).astype(int)
    rnF = (rng.random(n_rows) < 0.01).astype(int)

    machine_failure = ((hdf + pwf + osf + twf + rnF) > 0).astype(int)
    df = pd.DataFrame(
        {
            "UDI": udi,
            "Product ID": product_id,
            "Type": machine_type,
            "Air temperature [K]": air_temp.round(3),
            "Process temperature [K]": process_temp.round(3),
            "Rotational speed [rpm]": speed.round(3),
            "Torque [Nm]": torque.round(3),
            "Tool wear [min]": tool_wear.round(3),
            "Machine failure": machine_failure,
            "TWF": twf,
            "HDF": hdf,
            "PWF": pwf,
            "OSF": osf,
            "RNF": rnF,
        }
    )
    return df


def prepare_raw_datasets(project_root: Path | str) -> DatasetBundle:
    """Prepare the raw datasets inside the project directory with fallback generation."""

    project_root = Path(project_root)
    raw_dir = ensure_directory(project_root / "data" / "raw")
    legacy_root = project_root.parent / "data"

    metadata: dict[str, str] = {}
    azure_tables: dict[str, pd.DataFrame] = {}
    for name, file_name in AZURE_FILES.items():
        project_file = raw_dir / file_name
        legacy_file = legacy_root / "azure" / file_name
        source_df = _load_csv_if_exists(project_file)
        if source_df is None:
            source_df = _load_csv_if_exists(legacy_file)
        if source_df is not None:
            azure_tables[name] = source_df
            save_dataframe(source_df, project_file)
            metadata[name] = "real"
        else:
            metadata[name] = "synthetic"

    if set(azure_tables) != set(AZURE_FILES):
        telemetry, failures, errors, maintenance, machines = generate_synthetic_azure_data()
        generated = {
            "telemetry": telemetry,
            "failures": failures,
            "errors": errors,
            "maintenance": maintenance,
            "machines": machines,
        }
        for name, df in generated.items():
            if name not in azure_tables:
                azure_tables[name] = df
                save_dataframe(df, raw_dir / AZURE_FILES[name])
                metadata[name] = "synthetic"
    else:
        azure_tables = _compact_azure_tables(azure_tables)
        for name, df in azure_tables.items():
            save_dataframe(df, raw_dir / AZURE_FILES[name])
            metadata[name] = f"{metadata[name]}_compacted"

    ai4i_path = raw_dir / "ai4i2020.csv"
    legacy_ai4i = legacy_root / "ai4i" / "ai4i2020.csv"
    ai4i_df = _load_csv_if_exists(ai4i_path)
    if ai4i_df is None:
        ai4i_df = _load_csv_if_exists(legacy_ai4i)
    if ai4i_df is None:
        ai4i_df = generate_synthetic_ai4i_data()
        save_dataframe(ai4i_df, ai4i_path)
        metadata["ai4i"] = "synthetic"
    else:
        save_dataframe(ai4i_df, ai4i_path)
        metadata["ai4i"] = "real"

    return DatasetBundle(
        telemetry=_parse_datetime(azure_tables["telemetry"]),
        failures=_parse_datetime(azure_tables["failures"]),
        errors=_parse_datetime(azure_tables["errors"]),
        maintenance=_parse_datetime(azure_tables["maintenance"]),
        machines=azure_tables["machines"].copy(),
        ai4i=ai4i_df.copy(),
        metadata=metadata,
    )


def build_demo_machine_snapshot(features_df: pd.DataFrame) -> pd.DataFrame:
    """Build the latest machine-level status table for the dashboard."""

    latest = (
        features_df.sort_values(["machineID", "datetime"])
        .groupby("machineID")
        .tail(1)
        .copy()
    )
    latest["status"] = np.select(
        [
            latest["predicted_failure_probability"] >= 0.65,
            latest["predicted_failure_probability"] >= 0.4,
        ],
        ["Critique", "Attention"],
        default="Normal",
    )
    latest["last_anomaly"] = np.where(latest["error_count_cumulative"] > 0, "Erreur capteur", "RAS")
    cols = [
        "machineID",
        "status",
        "predicted_failure_probability",
        "predicted_rul",
        "last_anomaly",
        "failure_within_24h",
        *SENSOR_COLUMNS,
    ]
    return latest[cols].reset_index(drop=True)
