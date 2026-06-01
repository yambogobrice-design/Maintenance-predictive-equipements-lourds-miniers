"""Generate report figures from processed M3 project artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_engineering import get_feature_columns, temporal_train_test_split

PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "rapport" / "figures"


def savefig(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_facecolor("#ffffff")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    features = pd.read_csv(PROCESSED / "feature_table.csv", parse_dates=["datetime"])
    predictions = pd.read_csv(PROCESSED / "fleet_predictions.csv", parse_dates=["datetime"])
    metrics = json.loads((PROCESSED / "metrics_summary.json").read_text(encoding="utf-8"))
    return features, predictions, metrics


def plot_failure_distribution(features: pd.DataFrame) -> None:
    counts = features["failure_within_24h"].value_counts().sort_index()
    labels = ["Pas de panne 24h", "Panne <= 24h"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, counts.values, color=["#111111", "#9a9a9a"], width=0.55)
    ax.set_title("Desequilibre de la cible panne a 24 heures", fontweight="bold")
    ax.set_ylabel("Nombre d'observations")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height()):,}".replace(",", " "), ha="center", va="bottom")
    style_axes(ax)
    savefig("eda_distribution_pannes.png")


def plot_sensor_trends(features: pd.DataFrame) -> None:
    machine_id = int(features.groupby("machineID")["failure_within_24h"].sum().sort_values(ascending=False).index[0])
    machine = features[features["machineID"] == machine_id].sort_values("datetime").head(24 * 14)
    sensors = ["volt", "rotate", "pressure", "vibration"]
    colors = ["#111111", "#555555", "#8a8a8a", "#c2c2c2"]
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 6.2), sharex=True)
    for ax, sensor, color in zip(axes, sensors, colors):
        ax.plot(machine["datetime"], machine[sensor], color=color, linewidth=1.2)
        ax.set_ylabel(sensor)
        style_axes(ax)
    axes[0].set_title(f"Evolution des capteurs - machine {machine_id}", fontweight="bold")
    savefig("eda_evolution_capteurs.png")


def plot_model_comparison(metrics: dict) -> None:
    class_df = pd.DataFrame(metrics["classification"]).T.reset_index(names="model")
    class_df = class_df[["model", "precision_failure", "recall_failure", "f1_failure"]]
    x = np.arange(len(class_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(x - width, class_df["precision_failure"], width, label="Precision", color="#111111")
    ax.bar(x, class_df["recall_failure"], width, label="Rappel", color="#777777")
    ax.bar(x + width, class_df["f1_failure"], width, label="F1", color="#c5c5c5")
    ax.set_xticks(x)
    ax.set_xticklabels(class_df["model"])
    ax.set_ylim(0, 1.08)
    ax.set_title("Comparaison des modeles de classification", fontweight="bold")
    ax.legend(frameon=False, ncol=3)
    style_axes(ax)
    savefig("resultats_classification.png")


def plot_rul_errors(metrics: dict) -> None:
    reg_df = pd.DataFrame(metrics["regression"]).T.reset_index(names="model")
    x = np.arange(len(reg_df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(x - width / 2, reg_df["mae"], width, label="MAE", color="#111111")
    ax.bar(x + width / 2, reg_df["rmse"], width, label="RMSE", color="#9a9a9a")
    ax.set_xticks(x)
    ax.set_xticklabels(reg_df["model"])
    ax.set_ylabel("Erreur RUL (heures)")
    ax.set_title("Erreur de regression RUL", fontweight="bold")
    ax.legend(frameon=False)
    style_axes(ax)
    savefig("resultats_rul.png")


def plot_dashboard_fleet(predictions: pd.DataFrame) -> None:
    snapshot = (
        predictions.sort_values(["machineID", "datetime"])
        .groupby("machineID", as_index=False)
        .tail(1)
        .copy()
    )
    snapshot["priority"] = 100 * snapshot["predicted_failure_probability"] + np.maximum(0, 72 - snapshot["predicted_rul"])
    snapshot = snapshot.sort_values("priority", ascending=False).head(8)
    critical = int(((snapshot["predicted_failure_probability"] >= 0.65) | (snapshot["predicted_rul"] <= 20)).sum())
    avg_rul = predictions.groupby("machineID").tail(1)["predicted_rul"].mean()
    avg_risk = 100 * predictions.groupby("machineID").tail(1)["predicted_failure_probability"].mean()

    fig = plt.figure(figsize=(10.4, 6.0), facecolor="#ffffff")
    gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 1.2, 2.2])
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0, 0.70, "Fleet Risk Control", fontsize=24, fontweight="bold", color="#090909")
    ax_title.text(0, 0.18, "Vue decision flotte - style dashboard blanc/noir", fontsize=11, color="#666666")
    for i, (label, value) in enumerate(
        [
            ("Machines critiques", f"{critical}"),
            ("Risque moyen", f"{avg_risk:.1f}%"),
            ("RUL moyen", f"{avg_rul:.0f} h"),
            ("Machines suivies", f"{predictions['machineID'].nunique()}"),
        ]
    ):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor("#f6f6f4")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#dedede")
        ax.text(0.08, 0.62, value, fontsize=22, fontweight="bold", color="#090909", transform=ax.transAxes)
        ax.text(0.08, 0.25, label, fontsize=9, color="#666666", transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, :])
    y = np.arange(len(snapshot))
    ax.barh(y, 100 * snapshot["predicted_failure_probability"], color="#111111", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels([f"Machine {int(mid)}" for mid in snapshot["machineID"]])
    ax.invert_yaxis()
    ax.set_xlabel("Probabilite de panne (%)")
    ax.set_title("Top machines par priorite", fontweight="bold")
    style_axes(ax)
    savefig("dashboard_capture_flotte.png")


def plot_dashboard_machine(predictions: pd.DataFrame) -> None:
    latest = predictions.sort_values("datetime").groupby("machineID").tail(1)
    machine_id = int(latest.sort_values("predicted_failure_probability", ascending=False).iloc[0]["machineID"])
    machine = predictions[predictions["machineID"] == machine_id].sort_values("datetime").tail(24 * 14)
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.2), sharex=True)
    axes[0].plot(machine["datetime"], 100 * machine["predicted_failure_probability"], color="#111111", linewidth=2)
    axes[0].axhline(65, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Panne (%)")
    axes[0].set_title(f"Diagnostic machine {machine_id} - risque et RUL", fontweight="bold")
    style_axes(axes[0])
    axes[1].plot(machine["datetime"], machine["predicted_rul"], color="#555555", linewidth=2)
    axes[1].axhline(72, color="#999999", linestyle="--", linewidth=1)
    axes[1].set_ylabel("RUL (h)")
    style_axes(axes[1])
    savefig("dashboard_capture_machine.png")


def plot_interpretability(features: pd.DataFrame) -> None:
    artifact = joblib.load(ROOT / "models" / "classifier_rf.joblib")
    model = artifact.pipeline.named_steps["model"]
    preprocessor = artifact.pipeline.named_steps["preprocessor"]
    if hasattr(model, "feature_importances_"):
        names = preprocessor.get_feature_names_out()
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        names = preprocessor.get_feature_names_out()
        values = np.abs(model.coef_[0])
    else:
        train_df, _ = temporal_train_test_split(features)
        numeric = train_df[get_feature_columns(train_df)].select_dtypes(include=[np.number])
        values = numeric.corrwith(train_df["failure_within_24h"]).abs().fillna(0).to_numpy()
        names = numeric.columns.to_numpy()

    names = np.array([name.replace("num__", "").replace("cat__", "") for name in names])
    top_idx = np.argsort(values)[-12:]
    top_names = names[top_idx]
    top_values = values[top_idx]
    order = np.argsort(top_values)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.barh(top_names[order], top_values[order], color="#111111")
    ax.set_title("Interpretabilite - variables les plus influentes", fontweight="bold")
    ax.set_xlabel("Importance relative")
    style_axes(ax)
    savefig("interpretabilite_importances.png")


def main() -> None:
    features, predictions, metrics = load_inputs()
    plot_failure_distribution(features)
    plot_sensor_trends(features)
    plot_model_comparison(metrics)
    plot_rul_errors(metrics)
    plot_dashboard_fleet(predictions)
    plot_dashboard_machine(predictions)
    plot_interpretability(features)


if __name__ == "__main__":
    main()
