"""Generate new report figures in figure1 folder."""

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
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
    roc_curve,
    auc,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
FIGURE1 = ROOT / "rapport" / "figure1"
MODELS = ROOT / "models"

def savefig(name: str) -> None:
    FIGURE1.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE1 / name, dpi=200, bbox_inches="tight")
    plt.close()

def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, linestyle="--")
    ax.set_facecolor("#ffffff")

def load_data():
    features = pd.read_csv(PROCESSED / "feature_table.csv", parse_dates=["datetime"])
    predictions = pd.read_csv(PROCESSED / "fleet_predictions.csv", parse_dates=["datetime"])
    all_preds = pd.read_csv(PROCESSED / "all_predictions.csv", parse_dates=["datetime"])
    metrics = json.loads((PROCESSED / "metrics_summary.json").read_text(encoding="utf-8"))
    return features, predictions, all_preds, metrics

def plot_confusion_matrix(all_preds: pd.DataFrame):
    # Get the best model predictions (Random Forest is usually the best here)
    rf_preds = all_preds[all_preds["model_name"] == "rf"].copy()
    if rf_preds.empty:
        rf_preds = all_preds[all_preds["model_name"] == "classifier_rf"].copy()
    
    if rf_preds.empty:
        # Fallback to the first available classifier model
        classifier_models = [m for m in all_preds["model_name"].unique() if "rul" not in m and "dummy" not in m]
        if classifier_models:
            rf_preds = all_preds[all_preds["model_name"] == classifier_models[0]].copy()
        else:
            return

    y_true = rf_preds["failure_within_24h"]
    y_pred = rf_preds["predicted_failure_label"]
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pas de panne", "Panne"])
    disp.plot(ax=ax, cmap="Greys", colorbar=False)
    ax.set_title("Matrice de Confusion (Random Forest)", fontweight="bold", pad=20)
    savefig("confusion_matrix.png")

def plot_pr_curves(all_preds: pd.DataFrame):
    classifier_models = [m for m in all_preds["model_name"].unique() if "rul" not in m]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name in classifier_models:
        df = all_preds[all_preds["model_name"] == model_name]
        precision, recall, _ = precision_recall_curve(df["failure_within_24h"], df["predicted_failure_probability"])
        ax.plot(recall, precision, label=f"{model_name}", linewidth=2)
    
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Courbes Précision-Rappel", fontweight="bold")
    ax.legend(frameon=False)
    style_axes(ax)
    ax.grid(True, linestyle="--", alpha=0.7)
    savefig("precision_recall_curves.png")

def plot_roc_curves(all_preds: pd.DataFrame):
    classifier_models = [m for m in all_preds["model_name"].unique() if "rul" not in m]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name in classifier_models:
        df = all_preds[all_preds["model_name"] == model_name]
        fpr, tpr, _ = roc_curve(df["failure_within_24h"], df["predicted_failure_probability"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})", linewidth=2)
    
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_xlabel("Taux de Faux Positifs")
    ax.set_ylabel("Taux de Vrais Positifs")
    ax.set_title("Courbes ROC", fontweight="bold")
    ax.legend(frameon=False)
    style_axes(ax)
    ax.grid(True, linestyle="--", alpha=0.7)
    savefig("roc_curves.png")

def plot_rul_predictions(all_preds: pd.DataFrame):
    rul_models = [m for m in all_preds["model_name"].unique() if "rul" in m and "dummy" not in m]
    if not rul_models:
        return
    
    model_name = rul_models[0]
    df = all_preds[all_preds["model_name"] == model_name].dropna(subset=["rul_hours", "predicted_rul"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["rul_hours"], df["predicted_rul"], alpha=0.3, color="#111111", s=10)
    ax.plot([0, df["rul_hours"].max()], [0, df["rul_hours"].max()], "r--", linewidth=2)
    ax.set_xlabel("RUL Réelle (h)")
    ax.set_ylabel("RUL Prédite (h)")
    ax.set_title(f"Prédictions RUL vs Réalité ({model_name})", fontweight="bold")
    style_axes(ax)
    savefig("rul_scatter.png")

def main():
    features, predictions, all_preds, metrics = load_data()
    plot_confusion_matrix(all_preds)
    plot_pr_curves(all_preds)
    plot_roc_curves(all_preds)
    plot_rul_predictions(all_preds)
    
    # Also copy/regenerate some of the important ones from the original script but in figure1
    import scripts.generate_report_figures as old_gen
    old_gen.FIGURES = FIGURE1
    old_gen.plot_failure_distribution(features)
    old_gen.plot_sensor_trends(features)
    old_gen.plot_model_comparison(metrics)
    old_gen.plot_rul_errors(metrics)
    old_gen.plot_dashboard_fleet(predictions)
    old_gen.plot_dashboard_machine(predictions)
    old_gen.plot_interpretability(features)
    print("Figures generated successfully in rapport/figure1")

if __name__ == "__main__":
    main()
