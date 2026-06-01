"""Decision-oriented Streamlit dashboard for the M3 predictive maintenance project."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CRITICAL_PROBA = 0.65
WATCH_PROBA = 0.40
CRITICAL_RUL = 20.0
WATCH_RUL = 72.0

CSS = """
<style>
    :root {
        --ink: #090909;
        --muted: #5f6368;
        --line: #dedede;
        --soft: #f6f6f4;
        --panel: #ffffff;
    }
    .stApp {
        background: #ffffff;
        color: var(--ink);
    }
    .block-container {
        max-width: 1480px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }
    section[data-testid="stSidebar"] {
        background: #0b0b0b;
        border-right: 1px solid #1f1f1f;
    }
    section[data-testid="stSidebar"] * {
        color: #f8f8f8;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.045);
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: 0.82rem;
    }
    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 760;
    }
    .hero {
        border-bottom: 1px solid var(--line);
        padding: 0.4rem 0 1rem 0;
        margin-bottom: 1rem;
    }
    .eyebrow {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .hero-title {
        font-size: clamp(2rem, 4vw, 4.3rem);
        line-height: 0.92;
        font-weight: 820;
        margin: 0.25rem 0 0.55rem 0;
    }
    .hero-copy {
        color: var(--muted);
        max-width: 860px;
        font-size: 1.02rem;
    }
    .section-title {
        color: var(--ink);
        font-size: 1.1rem;
        font-weight: 780;
        margin: 1.1rem 0 0.45rem 0;
    }
    .decision-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 0.8rem 0 1.1rem 0;
    }
    .decision-item {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem;
        background: var(--soft);
    }
    .decision-item strong {
        display: block;
        font-size: 1.35rem;
        color: var(--ink);
    }
    .decision-item span {
        color: var(--muted);
        font-size: 0.82rem;
    }
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.16rem 0.52rem;
        border: 1px solid #111111;
        background: #111111;
        color: #ffffff;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-light {
        background: #ffffff;
        color: #111111;
    }
    @media (max-width: 900px) {
        .decision-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 560px) {
        .decision-strip { grid-template-columns: 1fr; }
    }
</style>
"""


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load persisted pipeline outputs."""

    fleet_predictions = pd.read_csv(PROCESSED_DIR / "fleet_predictions.csv", parse_dates=["datetime"])
    fleet_snapshot = pd.read_csv(PROCESSED_DIR / "fleet_snapshot.csv")
    metrics = json.loads((PROCESSED_DIR / "metrics_summary.json").read_text(encoding="utf-8"))
    return fleet_predictions, fleet_snapshot, metrics


def status_from_row(row: pd.Series) -> str:
    if row["predicted_failure_probability"] >= CRITICAL_PROBA or row["predicted_rul"] <= CRITICAL_RUL:
        return "Critique"
    if row["predicted_failure_probability"] >= WATCH_PROBA or row["predicted_rul"] <= WATCH_RUL:
        return "A surveiller"
    return "Normal"


def action_from_row(row: pd.Series) -> str:
    if row["predicted_rul"] <= CRITICAL_RUL:
        return "Arret controle et inspection immediate"
    if row["predicted_failure_probability"] >= CRITICAL_PROBA:
        return "Intervention preventive sous 24 h"
    if row["predicted_rul"] <= WATCH_RUL:
        return "Planifier maintenance sous 72 h"
    if row["predicted_failure_probability"] >= WATCH_PROBA:
        return "Surveillance renforcee"
    return "Operation normale"


def prepare_snapshot(fleet_snapshot: pd.DataFrame) -> pd.DataFrame:
    df = fleet_snapshot.copy()
    df["status"] = df.apply(status_from_row, axis=1)
    df["probabilite_panne_%"] = (100 * df["predicted_failure_probability"]).round(1)
    df["rul_h"] = df["predicted_rul"].round(1)
    df["priorite"] = (
        100 * df["predicted_failure_probability"]
        + np.maximum(0, WATCH_RUL - df["predicted_rul"]) * 1.2
        + np.where(df["status"] == "Critique", 60, 0)
    ).round(1)
    df["action_recommandee"] = df.apply(action_from_row, axis=1)
    return df.sort_values(["priorite", "probabilite_panne_%"], ascending=False)


def plot_layout(title: str, height: int = 390) -> dict:
    return {
        "template": "plotly_white",
        "height": height,
        "title": {"text": title, "font": {"color": "#090909", "size": 18}},
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"color": "#090909"},
        "margin": {"l": 35, "r": 20, "t": 58, "b": 35},
    }


def render_header(fleet_predictions: pd.DataFrame) -> None:
    selected_classifier = fleet_predictions.get("selected_classifier", pd.Series(["modele"])).dropna().tail(1).iloc[0]
    selected_regressor = fleet_predictions.get("selected_regressor", pd.Series(["modele"])).dropna().tail(1).iloc[0]
    last_update = fleet_predictions["datetime"].max().strftime("%d/%m/%Y %H:%M")
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Projet M3 - Maintenance predictive miniere</div>
            <div class="hero-title">Fleet Risk Control</div>
            <div class="hero-copy">
                Supervision des equipements lourds avec probabilite de panne, estimation RUL et recommandations
                de maintenance. Derniere donnee: {last_update}. Modeles actifs:
                <span class="badge badge-light">{selected_classifier}</span>
                <span class="badge badge-light">{selected_regressor}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fleet_page(fleet_predictions: pd.DataFrame, fleet_snapshot: pd.DataFrame) -> None:
    snapshot = prepare_snapshot(fleet_snapshot)
    critical = int((snapshot["status"] == "Critique").sum())
    watch = int((snapshot["status"] == "A surveiller").sum())
    avg_rul = snapshot["predicted_rul"].mean()
    max_risk = 100 * snapshot["predicted_failure_probability"].max()

    cols = st.columns(4)
    cols[0].metric("Machines critiques", critical)
    cols[1].metric("A surveiller", watch)
    cols[2].metric("RUL moyen flotte", f"{avg_rul:.1f} h")
    cols[3].metric("Risque maximum", f"{max_risk:.1f}%")

    top = snapshot.head(1).iloc[0]
    st.markdown(
        f"""
        <div class="decision-strip">
            <div class="decision-item"><strong>#{int(top['machineID'])}</strong><span>Machine prioritaire</span></div>
            <div class="decision-item"><strong>{top['probabilite_panne_%']:.1f}%</strong><span>Probabilite panne</span></div>
            <div class="decision-item"><strong>{top['rul_h']:.1f} h</strong><span>RUL estime</span></div>
            <div class="decision-item"><strong>{top['status']}</strong><span>{top['action_recommandee']}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_filter = st.segmented_control(
        "Filtrer par statut",
        ["Tous", "Critique", "A surveiller", "Normal"],
        default="Tous",
    )
    filtered = snapshot if status_filter == "Tous" else snapshot[snapshot["status"] == status_filter]

    st.markdown("<div class='section-title'>File de decision maintenance</div>", unsafe_allow_html=True)
    st.dataframe(
        filtered[
            [
                "machineID",
                "status",
                "probabilite_panne_%",
                "rul_h",
                "priorite",
                "last_anomaly",
                "action_recommandee",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "machineID": st.column_config.NumberColumn("Machine", format="%d"),
            "status": "Statut",
            "probabilite_panne_%": st.column_config.ProgressColumn(
                "Panne",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "rul_h": st.column_config.NumberColumn("RUL h", format="%.1f"),
            "priorite": st.column_config.NumberColumn("Priorite", format="%.1f"),
            "last_anomaly": "Signal",
            "action_recommandee": "Action recommandee",
        },
    )

    fleet_time = (
        fleet_predictions.groupby("datetime", as_index=False)
        .agg(
            avg_failure_probability=("predicted_failure_probability", "mean"),
            max_failure_probability=("predicted_failure_probability", "max"),
            avg_predicted_rul=("predicted_rul", "mean"),
        )
        .tail(24 * 14)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fleet_time["datetime"],
            y=100 * fleet_time["avg_failure_probability"],
            name="Risque moyen",
            line=dict(color="#111111", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fleet_time["datetime"],
            y=100 * fleet_time["max_failure_probability"],
            name="Risque max",
            line=dict(color="#7b7b7b", width=1.8, dash="dot"),
        )
    )
    fig.add_hline(y=100 * WATCH_PROBA, line_dash="dash", line_color="#9a9a9a", annotation_text="Surveillance")
    fig.add_hline(y=100 * CRITICAL_PROBA, line_dash="solid", line_color="#111111", annotation_text="Critique")
    fig.update_layout(**plot_layout("Evolution du risque flotte - 14 derniers jours"))
    fig.update_yaxes(title="Probabilite de panne (%)", rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)


def render_machine_page(fleet_predictions: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Diagnostic par equipement</div>", unsafe_allow_html=True)
    machine_ids = sorted(fleet_predictions["machineID"].unique().tolist())
    machine_id = st.selectbox("Machine", machine_ids)
    horizon = st.slider("Historique affiche (jours)", 1, 30, 14)
    machine_df = fleet_predictions[fleet_predictions["machineID"] == machine_id].sort_values("datetime").tail(24 * horizon)
    latest = machine_df.tail(1).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Statut", status_from_row(latest))
    c2.metric("Probabilite panne", f"{100 * latest['predicted_failure_probability']:.1f}%")
    c3.metric("RUL estime", f"{latest['predicted_rul']:.1f} h")
    c4.metric("Erreurs cumulees", f"{latest['error_count_cumulative']:.0f}")
    st.info(action_from_row(latest), icon=":material/build:")

    risk_fig = go.Figure()
    risk_fig.add_trace(
        go.Scatter(
            x=machine_df["datetime"],
            y=100 * machine_df["predicted_failure_probability"],
            name="Probabilite panne",
            line=dict(color="#090909", width=2.4),
        )
    )
    risk_fig.add_trace(
        go.Scatter(
            x=machine_df["datetime"],
            y=machine_df["predicted_rul"],
            name="RUL",
            yaxis="y2",
            line=dict(color="#6c6c6c", width=2),
        )
    )
    risk_fig.add_hline(y=100 * CRITICAL_PROBA, line_dash="solid", line_color="#111111")
    risk_fig.update_layout(
        **plot_layout(f"Risque et RUL - Machine {machine_id}", height=420),
        yaxis=dict(title="Probabilite panne (%)"),
        yaxis2=dict(title="RUL h", overlaying="y", side="right"),
    )
    st.plotly_chart(risk_fig, use_container_width=True)

    sensors = ["volt", "rotate", "pressure", "vibration"]
    sensor_fig = px.line(
        machine_df,
        x="datetime",
        y=sensors,
        title=f"Capteurs principaux - Machine {machine_id}",
        color_discrete_sequence=["#090909", "#4b4b4b", "#8b8b8b", "#c0c0c0"],
    )
    sensor_fig.update_layout(**plot_layout(f"Capteurs principaux - Machine {machine_id}", height=390))
    st.plotly_chart(sensor_fig, use_container_width=True)

    factors = pd.Series(
        {
            "Vibration": float(abs(latest.get("vibration_zscore", 0))),
            "Pression": float(abs(latest.get("pressure_zscore", 0))),
            "Tension": float(abs(latest.get("volt_zscore", 0))),
            "Rotation": float(abs(latest.get("rotate_zscore", 0))),
            "Erreurs": float(latest.get("error_count_cumulative", 0)) / max(len(machine_df), 1),
        }
    ).sort_values(ascending=True)
    factor_fig = px.bar(
        x=factors.values,
        y=factors.index,
        orientation="h",
        title="Facteurs de vigilance normalises",
        color_discrete_sequence=["#111111"],
    )
    factor_fig.update_layout(**plot_layout("Facteurs de vigilance normalises", height=320))
    factor_fig.update_xaxes(title="Intensite relative")
    factor_fig.update_yaxes(title="")
    st.plotly_chart(factor_fig, use_container_width=True)


def render_metrics_page(metrics: dict, fleet_predictions: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Qualite modele et seuils de decision</div>", unsafe_allow_html=True)
    class_df = pd.DataFrame(metrics["classification"]).T.reset_index(names="model")
    reg_df = pd.DataFrame(metrics["regression"]).T.reset_index(names="model")

    c1, c2 = st.columns(2)
    best_class = class_df[class_df["model"] != "dummy"].sort_values("f1_failure", ascending=False).head(1)
    best_reg = reg_df[reg_df["model"] != "dummy_rul"].sort_values("mae", ascending=True).head(1)
    if not best_class.empty:
        row = best_class.iloc[0]
        c1.metric("Meilleur classifieur", row["model"], f"F1 panne {row['f1_failure']:.3f}")
    if not best_reg.empty:
        row = best_reg.iloc[0]
        c2.metric("Meilleur RUL", row["model"], f"MAE {row['mae']:.1f} h")

    class_fig = px.bar(
        class_df,
        x="model",
        y=["f1_failure", "precision_failure", "recall_failure"],
        barmode="group",
        title="Classification panne: precision, rappel, F1",
        color_discrete_sequence=["#090909", "#777777", "#c7c7c7"],
    )
    class_fig.update_layout(**plot_layout("Classification panne: precision, rappel, F1"))
    st.plotly_chart(class_fig, use_container_width=True)
    st.dataframe(class_df.round(4), use_container_width=True, hide_index=True)

    reg_fig = px.bar(
        reg_df,
        x="model",
        y=["mae", "rmse", "critical_zone_mae"],
        barmode="group",
        title="Regression RUL: erreur globale et zone critique",
        color_discrete_sequence=["#090909", "#777777", "#c7c7c7"],
    )
    reg_fig.update_layout(**plot_layout("Regression RUL: erreur globale et zone critique"))
    st.plotly_chart(reg_fig, use_container_width=True)
    st.dataframe(reg_df.round(4), use_container_width=True, hide_index=True)

    residuals = fleet_predictions["predicted_rul"] - fleet_predictions["rul_hours"].fillna(fleet_predictions["predicted_rul"])
    residual_fig = px.histogram(
        residuals,
        nbins=45,
        title="Distribution des erreurs RUL operationnelles",
        color_discrete_sequence=["#111111"],
    )
    residual_fig.update_layout(**plot_layout("Distribution des erreurs RUL operationnelles"))
    residual_fig.update_xaxes(title="Erreur RUL predite - reelle (h)")
    st.plotly_chart(residual_fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="M3 Fleet Risk Control", page_icon="M3", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    fleet_predictions, fleet_snapshot, metrics = load_data()
    render_header(fleet_predictions)

    page = st.sidebar.radio(
        "Navigation",
        ["Decision flotte", "Diagnostic machine", "Performance modeles"],
    )
    st.sidebar.divider()
    st.sidebar.caption("Seuils operationnels")
    st.sidebar.write(f"Critique: panne >= {100 * CRITICAL_PROBA:.0f}% ou RUL <= {CRITICAL_RUL:.0f} h")
    st.sidebar.write(f"Surveillance: panne >= {100 * WATCH_PROBA:.0f}% ou RUL <= {WATCH_RUL:.0f} h")

    if page == "Decision flotte":
        render_fleet_page(fleet_predictions, fleet_snapshot)
    elif page == "Diagnostic machine":
        render_machine_page(fleet_predictions)
    else:
        render_metrics_page(metrics, fleet_predictions)


if __name__ == "__main__":
    main()
