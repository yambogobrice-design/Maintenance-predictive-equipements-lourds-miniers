"""Generate the five project notebooks with reusable content."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

import textwrap


PROJECT_IMPORTS = """from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import prepare_raw_datasets
from src.feature_engineering import build_feature_table, temporal_train_test_split, get_feature_columns
from src.project_pipeline import run_full_training
from src.models import train_classifiers, train_rul_models
from src.evaluation import evaluate_classifier, evaluate_regressor
from src.utils import COLOR_PALETTE, RANDOM_STATE, SENSOR_COLUMNS, CRITICAL_RUL_THRESHOLD

plt.style.use("seaborn-v0_8")
sns.set_palette([COLOR_PALETTE["accent"], COLOR_PALETTE["success"], COLOR_PALETTE["info"], COLOR_PALETTE["danger"]])
pd.set_option("display.max_columns", 120)
"""


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code_cell(code: str):
    return nbf.v4.new_code_cell(textwrap.dedent(code).strip() + "\n")


def build_eda_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        markdown_cell(
            """
            # Notebook 01 - EDA
            ## Projet M3 - Maintenance predictive d'equipements miniers

            Ce notebook realise une analyse exploratoire complete sur les donnees Azure Predictive Maintenance, avec integration du jeu AI4I 2020 comme reference complementaire. L'objectif est de caracteriser les capteurs, le desequilibre de classes, les pannes et les tendances temporelles avant toute modelisation.
            """
        ),
        markdown_cell(
            """
            ## 1. Constantes et chargement

            Toutes les constantes sont explicites pour garantir la reproductibilite. Aucun `random split` n'est utilise dans ce projet afin d'eviter toute fuite temporelle.
            """
        ),
        code_cell(PROJECT_IMPORTS),
        code_cell(
            """
            # Constantes de visualisation et de controle
            TOP_MACHINES = 5
            SAMPLE_MACHINES = [1, 2, 3, 4, 5]

            bundle = prepare_raw_datasets(PROJECT_ROOT)
            features = build_feature_table(bundle, PROJECT_ROOT)
            train_df, test_df = temporal_train_test_split(features)

            # ... Nettoyage : prints supprimés ...
            features.head()
            """
        ),
        markdown_cell("## 2. Statistiques descriptives\n\nLes statistiques suivantes couvrent les variables capteurs principales et mettent en evidence les echelles de variation par machine."),
        code_cell(
            """
            descriptive_stats = features[SENSOR_COLUMNS + ["age", "error_count", "error_count_cumulative", "rul_hours"]].describe().T
            descriptive_stats
            """
        ),
        markdown_cell("## 3. Distribution des capteurs\n\nOn verifie les distributions de `volt`, `rotate`, `pressure` et `vibration` pour identifier les asymetries, dispersions et eventuelles valeurs aberrantes."),
        code_cell(
            """
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for ax, sensor in zip(axes.ravel(), SENSOR_COLUMNS):
                sns.histplot(features[sensor], kde=True, ax=ax, color=COLOR_PALETTE["accent"])
                ax.set_title(f"Distribution de {sensor}")
                ax.set_xlabel(sensor)
                ax.set_ylabel("Frequence")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## 4. Frequence des pannes\n\nCette vue compare la frequence des pannes par composant et par machine, ce qui aide a justifier la difficulte du probleme de classification."),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            bundle.failures["failure"].value_counts().plot(kind="bar", ax=axes[0], color=COLOR_PALETTE["danger"])
            axes[0].set_title("Frequence des pannes par type")
            axes[0].set_xlabel("Type de panne")
            axes[0].set_ylabel("Nombre d'evenements")

            bundle.failures["machineID"].value_counts().head(15).sort_values().plot(kind="barh", ax=axes[1], color=COLOR_PALETTE["info"])
            axes[1].set_title("Top 15 machines les plus en panne")
            axes[1].set_xlabel("Nombre de pannes")
            axes[1].set_ylabel("Machine")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## 5. Correlations\n\nLa heatmap annotee permet d'identifier les associations lineaires utiles pour le feature engineering et d'anticiper les colinearites."),
        code_cell(
            """
            corr_cols = SENSOR_COLUMNS + ["age", "error_count_cumulative", "hours_since_last_failure", "rul_hours", "failure_within_24h"]
            corr = features[corr_cols].corr(numeric_only=True)
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
            plt.title("Heatmap de correlations")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## 6. Degradation temporelle\n\nNous visualisons 5 machines exemple sur la serie temporelle pour observer les signaux de degradation avant panne."),
        code_cell(
            """
            subset = features[features["machineID"].isin(SAMPLE_MACHINES)].copy()
            fig, axes = plt.subplots(len(SAMPLE_MACHINES), 1, figsize=(16, 14), sharex=True)
            for ax, machine_id in zip(axes, SAMPLE_MACHINES):
                machine_slice = subset[subset["machineID"] == machine_id].tail(24 * 14)
                ax.plot(machine_slice["datetime"], machine_slice["vibration"], label="Vibration", color=COLOR_PALETTE["danger"])
                ax.plot(machine_slice["datetime"], machine_slice["pressure"], label="Pressure", color=COLOR_PALETTE["info"], alpha=0.8)
                ax.set_title(f"Machine {machine_id} - degradation sur 14 jours")
                ax.set_ylabel("Valeur capteur")
                ax.legend()
            axes[-1].set_xlabel("Temps")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## 7. Desequilibre de classes\n\nLa variable cible est definie comme une panne dans les 24h. Le ratio ci-dessous justifie l'usage de `class_weight` et le test de `SMOTE`."),
        code_cell(
            """
            class_ratio = features["failure_within_24h"].value_counts(normalize=True).rename({0: "Normal", 1: "Panne <24h"})
            # ... Nettoyage : prints supprimés ...
            plt.figure(figsize=(6, 4))
            class_ratio.plot(kind="bar", color=[COLOR_PALETTE["success"], COLOR_PALETTE["danger"]])
            plt.title("Distribution de la cible")
            plt.ylabel("Proportion")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## 8. Valeurs manquantes et aberrantes\n\nCette section verifie la qualite des donnees et permet d'expliquer les decisions de pretraitement."),
        code_cell(
            """
            missing = features.isna().mean().sort_values(ascending=False).head(15)
            missing
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for ax, sensor in zip(axes.ravel(), SENSOR_COLUMNS):
                sns.boxplot(data=features, y=sensor, ax=ax, color=COLOR_PALETTE["accent"])
                ax.set_title(f"Boxplot - {sensor}")
                ax.set_ylabel(sensor)
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## 9. Synthese EDA\n\nLes capteurs montrent des regimes de fonctionnement distincts selon les machines, les pannes restent rares mais observables, et les variables temporelles seront essentielles pour capturer la degradation. Le split chronologique est deja prepare pour garantir une evaluation honnete."),
    ]
    return nb


def build_feature_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        markdown_cell("# Notebook 02 - Feature Engineering\n\nCe notebook construit les variables temporelles necessaires a la classification de panne et a l'estimation de la RUL."),
        code_cell(PROJECT_IMPORTS),
        markdown_cell("## 1. Construction de la table de features\n\nLes rolling windows, lags, differences et compteurs metier sont generes centralement dans `src.feature_engineering`."),
        code_cell(
            """
            bundle = prepare_raw_datasets(PROJECT_ROOT)
            features = build_feature_table(bundle, PROJECT_ROOT)
            train_df, test_df = temporal_train_test_split(features)

            # ... Nettoyage : prints supprimés ...
            features.filter(regex="rolling|lag|diff|rul|failure_within_24h").head()
            """
        ),
        markdown_cell("## 2. Verifications anti-fuite\n\nLe split est strictement chronologique par machine : les 80% premiers en train, les 20% derniers en test. Aucun echantillon futur n'alimente l'apprentissage du passe."),
        code_cell(
            """
            leakage_checks = []
            for machine_id, group in features.groupby("machineID"):
                train_part = train_df[train_df["machineID"] == machine_id]
                test_part = test_df[test_df["machineID"] == machine_id]
                leakage_checks.append(
                    {
                        "machineID": machine_id,
                        "train_max": train_part["datetime"].max(),
                        "test_min": test_part["datetime"].min(),
                        "is_valid": train_part["datetime"].max() <= test_part["datetime"].min(),
                    }
                )
            leakage_df = pd.DataFrame(leakage_checks)
            leakage_df["is_valid"].value_counts()
            """
        ),
        markdown_cell("## 3. Apercu des variables creees\n\nOn visualise les nouvelles familles de variables pour documenter le pipeline."),
        code_cell(
            """
            engineered_columns = [col for col in features.columns if any(token in col for token in ["rolling", "lag", "diff", "zscore"])]
            # ... Nettoyage : prints supprimés ...
            pd.Series(engineered_columns[:40])
            """
        ),
        code_cell(
            """
            sample_machine = features[features["machineID"] == 1].tail(72)
            plt.figure(figsize=(15, 5))
            plt.plot(sample_machine["datetime"], sample_machine["vibration"], label="Vibration brute", color=COLOR_PALETTE["danger"])
            plt.plot(sample_machine["datetime"], sample_machine["vibration_rolling_mean_24h"], label="Rolling mean 24h", color=COLOR_PALETTE["success"])
            plt.title("Machine 1 - Exemple de smoothing temporel")
            plt.xlabel("Temps")
            plt.ylabel("Vibration")
            plt.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## 4. Cible RUL\n\nLa Remaining Useful Life est calculee comme le nombre d'heures restantes avant la prochaine panne connue."),
        code_cell(
            """
            features[["machineID", "datetime", "rul_hours", "hours_since_last_failure", "next_failure_type"]].dropna().head(10)
            """
        ),
        markdown_cell("## 5. Export\n\nLa table finale est enregistree dans `data/processed/feature_table.csv` pour etre reutilisee par les notebooks suivants et le dashboard."),
    ]
    return nb


def build_classification_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        markdown_cell("# Notebook 03 - Classification panne / pas panne\n\nCe notebook compare une baseline, une regression logistique, une Random Forest et, si disponible, un modele XGBoost."),
        code_cell(PROJECT_IMPORTS),
        markdown_cell("## 1. Entrainement des classifieurs\n\nLa validation utilise exclusivement `TimeSeriesSplit(5)` pour respecter l'ordre temporel."),
        code_cell(
            """
            bundle = prepare_raw_datasets(PROJECT_ROOT)
            features = build_feature_table(bundle, PROJECT_ROOT)
            train_df, test_df = temporal_train_test_split(features)
            feature_columns = get_feature_columns(features)

            classifiers = train_classifiers(train_df)
            list(classifiers)
            """
        ),
        markdown_cell("## 2. Comparaison des performances\n\nLe cout metier d'une panne manquee etant dominant, le seuil de decision est optimise avec un biais en faveur du rappel tout en conservant un F1 robuste."),
        code_cell(
            """
            rows = []
            prediction_frames = {}
            for name, artifact in classifiers.items():
                metrics, preds = evaluate_classifier(artifact, test_df, feature_columns)
                rows.append({"model": name, **metrics})
                prediction_frames[name] = preds
            class_results = pd.DataFrame(rows).sort_values("f1_failure", ascending=False)
            class_results
            """
        ),
        code_cell(
            """
            best_model_name = class_results.iloc[0]["model"]
            best_preds = prediction_frames[best_model_name]
            cm = np.array([[best_preds["cm_tn"].iloc[0], best_preds["cm_fp"].iloc[0]], [best_preds["cm_fn"].iloc[0], best_preds["cm_tp"].iloc[0]]])

            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Matrice de confusion - {best_model_name}")
            plt.xlabel("Prediction")
            plt.ylabel("Reel")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## 3. Importance des variables\n\nUne importance par permutation est fournie en plus de SHAP quand la bibliotheque est disponible."),
        code_cell(
            """
            from sklearn.inspection import permutation_importance

            best_artifact = classifiers[best_model_name]
            X_test = test_df[feature_columns]
            y_test = test_df["failure_within_24h"]
            perm = permutation_importance(best_artifact.pipeline, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE)

            importance_df = pd.DataFrame({"feature": feature_columns, "importance": perm.importances_mean}).sort_values("importance", ascending=False).head(10)
            importance_df
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(8, 5))
            sns.barplot(data=importance_df, x="importance", y="feature", color=COLOR_PALETTE["accent"])
            plt.title("Top 10 features - importance par permutation")
            plt.xlabel("Importance moyenne")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            try:
                import shap

                model = best_artifact.pipeline.named_steps["model"]
                X_transformed = best_artifact.pipeline.named_steps["preprocessor"].transform(X_test)
                feature_names = best_artifact.pipeline.named_steps["preprocessor"].get_feature_names_out()
                sample_size = min(300, X_test.shape[0])
                X_sample = X_transformed[:sample_size]
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)
                shap.plots.beeswarm(shap_values, max_display=10)
            except Exception as exc:
                # ... Nettoyage : prints supprimés ...
                pass
            """
        ),
        markdown_cell("## 4. Conclusion classification\n\nLa comparaison met en evidence le compromis precision-rappel. Les variables de vibration, de pression, les compteurs d'erreurs et la proximite d'une panne jouent un role central dans la decision."),
    ]
    return nb


def build_rul_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        markdown_cell("# Notebook 04 - Regression RUL\n\nCe notebook estime la duree de vie residuelle des equipements et evalue la qualite de l'alerte en zone critique."),
        code_cell(PROJECT_IMPORTS),
        markdown_cell("## 1. Entrainement des modeles RUL\n\nLes modeles compares sont une baseline constante, une Random Forest, un SVR et XGBoost s'il est disponible."),
        code_cell(
            """
            bundle = prepare_raw_datasets(PROJECT_ROOT)
            features = build_feature_table(bundle, PROJECT_ROOT)
            train_df, test_df = temporal_train_test_split(features)
            feature_columns = get_feature_columns(features)

            regressors = train_rul_models(train_df)
            list(regressors)
            """
        ),
        code_cell(
            """
            rul_rows = []
            rul_predictions = {}
            for name, artifact in regressors.items():
                metrics, preds = evaluate_regressor(artifact, test_df, feature_columns)
                rul_rows.append({"model": name, **metrics})
                rul_predictions[name] = preds
            rul_results = pd.DataFrame(rul_rows).sort_values("mae")
            rul_results
            """
        ),
        markdown_cell("## 2. RUL predit vs RUL reel\n\nLe meilleur modele est compare a la diagonale ideale afin d'evaluer les sous-estimations et surestimations."),
        code_cell(
            """
            best_rul_model = rul_results.iloc[0]["model"]
            best_rul_preds = rul_predictions[best_rul_model]

            plt.figure(figsize=(7, 6))
            plt.scatter(best_rul_preds["rul_hours"], best_rul_preds["predicted_rul"], alpha=0.4, color=COLOR_PALETTE["info"])
            max_val = float(max(best_rul_preds["rul_hours"].max(), best_rul_preds["predicted_rul"].max()))
            plt.plot([0, max_val], [0, max_val], linestyle="--", color=COLOR_PALETTE["danger"], label="Ligne ideale")
            plt.title(f"RUL reel vs predit - {best_rul_model}")
            plt.xlabel("RUL reel")
            plt.ylabel("RUL predit")
            plt.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(8, 4))
            sns.histplot(best_rul_preds["residual"], bins=40, kde=True, color=COLOR_PALETTE["accent"])
            plt.title("Distribution des residus RUL")
            plt.xlabel("Erreur de prediction")
            plt.ylabel("Frequence")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## 3. Zone critique\n\nNous evaluons explicitement les observations dont la RUL reelle est inferieure au seuil critique."),
        code_cell(
            """
            critical_zone = best_rul_preds[best_rul_preds["rul_hours"] < CRITICAL_RUL_THRESHOLD]
            critical_zone.head()
            """
        ),
        markdown_cell("## 4. Conclusion RUL\n\nLa performance en MAE globale est completee par une lecture metier de la zone critique, qui est la plus importante pour la maintenance anticipative."),
    ]
    return nb


def build_final_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        markdown_cell("# Notebook 05 - Evaluation finale\n\nCe notebook orchestre le pipeline complet, sauvegarde les modeles et consolide les resultats finaux."),
        code_cell(PROJECT_IMPORTS),
        markdown_cell("## 1. Execution du pipeline complet\n\nCette cellule produit la table de features, les modeles entraines, les predicions et le resume des metriques pour le dashboard."),
        code_cell(
            """
            artifacts = run_full_training(PROJECT_ROOT)
            # ... Nettoyage : prints supprimés ...
            """
        ),
        markdown_cell("## 2. Resume des metriques\n\nLes metriques de classification et de regression sont affichees avec quatre decimales pour faciliter la comparaison en soutenance."),
        code_cell(
            """
            summary_path = PROJECT_ROOT / "data" / "processed" / "metrics_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            pd.DataFrame(summary["classification"]).T
            """
        ),
        code_cell(
            """
            pd.DataFrame(summary["regression"]).T
            """
        ),
        markdown_cell("## 3. Verification des artefacts\n\nLes modeles `joblib`, les predictions flotte et les tableaux de metriques doivent etre presents dans les dossiers de sortie."),
        code_cell(
            """
            sorted(str(path.relative_to(PROJECT_ROOT)) for path in PROJECT_ROOT.rglob("*") if path.is_file())[:40]
            """
        ),
        markdown_cell("## 4. Conclusion generale\n\nLe pipeline tourne de bout en bout et alimente directement le dashboard Streamlit. Le projet est donc pret pour la demonstration et la redaction finale."),
    ]
    return nb


def main() -> None:
    project_root = Path(__file__).resolve().parent
    notebooks_dir = project_root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    notebooks = {
        "01_EDA.ipynb": build_eda_notebook(),
        "02_feature_engineering.ipynb": build_feature_notebook(),
        "03_classification_panne.ipynb": build_classification_notebook(),
        "04_regression_RUL.ipynb": build_rul_notebook(),
        "05_evaluation_finale.ipynb": build_final_notebook(),
    }
    for file_name, notebook in notebooks.items():
        nbf.write(notebook, notebooks_dir / file_name)


if __name__ == "__main__":
    main()
