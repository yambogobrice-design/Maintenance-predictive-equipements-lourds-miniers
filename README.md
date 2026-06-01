# Projet M3 - Maintenance predictive des equipements lourds miniers

Projet realise pour la fiche **M3 - Mines** du cours d'Introduction au Machine Learning.  
Objectif: predire les pannes d'equipements lourds avant leur apparition, estimer la RUL
(`Remaining Useful Life`) et fournir un dashboard de decision pour la maintenance.

## Conformite avec la fiche M3

- Donnees: Microsoft Azure Predictive Maintenance + reference AI4I 2020.
- Famille ML: series temporelles, classification binaire, regression RUL.
- Feature engineering: rolling mean/std, lag features, differences, z-score par machine.
- Modeles: Logistic Regression, Random Forest, SVR, baselines, XGBoost pret a etre active.
- Validation: split temporel strict, entrainement sur le passe et test sur le futur.
- Livrables: pipeline complet, modeles sauvegardes, dashboard Streamlit, rapport LaTeX.

## Structure

```text
ProjetML/
|-- data/
|   |-- raw/
|   `-- processed/
|-- dashboard/
|   `-- app.py
|-- models/
|-- notebooks/
|-- rapport/
|   |-- rapport_M3.md
|   |-- rapport_M3.pdf
|   `-- rapport_M3.tex
|-- src/
|-- generate_notebooks.py
|-- run_pipeline.py
|-- requirements.txt
`-- README.md
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Reproduire le pipeline

```bash
python run_pipeline.py
```

La commande regenere:

- `data/processed/feature_table.csv`
- `data/processed/fleet_predictions.csv`
- `data/processed/fleet_snapshot.csv`
- `data/processed/metrics_summary.json`
- `models/*.joblib`

## Lancer le dashboard

```bash
python -m streamlit run dashboard/app.py
```

Le dashboard contient trois vues:

- **Decision flotte**: KPI, machine prioritaire, file d'actions maintenance.
- **Diagnostic machine**: risque, RUL, capteurs et facteurs de vigilance.
- **Performance modeles**: metriques classification panne et regression RUL.

## Rapport

Le rapport LaTeX est disponible dans:

```text
rapport/rapport_M3.tex
```

Les figures du rapport peuvent etre regenerees avec:

```bash
python scripts/generate_report_figures.py
```

Compilation possible avec:

```bash
pdflatex rapport/rapport_M3.tex
```

## Resultats actuels

Le pipeline selectionne automatiquement le meilleur classifieur non trivial selon le F1 de la classe panne,
puis le meilleur regresseur selon la MAE RUL. Les seuils operationnels du dashboard sont:

- critique si probabilite de panne >= 65% ou RUL <= 20 h;
- surveillance si probabilite de panne >= 40% ou RUL <= 72 h.

Derniere execution du pipeline:

- meilleur modele panne: Random Forest, precision panne = 0.377, rappel panne = 0.953, F1 panne = 0.541;
- meilleur modele RUL: Extra Trees, MAE = 120.83 h;
- dashboard Streamlit alimente par les predictions flotte, la priorisation machine et les recommandations d'action.

## Auteur

Brice Yambogo  
2iE - Institut International d'Ingenierie de l'Eau et de l'Environnement
