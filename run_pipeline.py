"""Convenience entry point to execute the full M3 training pipeline."""

from __future__ import annotations

from pathlib import Path

from src.project_pipeline import run_full_training


def main() -> None:
    project_root = Path(__file__).resolve().parent
    artifacts = run_full_training(project_root)
    # ... Nettoyage : prints supprimés ...


if __name__ == "__main__":
    main()
