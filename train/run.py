"""Stage 5 — Model training.

Runs GridSearchCV over RandomForestClassifier, logs every combination
to MLflow, and registers the best model in MLflow Model Registry.
"""
import logging
import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODEL_REGISTRY_NAME = "PrestamosRenovacionModel"


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_param_grid(raw: dict) -> dict:
    """Convert None strings from YAML to Python None for max_depth."""
    grid = {}
    for k, v in raw.items():
        grid[k] = [None if x == "null" or x is None else x for x in v]
    return grid


def run(cfg: dict) -> None:
    train_path = cfg["data"]["train_path"]
    target = cfg["preprocessing"]["target"]
    train_cfg = cfg["train"]
    model_artifact = train_cfg["model_artifact"]

    log.info("Cargando datos de entrenamiento desde: %s", train_path)
    df = pd.read_parquet(train_path)
    X = df.drop(columns=[target])
    y = df[target]

    log.info("Train shape: %s | Positivos: %.2f%%", X.shape, y.mean() * 100)

    mlflow.set_tracking_uri(cfg["train"].get("tracking_uri", "mlruns"))
    mlflow.set_experiment(train_cfg["experiment_name"])

    param_grid = build_param_grid(train_cfg["param_grid"])
    cv = StratifiedKFold(
        n_splits=train_cfg["cv_folds"],
        shuffle=True,
        random_state=train_cfg["random_state"],
    )

    base_model = RandomForestClassifier(random_state=train_cfg["random_state"])

    log.info("Iniciando GridSearchCV — %d combinaciones × %d folds",
             _count_combinations(param_grid), train_cfg["cv_folds"])

    gs = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring=train_cfg["scoring"],
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    with mlflow.start_run(run_name=train_cfg["run_name"]) as run:
        gs.fit(X, y)

        best = gs.best_estimator_
        best_params = gs.best_params_
        best_cv_score = gs.best_score_

        log.info("Mejores parámetros: %s", best_params)
        log.info("Mejor %s en CV: %.4f", train_cfg["scoring"], best_cv_score)

        mlflow.log_params(best_params)
        mlflow.log_metric(f"cv_{train_cfg['scoring']}", best_cv_score)
        mlflow.log_param("cv_folds", train_cfg["cv_folds"])
        mlflow.log_param("scoring", train_cfg["scoring"])
        mlflow.log_param("n_combinations", _count_combinations(param_grid))

        # Log all CV results as child runs for comparison
        results = pd.DataFrame(gs.cv_results_)
        for _, row in results.iterrows():
            with mlflow.start_run(run_name="grid_combo", nested=True):
                for p in param_grid:
                    mlflow.log_param(p, row[f"param_{p}"])
                mlflow.log_metric(f"mean_test_{train_cfg['scoring']}",
                                  row[f"mean_test_score"])

        mlflow.sklearn.log_model(
            best,
            artifact_path="random_forest_model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )

        run_id = run.info.run_id
        log.info("MLflow run_id: %s", run_id)

    # Save model artifact locally so evaluate/serve can use it
    os.makedirs(os.path.dirname(model_artifact), exist_ok=True)
    with open(model_artifact, "wb") as f:
        pickle.dump({"model": best, "feature_names": list(X.columns)}, f)

    log.info("Modelo guardado en: %s", model_artifact)


def _count_combinations(param_grid: dict) -> int:
    count = 1
    for v in param_grid.values():
        count *= len(v)
    return count


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
