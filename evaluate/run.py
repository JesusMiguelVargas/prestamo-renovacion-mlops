"""Stage 6 — Model evaluation + quality gate.

Computes full metrics on the held-out test set, saves metrics.json,
generates a confusion matrix plot, and exits with code 1 if quality
thresholds are not met (recall < recall_min or roc_auc < roc_auc_min).
"""
import json
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict) -> None:
    test_path = cfg["data"]["test_path"]
    target = cfg["preprocessing"]["target"]
    model_artifact = cfg["train"]["model_artifact"]
    eval_cfg = cfg["evaluate"]
    metrics_path = eval_cfg["metrics_path"]
    plot_path = eval_cfg["plot_path"]
    recall_min = eval_cfg["recall_min"]
    roc_auc_min = eval_cfg["roc_auc_min"]

    log.info("Cargando test set desde: %s", test_path)
    df = pd.read_parquet(test_path)
    X_test = df.drop(columns=[target])
    y_test = df[target]

    log.info("Cargando modelo desde: %s", model_artifact)
    with open(model_artifact, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    feature_names = artifact["feature_names"]

    # Align columns — in case one-hot produced extra/missing cols
    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_names]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    log.info("Métricas de evaluación:")
    for k, v in metrics.items():
        log.info("  %-25s %s", k, v)

    log.info("\n%s", classification_report(y_test, y_pred, target_names=["No renueva", "Renueva"]))

    # Confusion matrix plot
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No renueva", "Renueva"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Matriz de Confusión — Test Set")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()
    log.info("Matriz de confusión guardada en: %s", plot_path)

    # Save metrics JSON
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Métricas guardadas en: %s", metrics_path)

    # Log to MLflow (attach to last active run if exists)
    try:
        mlflow.set_tracking_uri(cfg["train"].get("tracking_uri", "mlruns"))
        mlflow.set_experiment(cfg["train"]["experiment_name"])
        with mlflow.start_run(run_name="evaluate"):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(plot_path)
            mlflow.log_artifact(metrics_path)
    except Exception as e:
        log.warning("No se pudo loggear en MLflow: %s", e)

    # Quality gate
    passed = True
    if rec < recall_min:
        log.error("QUALITY GATE FAIL — Recall=%.4f < mínimo=%.4f", rec, recall_min)
        passed = False
    else:
        log.info("QUALITY GATE OK  — Recall=%.4f >= mínimo=%.4f", rec, recall_min)

    if auc < roc_auc_min:
        log.error("QUALITY GATE FAIL — ROC-AUC=%.4f < mínimo=%.4f", auc, roc_auc_min)
        passed = False
    else:
        log.info("QUALITY GATE OK  — ROC-AUC=%.4f >= mínimo=%.4f", auc, roc_auc_min)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
