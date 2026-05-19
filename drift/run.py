"""Stage 7 — Data drift detection with EvidentlyAI.

Compares the train distribution vs the test distribution and generates
an HTML drift report.
"""
import logging
import os

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict) -> None:
    train_path = cfg["data"]["train_path"]
    test_path = cfg["data"]["test_path"]
    target = cfg["preprocessing"]["target"]
    report_path = cfg["drift"]["report_path"]

    log.info("Cargando splits para análisis de drift...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    feature_cols = [c for c in df_train.columns if c != target]
    ref = df_train[feature_cols]
    cur = df_test[feature_cols]

    try:
        # evidently ≥0.7 moved legacy API under evidently.legacy
        from evidently.legacy.metric_preset import DataDriftPreset
        from evidently.legacy.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report.save_html(report_path)
        log.info("Reporte de drift guardado en: %s", report_path)

        result = report.as_dict()
        for m in result.get("metrics", []):
            if m.get("metric") == "DatasetDriftMetric":
                r = m.get("result", {})
                dataset_drift = r.get("dataset_drift", False)
                n_drifted = r.get("number_of_drifted_columns", 0)
                share = r.get("share_of_drifted_columns", 0.0)
                log.info(
                    "Dataset drift: %s | Columnas con drift: %d (%.1f%%)",
                    dataset_drift, n_drifted, share * 100,
                )
                if dataset_drift:
                    log.warning("DRIFT DETECTADO — revisar reporte: %s", report_path)
                else:
                    log.info("Sin drift significativo detectado.")

    except ImportError:
        log.warning("evidently no disponible. Ejecutando comparación estadística básica.")
        numeric_cols = ref.select_dtypes(include="number").columns
        for col in numeric_cols[:10]:
            train_mean = ref[col].mean()
            test_mean = cur[col].mean()
            diff_pct = abs(train_mean - test_mean) / (abs(train_mean) + 1e-9) * 100
            log.info("  %-45s train=%.3f  test=%.3f  diff=%.1f%%",
                     col, train_mean, test_mean, diff_pct)


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
