"""Stage 1 — Download / validate raw data.

Reads the raw CSV, validates required columns exist and basic integrity,
then saves a parquet snapshot so downstream stages are format-agnostic.
"""
import logging
import os
import sys

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

REQUIRED_COLS = {
    "MES", "CLIENTE", "LINEA_RENOVADO", "PLAZO_RENOVADO", "FLAG_VENTA",
    "USO_LINEA_TOTAL_TC_T2", "USO_TRIM_LINEA_BBVA", "NR_ENTIDADES_TOTAL_T2",
    "DIFF_NRO_ENTIDA_TOTALES_T2_T12", "SDO_CONSUMO_T2",
    "RESENCIA_OFERTA_PLD_RENOVADO", "Ahorro_Sldo_Bco_T1", "PConsumo_Sldo_Bco_T1",
    "SDO_BCO_tot_sm_pasivo_Bco_6M", "EDAD", "SEXO", "EST_CIVIL",
    "ANTIGUEDAD_MES", "REGION", "FLAG_LIMA_PROVINCIA", "SUELDO_ESTIMADO",
    "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD",
}


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict) -> None:
    raw_path = cfg["data"]["raw_path"]
    sep = cfg["data"]["sep"]
    out_path = cfg["data"]["processed_path"]

    log.info("Cargando dataset desde: %s", raw_path)
    if not os.path.exists(raw_path):
        log.error("Archivo no encontrado: %s", raw_path)
        sys.exit(1)

    df = pd.read_csv(raw_path, sep=sep, low_memory=False)
    log.info("Dataset cargado: %d filas × %d columnas", *df.shape)

    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        log.error("Columnas faltantes: %s", missing_cols)
        sys.exit(1)

    assert df["FLAG_VENTA"].isin([0, 1]).all(), "FLAG_VENTA contiene valores fuera de {0,1}"
    assert df["EDAD"].dropna().between(18, 100).all(), "EDAD fuera de rango esperado"
    assert df["LINEA_RENOVADO"].gt(0).all(), "LINEA_RENOVADO debe ser positivo"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Raw data guardado en: %s", out_path)

    positive_rate = df["FLAG_VENTA"].mean() * 100
    log.info(
        "Distribución target — Renovación: %.2f%% | Sin renovación: %.2f%%",
        positive_rate, 100 - positive_rate,
    )


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
