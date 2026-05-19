"""Stage 2 — Preprocessing.

Cleans nulls, winsorizes outliers, encodes categoricals, and saves
the cleaned dataset ready for splitting.
"""
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def winsorize(df: pd.DataFrame, cols: list, low: float, high: float) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            lo = df[col].quantile(low)
            hi = df[col].quantile(high)
            df[col] = df[col].clip(lo, hi)
    return df


def impute_nulls(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> pd.DataFrame:
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            log.info("  Imputado %s con mediana=%.2f", col, median)

    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
            log.info("  Imputado %s con moda='%s'", col, mode)

    return df


def encode_categoricals(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    # Binary encode SEXO: M→1, F→0
    if "SEXO" in df.columns:
        df["SEXO"] = df["SEXO"].map({"M": 1, "F": 0}).fillna(0).astype(int)

    # One-hot encode remaining categoricals
    ohe_cols = [c for c in categorical_cols if c in df.columns and c != "SEXO"]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    return df


def run(cfg: dict) -> None:
    in_path = cfg["data"]["processed_path"]   # raw parquet from stage 1
    out_path = cfg["data"]["processed_path"]  # overwrite with clean version

    prep = cfg["preprocessing"]
    target = prep["target"]
    drop_cols = prep["drop_cols"]
    numeric_cols = prep["numeric_cols"]
    categorical_cols = prep["categorical_cols"]
    winsor_low = prep["winsor_low"]
    winsor_high = prep["winsor_high"]

    log.info("Cargando datos desde: %s", in_path)
    df = pd.read_parquet(in_path)
    log.info("Filas iniciales: %d", len(df))

    # Drop ID/metadata columns
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    log.info("Columnas eliminadas: %s", drop_cols)

    # Nulls before imputation
    null_counts = df[numeric_cols + categorical_cols].isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        log.info("Nulls detectados:\n%s", null_counts.to_string())

    # Impute
    df = impute_nulls(df, numeric_cols, categorical_cols)

    # Winsorize
    df = winsorize(df, numeric_cols, winsor_low, winsor_high)
    log.info("Winsorización aplicada [%.0f%%–%.0f%%]", winsor_low * 100, winsor_high * 100)

    # Encode categoricals
    df = encode_categoricals(df, categorical_cols)
    log.info("Encoding aplicado. Columnas totales: %d", df.shape[1])

    # Verify no nulls remain in features
    feature_cols = [c for c in df.columns if c != target]
    remaining_nulls = df[feature_cols].isnull().sum().sum()
    if remaining_nulls > 0:
        log.error("Quedan %d nulls tras imputación", remaining_nulls)
        sys.exit(1)

    df.to_parquet(out_path, index=False)
    log.info("Datos limpios guardados en: %s — %d filas × %d cols", out_path, *df.shape)


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
