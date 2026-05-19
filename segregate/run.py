"""Stage 3 — Segregate.

Stratified train/test split. Saves train.parquet and test.parquet.
"""
import logging
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict) -> None:
    in_path = cfg["data"]["processed_path"]
    train_path = cfg["data"]["train_path"]
    test_path = cfg["data"]["test_path"]
    target = cfg["preprocessing"]["target"]
    test_size = cfg["segregate"]["test_size"]
    random_state = cfg["segregate"]["random_state"]

    log.info("Cargando datos limpios desde: %s", in_path)
    df = pd.read_parquet(in_path)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train = X_train.copy()
    train[target] = y_train.values
    test = X_test.copy()
    test[target] = y_test.values

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)

    log.info("Train: %d filas (positivos: %.2f%%)", len(train), y_train.mean() * 100)
    log.info("Test:  %d filas (positivos: %.2f%%)", len(test), y_test.mean() * 100)
    log.info("Splits guardados en: %s / %s", train_path, test_path)


if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
