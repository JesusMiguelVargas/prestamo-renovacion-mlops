"""pytest fixtures shared across check_data tests."""
import os

import pandas as pd
import pytest
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def df_train(cfg):
    path = cfg["data"]["train_path"]
    assert os.path.exists(path), f"Train parquet no encontrado: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def df_test(cfg):
    path = cfg["data"]["test_path"]
    assert os.path.exists(path), f"Test parquet no encontrado: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def target(cfg):
    return cfg["preprocessing"]["target"]
