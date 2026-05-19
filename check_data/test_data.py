"""Stage 4 — Data validation tests.

Validates schema, null counts, value ranges, and class distribution
in both train and test splits before model training begins.
"""
import pandas as pd
import pytest


# ── Schema ────────────────────────────────────────────────────────────────────

def test_train_not_empty(df_train):
    assert len(df_train) > 0, "El dataset de entrenamiento está vacío"


def test_test_not_empty(df_test):
    assert len(df_test) > 0, "El dataset de test está vacío"


def test_train_has_target(df_train, target):
    assert target in df_train.columns, f"Columna target '{target}' ausente en train"


def test_test_has_target(df_test, target):
    assert target in df_test.columns, f"Columna target '{target}' ausente en test"


def test_same_columns(df_train, df_test):
    assert set(df_train.columns) == set(df_test.columns), (
        f"Train y test tienen columnas distintas:\n"
        f"Solo en train: {set(df_train.columns) - set(df_test.columns)}\n"
        f"Solo en test: {set(df_test.columns) - set(df_train.columns)}"
    )


# ── Nulls ─────────────────────────────────────────────────────────────────────

def test_no_nulls_in_train_features(df_train, target):
    feature_cols = [c for c in df_train.columns if c != target]
    nulls = df_train[feature_cols].isnull().sum()
    assert nulls.sum() == 0, f"Nulls en features de train:\n{nulls[nulls > 0]}"


def test_no_nulls_in_test_features(df_test, target):
    feature_cols = [c for c in df_test.columns if c != target]
    nulls = df_test[feature_cols].isnull().sum()
    assert nulls.sum() == 0, f"Nulls en features de test:\n{nulls[nulls > 0]}"


# ── Target values ─────────────────────────────────────────────────────────────

def test_target_binary_train(df_train, target):
    vals = set(df_train[target].unique())
    assert vals <= {0, 1}, f"TARGET en train contiene valores inesperados: {vals}"


def test_target_binary_test(df_test, target):
    vals = set(df_test[target].unique())
    assert vals <= {0, 1}, f"TARGET en test contiene valores inesperados: {vals}"


def test_both_classes_in_train(df_train, target):
    counts = df_train[target].value_counts()
    assert 0 in counts and 1 in counts, "Train solo tiene una clase — falla el balance"


def test_both_classes_in_test(df_test, target):
    counts = df_test[target].value_counts()
    assert 0 in counts and 1 in counts, "Test solo tiene una clase"


# ── Value ranges ──────────────────────────────────────────────────────────────

def test_edad_range(df_train):
    if "EDAD" in df_train.columns:
        assert df_train["EDAD"].between(18, 100).all(), "EDAD fuera del rango [18, 100]"


def test_linea_renovado_positive(df_train):
    if "LINEA_RENOVADO" in df_train.columns:
        assert (df_train["LINEA_RENOVADO"] > 0).all(), "LINEA_RENOVADO tiene valores <= 0"


def test_sexo_binary(df_train):
    if "SEXO" in df_train.columns:
        assert df_train["SEXO"].isin([0, 1]).all(), "SEXO debe ser binario {0, 1}"


# ── Size ratio ────────────────────────────────────────────────────────────────

def test_split_ratio(df_train, df_test):
    total = len(df_train) + len(df_test)
    test_ratio = len(df_test) / total
    assert 0.15 <= test_ratio <= 0.35, (
        f"Ratio test={test_ratio:.2f} fuera del rango esperado [0.15, 0.35]"
    )
