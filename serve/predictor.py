"""Model loader and predictor for the FastAPI service."""
import logging
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")


class Predictor:
    def __init__(self):
        self.model = None
        self.feature_names: list = []

    def load(self, path: str = MODEL_PATH) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo no encontrado: {path}")
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        log.info("Modelo cargado: %s | Features: %d", type(self.model).__name__, len(self.feature_names))

    def predict(self, data: dict) -> dict:
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Llama a predictor.load() primero.")

        df = pd.DataFrame([data])

        # Align to training feature set
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]

        # Fill any remaining nulls with 0 (median already applied at training time)
        df = df.fillna(0)

        pred = int(self.model.predict(df)[0])
        proba = float(self.model.predict_proba(df)[0][1])

        return {
            "renueva": pred,
            "probabilidad_renovacion": round(proba, 4),
            "modelo": type(self.model).__name__,
        }


predictor = Predictor()
