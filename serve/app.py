"""FastAPI prediction service for Prestamo Renovacion model.

Endpoints:
  GET  /         — API info
  GET  /health   — health check + model status
  POST /predict  — prediction
  GET  /docs     — Swagger UI (auto-generated)
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from serve.predictor import predictor
from serve.schemas import ClienteInput, HealthResponse, PrediccionOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Cargando modelo desde: %s", MODEL_PATH)
    try:
        predictor.load(MODEL_PATH)
        log.info("Modelo listo: %s", type(predictor.model).__name__)
    except FileNotFoundError as e:
        log.error("No se pudo cargar el modelo: %s", e)
    yield
    log.info("API cerrando")


app = FastAPI(
    title="API Renovación de Préstamo — MLOps",
    description=(
        "Predicción de renovación de línea de crédito bancaria. "
        "Caso de uso: Banco — Proyecto Final MLOps."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["General"])
def root():
    return {
        "api": "Renovación de Préstamo MLOps",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return HealthResponse(
        status="ok",
        modelo=type(predictor.model).__name__,
    )


@app.post("/predict", response_model=PrediccionOutput, tags=["Predicción"])
def predict(cliente: ClienteInput):
    """Predice si un cliente renovará su línea de crédito."""
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    try:
        result = predictor.predict(cliente.model_dump())
        return PrediccionOutput(**result)
    except Exception as e:
        log.error("Error en predicción: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve.app:app", host="0.0.0.0", port=8000, reload=False)
