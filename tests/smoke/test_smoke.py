"""Smoke tests — verifican que los servicios del stack están vivos.

Se corren DESPUÉS de docker compose up --build.
No prueban lógica de negocio, solo que cada servicio responde.
"""
import pytest
import requests

API_URL = "http://localhost:8000"
MLFLOW_URL = "http://localhost:5000"


# ── API ───────────────────────────────────────────────────────────────────────

def test_api_root_responde():
    """La API responde en el endpoint raíz."""
    r = requests.get(f"{API_URL}/", timeout=10)
    assert r.status_code == 200


def test_api_health_ok():
    """El health check confirma que el modelo está cargado."""
    r = requests.get(f"{API_URL}/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "modelo" in body


def test_api_docs_disponibles():
    """Swagger UI está disponible."""
    r = requests.get(f"{API_URL}/docs", timeout=10)
    assert r.status_code == 200


def test_api_prediccion_cliente_tipo():
    """El endpoint /predict responde con predicción válida."""
    cliente = {
        "LINEA_RENOVADO": 15000,
        "PLAZO_RENOVADO": 36,
        "USO_LINEA_TOTAL_TC_T2": 0.45,
        "USO_TRIM_LINEA_BBVA": 0.30,
        "NR_ENTIDADES_TOTAL_T2": 3,
        "DIFF_NRO_ENTIDA_TOTALES_T2_T12": 0,
        "SDO_CONSUMO_T2": 5000.0,
        "RESENCIA_OFERTA_PLD_RENOVADO": 6.0,
        "Ahorro_Sldo_Bco_T1": 1200.0,
        "PConsumo_Sldo_Bco_T1": 8000.0,
        "SDO_BCO_tot_sm_pasivo_Bco_6M": 950.0,
        "EDAD": 42.0,
        "SEXO": 1,
        "ANTIGUEDAD_MES": 120.0,
        "FLAG_LIMA_PROVINCIA": 1,
        "SUELDO_ESTIMADO": 4500.0,
        "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": 0.65,
        "EST_CIVIL_DIVORCIADO": 0,
        "EST_CIVIL_SEPARADO": 0,
        "EST_CIVIL_SOLTERO": 0,
        "EST_CIVIL_UNION_LIBRE": 0,
        "EST_CIVIL_VIUDO": 0
    }
    r = requests.post(f"{API_URL}/predict", json=cliente, timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["renueva"] in [0, 1]
    assert 0.0 <= body["probabilidad_renovacion"] <= 1.0
    assert "modelo" in body


def test_api_prediccion_cliente_joven():
    """Prueba con perfil de cliente joven y línea baja."""
    cliente = {
        "LINEA_RENOVADO": 5000,
        "PLAZO_RENOVADO": 12,
        "USO_LINEA_TOTAL_TC_T2": 0.10,
        "USO_TRIM_LINEA_BBVA": 0.05,
        "NR_ENTIDADES_TOTAL_T2": 1,
        "DIFF_NRO_ENTIDA_TOTALES_T2_T12": 1,
        "SDO_CONSUMO_T2": 1000.0,
        "RESENCIA_OFERTA_PLD_RENOVADO": 2.0,
        "Ahorro_Sldo_Bco_T1": 500.0,
        "PConsumo_Sldo_Bco_T1": 3000.0,
        "SDO_BCO_tot_sm_pasivo_Bco_6M": 300.0,
        "EDAD": 25.0,
        "SEXO": 0,
        "ANTIGUEDAD_MES": 24.0,
        "FLAG_LIMA_PROVINCIA": 1,
        "SUELDO_ESTIMADO": 2000.0,
        "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": 0.30,
        "EST_CIVIL_DIVORCIADO": 0,
        "EST_CIVIL_SEPARADO": 0,
        "EST_CIVIL_SOLTERO": 1,
        "EST_CIVIL_UNION_LIBRE": 0,
        "EST_CIVIL_VIUDO": 0
    }
    r = requests.post(f"{API_URL}/predict", json=cliente, timeout=10)
    assert r.status_code == 200


# ── MLflow ────────────────────────────────────────────────────────────────────

def test_mlflow_ui_responde():
    """MLflow UI está corriendo en el puerto 5000."""
    r = requests.get(f"{MLFLOW_URL}/", timeout=10)
    assert r.status_code == 200


def test_mlflow_experimento_existe():
    """El experimento del pipeline fue registrado en MLflow."""
    r = requests.get(
        f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
        timeout=10
    )
    assert r.status_code == 200
    experiments = r.json().get("experiments", [])
    nombres = [e["name"] for e in experiments]
    assert any("prestamo" in n.lower() for n in nombres), (
        f"Experimento 'prestamo' no encontrado. Experimentos: {nombres}"
    )
