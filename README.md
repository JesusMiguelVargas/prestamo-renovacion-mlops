# Prestamo Renovacion MLOps

Proyecto Final del curso MLOps — Pipeline end-to-end para predicción de renovación de línea de crédito bancaria.

## Inicio Rápido — GitHub Codespaces

### Paso 1 — Abrir el Codespace
Haz clic en **Code → Codespaces → Create codespace on main**.

Al abrirse, el Codespace ejecuta automáticamente:
```bash
pip install -r requirements.txt && cp .env.example .env.preprod
```
Espera a que termine (barra de progreso en la terminal).

### Paso 2 — Levantar el stack completo
```bash
docker compose -f docker-compose.preprod.yml up --build
```
Esto levanta 3 servicios en orden:
1. **MLflow** (puerto 5000) — servidor de tracking de experimentos
2. **Trainer** — entrena el modelo (~5 min) y lo guarda en `artifacts/model.pkl`
3. **API** (puerto 8000) — sirve predicciones con el modelo entrenado

Cuando veas `prestamo-trainer exited with code 0`, todo está listo.

### Paso 3 — Acceder a los servicios
Abre la pestaña **PUERTOS** en VS Code y haz clic en los links:

| Puerto | Servicio | URL de ejemplo |
|---|---|---|
| **8000** | API FastAPI | `/docs` para Swagger UI interactivo |
| **5000** | MLflow UI | Experimentos y métricas del modelo |

> **Nota:** Si `/docs` da 404, clic derecho en el puerto 8000 → **Visibilidad → Public**.

## Caso de Uso

**Objetivo:** Predecir si un cliente renovará su línea de crédito, permitiendo al banco focalizar campañas comerciales en los clientes con mayor probabilidad de renovación.

| Atributo | Detalle |
|---|---|
| Dataset | Dataset Renovacion_prestamo.csv |
| Filas | 87,556 clientes |
| Target | `FLAG_VENTA` (0=No renueva, 1=Renueva) |
| Desbalanceo | 96% No renueva / 4% Renueva |
| Métrica principal | **Recall** (minimizar falsos negativos) |

## Arquitectura MLOps

```
_data/run.py          → Etapa 1: Carga y validación del CSV raw
preprocess/run.py     → Etapa 2: Limpieza, imputación, encoding
segregate/run.py      → Etapa 3: Split estratificado train/test (80/20)
check_data/           → Etapa 4: 16 tests de validación con pytest
train/run.py          → Etapa 5: GridSearchCV + MLflow tracking
evaluate/run.py       → Etapa 6: Métricas + quality gate
drift/run.py          → Etapa 7: EvidentlyAI drift detection
serve/app.py          → FastAPI: POST /predict GET /health GET /docs
.github/workflows/    → CI/CD con GitHub Actions
```

## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| ML | scikit-learn (RandomForest + GridSearchCV) |
| Tracking | MLflow |
| Serving | FastAPI + Uvicorn |
| Contenedor | Docker |
| Drift | EvidentlyAI |
| Tests | pytest |
| CI/CD | GitHub Actions |

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecutar el pipeline completo

```bash
# Todas las etapas
python main.py

# O con Make
make pipeline
```

## Ejecutar etapas individuales

```bash
# Etapa 1: Cargar dataset
python _data/run.py

# Etapa 2: Preprocesar
python preprocess/run.py

# Etapa 3: Split train/test
python segregate/run.py

# Etapa 4: Validar datos
python -m pytest check_data/ -v

# Etapa 5: Entrenar modelo
python train/run.py

# Etapa 6: Evaluar
python evaluate/run.py

# Etapa 7: Drift detection
python drift/run.py
```

## Levantar la API

```bash
# Directamente
uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload

# O con Make
make serve
```

Endpoints disponibles:
- `GET  /health` — estado del modelo
- `POST /predict` — predicción
- `GET  /docs` — Swagger UI interactivo

## Ver experimentos en MLflow

```bash
mlflow ui --host 0.0.0.0 --port 5000
# Abrir: http://localhost:5000
```

## Docker

```bash
# Construir imagen
make docker-build

# Correr API en Docker
make docker-run
```

## Estructura del proyecto

```
prestamo-renovacion-mlops/
├── .github/workflows/ml_pipeline.yml   CI/CD GitHub Actions
├── _data/run.py                         Etapa 1: ingestión
├── preprocess/run.py                    Etapa 2: preprocesamiento
├── segregate/run.py                     Etapa 3: split
├── check_data/
│   ├── conftest.py                      fixtures pytest
│   └── test_data.py                     16 tests de validación
├── train/run.py                         Etapa 5: entrenamiento
├── evaluate/run.py                      Etapa 6: evaluación
├── drift/run.py                         Etapa 7: drift detection
├── serve/
│   ├── app.py                           FastAPI
│   ├── predictor.py                     carga y predicción
│   ├── schemas.py                       Pydantic schemas
│   └── Dockerfile                       imagen Docker
├── data/
│   └── Dataset Renovacion_prestamo.csv  dataset original
├── artifacts/                           generado por pipeline (no en Git)
├── config.yaml                          configuración centralizada
├── main.py                              orquestador del pipeline
├── Makefile                             comandos útiles
└── requirements.txt                     dependencias
```

## Quality Gate

El pipeline falla automáticamente si el modelo no cumple:

| Métrica | Umbral mínimo |
|---|---|
| **Recall** | ≥ 0.55 |
| **ROC-AUC** | ≥ 0.65 |

## Autor

Proyecto Final MLOps — compartido con: [jcbf08](https://github.com/jcbf08)
