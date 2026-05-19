.PHONY: install pipeline download preprocess segregate check_data train evaluate drift serve docker-build docker-run clean help

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Pipeline stages ───────────────────────────────────────────────────────────
pipeline:
	python main.py

download:
	python _data/run.py

preprocess:
	python preprocess/run.py

segregate:
	python segregate/run.py

check_data:
	python -m pytest check_data/ -v --tb=short

train:
	python train/run.py

evaluate:
	python evaluate/run.py

drift:
	python drift/run.py

# ── Serving ───────────────────────────────────────────────────────────────────
serve:
	uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -f serve/Dockerfile -t prestamo-renovacion-api:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/artifacts:/app/artifacts prestamo-renovacion-api:latest

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf artifacts/ mlruns/ __pycache__ **/__pycache__ .pytest_cache

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  make install      Instala dependencias"
	@echo "  make pipeline     Ejecuta todo el pipeline"
	@echo "  make download     Etapa 1: carga del dataset"
	@echo "  make preprocess   Etapa 2: limpieza y encoding"
	@echo "  make segregate    Etapa 3: split train/test"
	@echo "  make check_data   Etapa 4: tests de validación"
	@echo "  make train        Etapa 5: entrenamiento + MLflow"
	@echo "  make evaluate     Etapa 6: métricas + quality gate"
	@echo "  make drift        Etapa 7: detección de drift"
	@echo "  make serve        Levanta FastAPI en :8000"
	@echo "  make mlflow-ui    Levanta MLflow UI en :5000"
	@echo "  make docker-build Construye imagen Docker"
	@echo "  make docker-run   Corre la API en Docker"
	@echo "  make clean        Limpia artefactos generados"
	@echo ""
