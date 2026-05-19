FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY serve/ serve/
COPY config.yaml .

# Artifacts are mounted at runtime via Docker volume
# Expose API port
EXPOSE 8000

ENV MODEL_PATH=artifacts/model.pkl

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
