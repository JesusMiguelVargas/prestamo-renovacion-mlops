"""Pipeline orchestrator.

Runs the full end-to-end MLOps pipeline or selected stages.

Usage:
  python main.py                          # run all stages
  python main.py --steps download preprocess segregate
  python main.py --steps check_data
  python main.py --steps train evaluate drift
"""
import argparse
import logging
import subprocess
import sys
import time
from typing import List

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | PIPELINE | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

ALL_STAGES = ["download", "preprocess", "segregate", "check_data", "train", "evaluate", "drift"]

STAGE_CMD = {
    "download":   [sys.executable, "_data/run.py"],
    "preprocess": [sys.executable, "preprocess/run.py"],
    "segregate":  [sys.executable, "segregate/run.py"],
    "check_data": [sys.executable, "-m", "pytest", "check_data/", "-v", "--tb=short"],
    "train":      [sys.executable, "train/run.py"],
    "evaluate":   [sys.executable, "evaluate/run.py"],
    "drift":      [sys.executable, "drift/run.py"],
}

STAGE_DESC = {
    "download":   "Carga y valida el dataset raw",
    "preprocess": "Limpieza, imputación y encoding",
    "segregate":  "Split estratificado train/test",
    "check_data": "Tests de validación con pytest",
    "train":      "GridSearchCV + MLflow tracking",
    "evaluate":   "Métricas + quality gate",
    "drift":      "EvidentlyAI drift detection",
}


def run_stage(name: str) -> float:
    cmd = STAGE_CMD[name]
    log.info("=" * 60)
    log.info("INICIANDO: %s — %s", name.upper(), STAGE_DESC[name])
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log.error("FALLO en etapa '%s' (código %d)", name, result.returncode)
        sys.exit(result.returncode)
    log.info("<<< Completado: %s (%.2f s)", name, elapsed)
    return elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline MLOps — Renovación de Préstamo")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STAGES,
        default=ALL_STAGES,
        help="Etapas a ejecutar (default: todas)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Ruta al archivo de configuración",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = args.steps

    log.info("=" * 60)
    log.info("PIPELINE MLOps — Renovación de Préstamo")
    log.info("Etapas: %s", " → ".join(steps))
    log.info("=" * 60)

    total_start = time.time()
    for stage in steps:
        run_stage(stage)

    total = time.time() - total_start
    log.info("=" * 60)
    log.info("PIPELINE COMPLETADO EN %.2f segundos", total)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
