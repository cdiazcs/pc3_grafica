"""Integrante 1: metricas y comparativas A -> B -> A y B -> A -> B.

Comparar modelos sobre canciones NO usadas para el entrenamiento.
Guardar tablas CSV y graficos. No inventar metricas ni evidencias.
"""
import argparse
from src.common.config import load_config


def evaluate(config: dict) -> None:
    """TODO (Integrante 1): implementar evaluacion reproducible."""
    raise NotImplementedError("Integrante 1: implementar evaluate()")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluacion CycleGAN")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
