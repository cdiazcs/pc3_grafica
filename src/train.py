"""Integrante 2: entrenamiento independiente con checkpoints y logs."""
import argparse
from src.common.config import load_config


def train(config: dict) -> None:
    """TODO (Integrante 2): datos, modelos, optimizacion y evidencias reales."""
    raise NotImplementedError("Integrante 2: implementar train(); no hay entrenamiento en el ZIP base")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento CycleGAN")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
