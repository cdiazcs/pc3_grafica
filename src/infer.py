"""Integrante 1: inferencia sin volver a entrenar (pesos guardados)."""
import argparse
from src.common.config import load_config


def infer(config: dict, input_path: str, direction: str, checkpoint_path: str | None = None) -> None:
    """TODO (Integrante 1): cargar solo generador necesario y guardar .npy/PNG."""
    raise NotImplementedError("Integrante 1: implementar infer() y cargar checkpoints reales")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia de espectrogramas CycleGAN")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", required=True, help="Ruta a patch .npy")
    parser.add_argument("--direction", choices=("A2B", "B2A"), default="A2B")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    infer(load_config(args.config), args.input, args.direction, args.checkpoint)


if __name__ == "__main__":
    main()
