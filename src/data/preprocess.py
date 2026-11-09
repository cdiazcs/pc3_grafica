"""Integrante 1: audios (30 s) -> espectrogramas Mel -> patches .npy.

Interfaz acordada: leer los archivos de paths.raw_a y paths.raw_b,
separar canciones train/test ANTES de crear patches, guardar matrices
float32 en [-1, 1] con shape (128, 128, 1).
Guardar tambien metadatos de normalizacion para pruebas y posible audio.
"""
import argparse
from src.common.config import load_config


def prepare_data(config: dict) -> None:
    """TODO (Integrante 1): implementar audio, espectrogramas y particion por cancion."""
    raise NotImplementedError("Integrante 1: implementar prepare_data() en src/data/preprocess.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesar archivos musicales para CycleGAN")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    prepare_data(load_config(args.config))


if __name__ == "__main__":
    main()
