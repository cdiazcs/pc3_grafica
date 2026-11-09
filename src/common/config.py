"""Carga y validacion de la configuracion compartida. No modificar unilateralmente."""
from pathlib import Path
from typing import Any
import yaml

REQUIRED_TOP_LEVEL = {"project", "paths", "audio", "model", "training"}


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Leer YAML y validar contratos basicos (independiente de TensorFlow)."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe configuracion: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("La configuracion debe ser un mapa YAML")
    missing = REQUIRED_TOP_LEVEL - config.keys()
    if missing:
        raise ValueError(f"Secciones faltantes: {sorted(missing)}")
    shape = config["model"].get("input_shape")
    if shape != [128, 128, 1]:
        raise ValueError("Contrato actual: input_shape debe ser [128, 128, 1]")
    if config["audio"].get("n_mels") != 128 or config["audio"].get("patch_frames") != 128:
        raise ValueError("Contrato actual: n_mels=128 y patch_frames=128")
    if config["audio"].get("segment_seconds") != 30:
        raise ValueError("La evaluacion debe usar fragmentos de 30 segundos")
    if config["training"].get("batch_size", 0) < 1:
        raise ValueError("batch_size debe ser positivo")
    return config
