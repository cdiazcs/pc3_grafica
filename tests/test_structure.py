"""Archivos minimos de la rama principal; no exige entrenamiento."""
from pathlib import Path


REQUIRED = [
    "config/config.yaml",
    "src/common/config.py",
    "src/data/preprocess.py",
    "src/data/dataset.py",
    "src/models/generator.py",
    "src/models/discriminator.py",
    "src/models/cyclegan.py",
    "src/train.py",
    "src/infer.py",
    "src/evaluate.py",
    "docs/CONTRATOS.md",
]


def test_required_files_exist():
    for filename in REQUIRED:
        assert Path(filename).is_file(), f"Falta {filename}"
