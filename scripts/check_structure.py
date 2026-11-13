
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    "README.md", "config/config.yaml", "src/common/config.py",
    "src/data/preprocess.py", "src/data/dataset.py",
    "src/models/generator.py", "src/models/discriminator.py", "src/models/cyclegan.py",
    "src/train.py", "src/infer.py", "src/evaluate.py",
    "tests/test_config.py", "docs/CONTRATOS.md",
]


def main():
    missing = [item for item in EXPECTED if not (ROOT / item).exists()]
    if missing:
        print("Faltan archivos:\n" + "\n".join(missing))
        return 1
    print(f"OK: estructura comun verificada ({len(EXPECTED)} archivos esenciales).")
    print("AVISO: entrenamiento e inferencia todavia son TODO; esto es solo la base.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
