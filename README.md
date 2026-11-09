# PC3 - MusicStyle CycleGAN | Base comun (`main`)

Práctica Calificada 3 - Computación Gráfica, UNI, 2026-II.
Equipo: **máximo 2 integrantes**. Exposición: **viernes 13/11/2026**, 7 minutos.

## Objetivo

Adaptar CycleGAN para traducir el estilo de espectrogramas musicales entre dos géneros A y B. El material parte del notebook de clase, ubicado en `references/`, **sin afirmar que haya entrenamiento realizado**.

> Este ZIP es exclusivamente el **punto de partida de `main`**: estructura, configuración, contratos y verificaciones iniciales. Los módulos de datos y entrenamiento contienen `NotImplementedError` de forma deliberada. Las funcionalidades se desarrollan en dos ramas y se fusionan directamente a `main`. No se usa `develop`.

## Organización

| Responsable | Rama | Archivos a implementar |
|---|---|---|
| Integrante 1 | `feature/data-inference` | `src/data/preprocess.py`, `src/data/dataset.py`, `src/infer.py`, `src/evaluate.py`, `src/visualization.py`, `tests/test_data.py` |
| Integrante 2 | `feature/model-training` | `src/models/generator.py`, `src/models/discriminator.py`, `src/models/cyclegan.py`, `src/train.py`, `tests/test_models.py` |
| Ambos | `main` (via PR) | Integración, `tests/test_integration.py`, `presentation/`, evidencias reales |

**No cambiar sin acordarlo:** `config/config.yaml`, `src/common/config.py` y `docs/CONTRATOS.md`. Cualquier cambio de interfaz debe integrarse primero mediante PR específico.

## Comenzar desde el ZIP

1. Extraer `PC3-CycleGAN-BASE-MAIN.zip` y abrir la carpeta raíz.
2. Crear repositorio Git local sobre la base:

```bash
git init -b main
git add .
git commit -m "chore: initialize PC3 CycleGAN project base"
git remote add origin URL_DEL_REPOSITORIO_GITHUB
git push -u origin main
```

3. Cada integrante clona ese repositorio y crea su rama desde `main`:

```bash
git clone URL_DEL_REPOSITORIO_GITHUB
cd PC3-CycleGAN-BASE-MAIN
git switch -c feature/data-inference        # integrante 1
# o: git switch -c feature/model-training  # integrante 2
git push -u origin NOMBRE_DE_MI_RAMA
```

4. Crear entorno Python 3.11 y ejecutar comprobaciones de **base**:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/check_structure.py
python -m pytest tests/test_config.py tests/test_structure.py -q
```

5. Para ejecutar el proyecto completo cuando esté implementado:

```bash
pip install -r requirements.txt
python -m src.data.preprocess --config config/config.yaml
python -m src.train --config config/config.yaml
python -m src.infer --config config/config.yaml --direction A2B --input ruta/espectrograma.npy
python -m src.evaluate --config config/config.yaml
```

Estos últimos comandos **no funcionan hasta que los integrantes implementen los TODO**; esto es intencional.

## Contratos y documentación

- `docs/CONTRATOS.md`: formatos, firmas y dependencias entre integrantes.
- `docs/GIT_WORKFLOW.md`: trabajo por ramas, sincronización, PR y merge sin `develop`.
- `docs/RUBRICA_Y_EVIDENCIAS.md`: lista de evidencias para 7 + 5 + 8 puntos.
- `docs/ADAPTACION_NOTEBOOK.md`: guía para trasladar el notebook del profesor a espectrogramas.

## Dataset y salidas

`data/raw/genre_A/` y `genre_B/`: audios de referencia autorizados; la evaluación debe contener fragmentos **de 30 segundos** para ambos géneros.
`data/processed/trainA`, `trainB`, `testA`, `testB`: matrices `.npy` preprocesadas. Separar **por canción** antes de generar ventanas para impedir fuga de datos entre train y test.

`checkpoints/` y `results/` se generan durante entrenamiento e inferencia, pero no vienen con pesos ni resultados ficticios. Mantener checkpoints importantes en almacenamiento compartido y documentar el procedimiento de recuperación.

## Integración

Todos los cambios se integran **directamente a `main` mediante Pull Request**, tras revisión y pruebas. No existe rama `develop`. Ver `docs/GIT_WORKFLOW.md`.
