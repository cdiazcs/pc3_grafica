# MusicStyle-CycleGAN — PC3 Computación Gráfica

**Universidad Nacional de Ingeniería (UNI) · 2026-II**  
**Práctica Calificada 3 · Opción 2: CycleGAN**  
**Repositorio:** https://github.com/cdiazcs/pc3_grafica

##Integrantes :
-Enoc Rosales 
-Cristian Diaz

## Objetivo

Implementar una **CycleGAN** que traduzca características de espectrogramas musicales entre **cumbia (dominio A)** y **rock (dominio B)**, en ambos sentidos, sin requerir canciones emparejadas.

## Trabajo desarrollado

- **Datos (integrante 1(Enoc)):** carga de audios, conversión a espectrogramas Mel, normalización a `[-1, 1]`, extracción de parches `(128, 128, 1)` y separación de entrenamiento/prueba por archivo musical.
- **Modelos (integrante 2(Cristian)):** dos generadores U-Net (`G_A2B`, `G_B2A`) y dos discriminadores PatchGAN (`D_A`, `D_B`).
- **Aprendizaje:** pérdidas adversarias, consistencia cíclica L1, pérdida de identidad opcional y cuatro optimizadores Adam.
- **Entrenamiento:** script por épocas con métricas CSV, curvas de pérdidas y guardado/recarga de checkpoints.
- **Resultados:** scripts independientes de inferencia (`src/infer.py`) y evaluación (`src/evaluate.py`) para traducciones y reconstrucciones cíclicas.
- **Pruebas:** tests de estructura, configuración, datos, modelos e integración.

## Estructura principal

```text
config/config.yaml           # Parámetros compartidos
src/data/                    # Preprocesamiento y dataset
src/models/                  # Generadores, discriminadores y CycleGAN
src/train.py                 # Entrenamiento
src/infer.py                 # Traducción de espectrogramas
src/evaluate.py              # Evaluación
src/visualization.py         # Visualizaciones
tests/                       # Pruebas
data/raw/                    # Audios por género
data/processed/              # Espectrogramas .npy
checkpoints/                 # Pesos entrenados
results/                     # Métricas, gráficas y traducciones
presentation/                # Presentación de la PC3
```

## Entorno y ejecución

Entorno utilizado: **Python 3.12.16** y **TensorFlow 2.16.2** en Ubuntu.

```bash
# Desde la raíz del repositorio
source .venv/bin/activate
python -m pip install -r requirements.txt

# Preparar audios y espectrogramas
python -m src.data.preprocess --config config/config.yaml

# Ejecutar pruebas
python -m pytest tests/ -q

# Entrenar (requiere archivos .npy preparados)
python -m src.train --config config/config.yaml

# Inferir un parche individual (128,128,1)
python -m src.infer --config config/config.yaml \
  --direction A2B --input ruta/parche.npy \
  --checkpoint ruta/G_A2B.weights.h5

# Evaluar con los checkpoints indicados en la configuración
python -m src.evaluate --config config/config.yaml
```

## Evidencia de avance (13/11/2026)

- La auditoría reportó **17 pruebas de solo lectura aprobadas**.
- Se ejecutó una **prueba real de entrenamiento de 1 época y 19 batches** con espectrogramas musicales, usando batch size 1.
- Las pérdidas de la época se registraron en `results/metrics/prueba_cpu/training_log.csv`.
- Se guardaron pesos de ambos generadores en `checkpoints/prueba_cpu/`, además de checkpoints por época.

## Equipo y Git

| Responsable | Rama de trabajo | Módulos |
|---|---|---|
| Integrante 1 | `feature/data-inference` | Datos, inferencia, evaluación y visualización |
| Integrante 2 | `feature/model-training` | Generadores, discriminadores y entrenamiento |

Los cambios se integran directamente a **`main` mediante Pull Requests**. No se utiliza `develop`.


**Referencia académica:** notebook de CycleGAN proporcionado en clase, conservado en `references/`.
