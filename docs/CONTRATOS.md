# Contratos de integracion de las dos ramas

## Formato compartido (no modificar sin PR consensuado)

- Framework objetivo: **TensorFlow + Keras**, Python 3.11.
- Cada espectrograma de entrenamiento: `.npy` de `numpy.float32`, forma `(128,128,1)`, rango normalizado **[-1, 1]**.
- Cada batch: `(batch_size,128,128,1)`; modelo y dataset NO deben incorporar conversión adicional de rango.
- Segmentos de **30 segundos**, convertidos en ventanas de entrenamiento; los parches de 128 frames son más cortos que 30 s.
- Split por **canción original** antes de generar parches; guardar fuente e identificador en metadatos.
- A: cumbia; B: rock (configurables en `config/config.yaml`). Elegir muestras musicales preferentemente sin voz.
- Las rutas se leen siempre de `config/config.yaml`; relativas a la raíz del repo.

## Integrante 1 → Integrante 2

`src/data/dataset.py`

```python
load_unpaired_dataset(config: dict, split: str = "train") -> tf.data.Dataset
```

Cada elemento del dataset es una tupla `(batch_A, batch_B)`; ambos tensores tienen `dtype=float32` y `shape=(None,128,128,1)`. Las muestras son **no emparejadas**. Durante entrenamiento, barajar cada dominio independientemente.

`src/data/preprocess.py`

```python
prepare_data(config: dict) -> None
```

Escribe `.npy` en trainA/trainB/testA/testB y conserva metadatos de normalización por audio.

## Integrante 2 → Integrante 1

`src/models/generator.py`:

```python
build_generator(input_shape=(128,128,1), name="G_A2B") -> tf.keras.Model
```

Crea un generador con salida del mismo tamaño y rango. Nombres esperados: `G_A2B`, `G_B2A`.

`src/models/discriminator.py`:

```python
build_discriminator(input_shape=(128,128,1), name="D_A") -> tf.keras.Model
```

Crea discriminador con salida de **logits** (convención de pérdida adversaria `from_logits=True`). Nombres: `D_A`, `D_B`.

`src/models/cyclegan.py`:

```python
build_cyclegan(config: dict)
```

Devuelve objeto integrador de entrenamiento; sus componentes deben permitir exportar cada generador de forma independiente.

## Acuerdo de checkpoints

Se guardan por dirección los pesos de ambos generadores (nombres definidos):

- `checkpoints/G_A2B.weights.h5`
- `checkpoints/G_B2A.weights.h5`

**Importante:** `.weights.h5` requiere instanciar exactamente la arquitectura al cargarse. Guardar junto a los pesos `config.yaml` efectivo o archivo de metadatos. El contenido del ZIP NO incluye pesos entrenados.

## Comandos CLI acordados

```bash
python -m src.data.preprocess --config config/config.yaml
python -m src.train --config config/config.yaml
python -m src.infer --config config/config.yaml --input RUTA.npy --direction A2B
python -m src.evaluate --config config/config.yaml
```

Todo módulo posee entrada `main()` y API de Python documentada. Un script puede fallar con `NotImplementedError` hasta ser desarrollado; no reemplazarlo con resultados simulados.

## Prueba de integración obligatoria

1. Dataset devuelve 1 batch `(A,B)` con shape y rango esperados.
2. `G_A2B(A)` devuelve `(B,128,128,1)` y `G_B2A(B)` igual.
3. `D_A` y `D_B` aceptan sus espectrogramas respectivos.
4. Entrenar **un paso real** sin `NaN` ni errores de shape.
5. Guardar y recargar los pesos; ejecutar `infer.py` sin reentrenar.
6. Ejecutar traducción, vuelta cíclica y evaluación con ejemplos nunca vistos.
