# De `CycleGAN.ipynb` de clase al proyecto

Material de referencia original (sin modificar): `references/CycleGAN_profesor.ipynb`.
El notebook muestra transferencia de apariencia visual entre dominios de rostros con/sin mascarilla. **Este proyecto debe cambiar el dominio y adecuar la red a espectrogramas de audio**.

## Correspondencia de funciones del notebook

| Notebook de clase | Módulo nuevo | Cambio necesario |
|---|---|---|
| `preprocess()` | `src/data/preprocess.py` | Audio -> espectrograma Mel -> patch float32 normalizado |
| `downsample()` y `upsample()` | `src/models/generator.py` | Capas y normalización compatibles con entorno instalado |
| `generator()` | `src/models/generator.py` | De 3 canales RGB a 1 canal de espectrograma |
| `discriminator()` | `src/models/discriminator.py` | PatchGAN con entrada de 1 canal |
| `class CycleGAN` | `src/models/cyclegan.py` | Dos generadores y dos discriminadores, pérdidas y paso de entrenamiento |
| `discriminator_loss_fn`, `generator_loss_fn`, `cycle_loss_fn` | `src/models/cyclegan.py` | Mantener adversaria y ciclo; añadir identity si se experimenta |
| Callback de imagen de mascarillas | `src/visualization.py` | Graficar espectrogramas y curvas |

**Atención a compatibilidad:** el notebook importa `tensorflow_addons` y concatena tensores con `tf.concat`. En versiones recientes de Keras puede requerirse reemplazar InstanceNormalization y usar `keras.layers.Concatenate` dentro del grafo. Probar compatibilidad con la versión realmente instalada.

Separar canciones en entrenamiento y evaluación antes de crear los parches: datos solapados del mismo audio no deben entrar en ambos conjuntos.
No presentar espectrogramas distintos entre géneros como pares con objetivo exacto.
