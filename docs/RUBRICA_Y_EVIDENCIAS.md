# PC3: criterios y evidencias (20 puntos)

## 7 puntos: código desarrollado y evidencias del entrenamiento

- [ ] `src/train.py` realmente ejecuta entrenamiento; fuente y cambios reconocibles.
- [ ] `src/infer.py` funciona sin ejecutar entrenamiento nuevamente.
- [ ] Logs auténticos de cada época, hiperparámetros, timestamps y recursos usados.
- [ ] Checkpoints reales, verificados guardando y cargando el generador.
- [ ] Resultados A2B y B2A sobre fragmentos de prueba no vistos.
- [ ] Función de pérdida adversaria y consistencia cíclica explicadas en el código.
- [ ] Registrar qué se cambió respecto a `references/CycleGAN_profesor.ipynb`.

## 5 puntos: presentación

- [ ] Problema y requisito de dos géneros musicales, 30 segundos por ejemplo de evaluación.
- [ ] Diagrama de dos generadores, dos discriminadores y ciclo A↔B.
- [ ] Transformación de audio a espectrogramas y configuración empleada.
- [ ] Cambios de arquitectura, funciones de pérdida y parámetros.
- [ ] Evidencia comparativa, limitaciones y conclusión.

## 8 puntos: exposición (máximo 7 minutos)

- [ ] Ambos integrantes explican su parte y entienden la integración.
- [ ] Presentación ensayada para no superar 7:00.
- [ ] Demostración rápida: checkpoint real → espectrograma traducido.
- [ ] Respuestas listas: por qué CycleGAN, qué es la pérdida cíclica, cómo se evita fuga de datos.
- [ ] Si la reconstrucción de audio no funciona bien, explicar la limitación y mostrar espectrogramas.

## Experimentos sugeridos (no afirmar resultados antes de ejecutarlos)

- E1: CycleGAN base, `lambda_identity=0`.
- E2: probar `lambda_identity>0` sin variar los otros parámetros.
- Guardar por ejecución: configuración, pérdidas, comparaciones de espectrogramas y observaciones.
