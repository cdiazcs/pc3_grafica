# Flujo Git — solo `main` + ramas de funcionalidad

## Arranque

- Responsable inicial: crea repo con este ZIP y hace primer commit en `main`.
- Integrante 1: `feature/data-inference` desde `main`.
- Integrante 2: `feature/model-training` desde `main`.
- **Prohibido usar `develop`** o trabajar directamente en `main` después del arranque.
- Para evitar conflictos, cada integrante modifica sus archivos. `config/` y `src/common/` requieren coordinación.

## Comandos — crear rama (una sola vez)

```bash
git switch main
git pull origin main
git switch -c feature/data-inference    # o feature/model-training
git push -u origin feature/data-inference
```

## Registrar cambios pequeños

```bash
git status
git add src/data tests/test_data.py    # ejemplo del Integrante 1
git commit -m "feat(data): prepare mel spectrogram dataset"
git push
```

Crear Pull Request: `feature/...` → `main`. El compañero revisa firmas y pruebas antes del merge.

## Actualizar rama con la última versión de main

```bash
git switch main
git pull origin main
git switch feature/data-inference   # cambiar por rama propia
git merge main
git push
```

Si existe conflicto: resolverlo, ejecutar pruebas y confirmar el merge. No usar `git push --force` en `main`.

## Primeros PR sugeridos

1. PR 1 de datos: `prepare_data`, `.npy`, pruebas del dataset.
2. PR 1 del modelo: generador/discriminador, test de formas.
3. PR 2 del modelo: entrenamiento de un paso y checkpoint.
4. PR 2 de datos: inferencia y evaluación, usa contrato de checkpoint.
5. PR final de integración: pipeline completo, evidencias y presentación.

> Después de que los dos PR iniciales entren a `main`, los dos integrantes sincronizan sus ramas antes de continuar.
