"""Entrenamiento CycleGAN con pérdidas observadas y checkpoints verificables."""

import argparse
import copy
import csv
import itertools
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.common.config import load_config
from src.data.dataset import load_unpaired_dataset
from src.models.cyclegan import build_cyclegan


_MODEL_NAMES = ("G_A2B", "G_B2A", "D_A", "D_B")
_REQUIRED_METRICS = ("G_A2B_loss", "G_B2A_loss", "D_A_loss", "D_B_loss", "cycle_loss")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} debe ser un entero positivo")
    return value


def _validate_batch(batch, batch_size, domain):
    tensor = tf.convert_to_tensor(batch)
    if tensor.dtype != tf.float32:
        raise ValueError(f"Batch {domain}: se requiere float32, recibido {tensor.dtype.name}")
    expected = (batch_size, 128, 128, 1)
    if tuple(tf.shape(tensor).numpy()) != expected:
        raise ValueError(f"Batch {domain}: shape esperado {expected}, recibido {tensor.shape}")
    tf.debugging.assert_all_finite(tensor, f"Batch {domain}: valores no finitos")
    tf.debugging.assert_greater_equal(tensor, -1.0, message=f"Batch {domain}: rango inferior a -1")
    tf.debugging.assert_less_equal(tensor, 1.0, message=f"Batch {domain}: rango superior a 1")
    return tensor


def _validated_pair(pair, batch_size):
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError("El dataset debe devolver pares (batch_A, batch_B)")
    return (_validate_batch(pair[0], batch_size, "A"),
            _validate_batch(pair[1], batch_size, "B"))


def _save_checkpoint(gan, directory, epoch, batches):
    """Guarda los cuatro modelos y verifica pesos y predicciones tras recarga."""
    epoch_dir = directory / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    for name in _MODEL_NAMES:
        model = getattr(gan, name)
        path = epoch_dir / f"{name}.weights.h5"
        model.save_weights(path)
        restored = tf.keras.models.clone_model(model)
        restored.load_weights(path)
        for original_weight, restored_weight in zip(model.weights, restored.weights):
            np.testing.assert_array_equal(original_weight.numpy(), restored_weight.numpy())
        sample = batches[0] if name in ("G_A2B", "D_A") else batches[1]
        tf.debugging.assert_near(model(sample, training=False), restored(sample, training=False),
                                 atol=1e-6, rtol=1e-5,
                                 message=f"Recarga inconsistente de {name}")
        # Publicar para inferencia solo después de verificar el archivo guardado.
        if name.startswith("G_"):
            temporary = directory / f".{name}.weights.h5.tmp"
            shutil.copyfile(path, temporary)
            temporary.replace(directory / f"{name}.weights.h5")
        del restored


def _plot_losses(rows, path):
    # Backend sin ventana gráfica, válido también en servidores/CPU.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    groups = (("Generadores", ("G_A2B_loss", "G_B2A_loss")),
              ("Discriminadores", ("D_A_loss", "D_B_loss")),
              ("Ciclo e identidad (L1 sin ponderar)", ("cycle_loss", "identity_loss")))
    epochs = [row["epoch"] for row in rows]
    for axis, (title, keys) in zip(axes, groups):
        for key in keys:
            if key in rows[0]:
                axis.plot(epochs, [row[key] for row in rows], marker="o", label=key)
        axis.set(title=title, xlabel="Época", ylabel="Pérdida media por batch")
        axis.legend()
        axis.grid(alpha=0.3)
    figure.savefig(path, dpi=150)
    figure.clear()


def train(config: dict) -> None:
    """Entrena todas las épocas; CSV y curvas contienen medias reales por época.

    Las rutas relativas se resuelven desde la raíz del proyecto. Cada ejecución
    comienza desde cero y escribe su propio CSV; no reanuda optimizadores.
    Un checkpoint contiene los pesos de los cuatro modelos por época guardada.
    """
    config = copy.deepcopy(config)
    root = Path(__file__).resolve().parents[1]
    for name, path in config["paths"].items():
        config["paths"][name] = str(root / Path(path))
    training = config["training"]
    epochs = _positive_integer(training["epochs"], "epochs")
    batch_size = _positive_integer(training["batch_size"], "batch_size")
    checkpoint_every = _positive_integer(training["checkpoint_every"], "checkpoint_every")
    if tuple(config["model"]["input_shape"]) != (128, 128, 1):
        raise ValueError("El contrato requiere input_shape=(128, 128, 1)")
    tf.keras.utils.set_random_seed(config["project"]["seed"])
    try:
        dataset = load_unpaired_dataset(config, split="train")
    except FileNotFoundError as exc:
        raise FileNotFoundError("Faltan datos de entrenamiento .npy; no se generarán datos ni métricas simulados") from exc
    except tf.errors.InvalidArgumentError as exc:
        # El cargador existente usa shuffle(0) cuando uno de los dominios está vacío.
        if "buffer_size" in str(exc) and "greater than zero" in str(exc):
            raise ValueError("Dataset de entrenamiento vacío en al menos un dominio") from exc
        raise
    cardinality = int(tf.data.experimental.cardinality(dataset).numpy())
    if cardinality == int(tf.data.INFINITE_CARDINALITY):
        raise ValueError("El dataset debe ser finito para delimitar las épocas")
    iterator = iter(dataset)
    try:
        first_pair = _validated_pair(next(iterator), batch_size)
    except StopIteration as exc:
        raise ValueError("Dataset vacío: no hay batches completos en ambos dominios; revise batch_size y los datos") from exc

    gan = build_cyclegan(config)
    metrics_dir = Path(config["paths"]["metrics"])
    plots_dir = Path(config["paths"]["plots"])
    checkpoint_dir = Path(config["paths"]["checkpoints"])
    for directory in (metrics_dir, plots_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rows = []
    writer = None
    with (metrics_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            # Conservar el primer batch inspeccionado evita perderlo o repetirlo.
            pairs = itertools.chain((first_pair,), iterator) if epoch == 1 else iter(dataset)
            totals = {}
            steps = 0
            samples = 0
            for pair in pairs:
                batch_A, batch_B = _validated_pair(pair, batch_size)
                metrics = gan.train_step(batch_A, batch_B)
                if not all(key in metrics for key in _REQUIRED_METRICS):
                    raise ValueError("train_step no devolvió todas las pérdidas requeridas")
                if steps and metrics.keys() != totals.keys():
                    raise ValueError("Las métricas cambiaron durante la época")
                for key, value in metrics.items():
                    scalar = np.asarray(value)
                    if scalar.shape != () or not np.isfinite(scalar):
                        raise ValueError(f"Métrica {key} no escalar o no finita")
                    totals[key] = totals.get(key, 0.0) + float(scalar)
                steps += 1
                samples += batch_size
            if not steps:
                raise ValueError(f"Dataset vacío en la época {epoch}")
            row = {"epoch": epoch, "steps": steps, "samples_per_domain": samples,
                   **{key: value / steps for key, value in totals.items()}}
            if writer is None:
                writer = csv.DictWriter(log, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            log.flush()
            rows.append(row)
            if epoch % checkpoint_every == 0 or epoch == epochs:
                _save_checkpoint(gan, checkpoint_dir, epoch, first_pair)
            _plot_losses(rows, plots_dir / "training_losses.png")
            print(f"Época {epoch}/{epochs}, batches={steps}: " + ", ".join(
                f"{key}={row[key]:.6f}" for key in _REQUIRED_METRICS), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento CycleGAN")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
