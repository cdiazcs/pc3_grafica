

import copy
import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.common.config import load_config
from src.data.dataset import load_unpaired_dataset
from src.infer import infer
from src.models.cyclegan import build_cyclegan
from src.models.discriminator import build_discriminator
from src.models.generator import build_generator
from src.train import train


@pytest.fixture(autouse=True)
def reset_integration_keras():
    tf.keras.utils.set_random_seed(42)
    yield
    tf.keras.backend.clear_session()


@pytest.fixture(scope="module")
def synthetic_integration_config(tmp_path_factory):
    """Archivos del formato del preprocesador; son ruido, no espectrogramas musicales."""
    root = tmp_path_factory.mktemp("phase4_synthetic_technical_only")
    config = load_config("config/config.yaml")
    for name in config["paths"]:
        config["paths"][name] = str(root / name)
    config["training"].update(epochs=1, batch_size=1, checkpoint_every=2)
    rng = np.random.default_rng(2026)
    for split in ("train", "test"):
        for domain, limits in (("A", (-1, 0)), ("B", (0, 1))):
            directory = Path(config["paths"][f"{split}_{domain.lower()}"])
            directory.mkdir(parents=True)
            # Longitudes distintas verifican zip sin suponer correspondencia A/B.
            count = 3 if domain == "A" else 2
            data = rng.uniform(*limits, size=(count, 128, 128, 1)).astype(np.float32)
            np.save(directory / f"{split}{domain}_data.npy", data)
    return config


@pytest.fixture(scope="module")
def trained_synthetic_integration(synthetic_integration_config):
    """Ejecuta train sin sustituir ni dataset, ni modelos, ni train_step."""
    config = copy.deepcopy(synthetic_integration_config)
    train(config)
    return config


@pytest.mark.parametrize("split", ["train", "test"])
def test_dataset_contract_with_integrante_1(synthetic_integration_config, split):
    dataset = load_unpaired_dataset(synthetic_integration_config, split=split)
    assert int(tf.data.experimental.cardinality(dataset).numpy()) == 2
    count = 0
    for batch_A, batch_B in dataset:
        for batch in (batch_A, batch_B):
            assert batch.shape == (1, 128, 128, 1)
            assert batch.dtype == tf.float32
            assert np.all(np.isfinite(batch.numpy()))
            assert np.all(np.abs(batch.numpy()) <= 1)
        assert np.all(batch_A.numpy() <= 0)
        assert np.all(batch_B.numpy() >= 0)
        count += 1
    assert count == 2


def test_dataset_train_step_updates_all_models(synthetic_integration_config):
    gan = build_cyclegan(synthetic_integration_config)
    batch_A, batch_B = next(iter(load_unpaired_dataset(synthetic_integration_config)))
    names = ("G_A2B", "G_B2A", "D_A", "D_B")
    before = {name: [weight.numpy().copy() for weight in getattr(gan, name).trainable_variables]
              for name in names}
    assert gan.G_A2B(batch_A).shape == batch_B.shape
    assert gan.G_B2A(batch_B).shape == batch_A.shape
    assert gan.D_A(batch_A).shape == (1, 16, 16, 1)
    assert gan.D_B(batch_B).shape == (1, 16, 16, 1)
    metrics = gan.train_step(batch_A, batch_B)
    required = {"G_A2B_loss", "G_B2A_loss", "D_A_loss", "D_B_loss", "cycle_A", "cycle_B", "cycle_loss"}
    assert required <= metrics.keys()
    for value in metrics.values():
        assert value.shape == ()
        assert np.isfinite(float(value.numpy()))
    for name in names:
        current = getattr(gan, name).trainable_variables
        assert all(np.all(np.isfinite(weight.numpy())) for weight in current)
        assert any(np.any(old != new.numpy()) for old, new in zip(before[name], current))
        assert int(gan.optimizers[name].iterations.numpy()) == 1


def _checkpoint(config, name, source):
    directory = Path(config["paths"]["checkpoints"])
    if source == "epoch":
        directory /= "epoch_0001"
    return directory / f"{name}.weights.h5"


def test_checkpoint_recovery_all_models(trained_synthetic_integration):
    config = trained_synthetic_integration
    batch_A, batch_B = next(iter(load_unpaired_dataset(config, split="test")))
    for name in ("G_A2B", "G_B2A", "D_A", "D_B"):
        builder = build_generator if name.startswith("G_") else build_discriminator
        model = builder(tuple(config["model"]["input_shape"]), name=name)
        model.load_weights(_checkpoint(config, name, "epoch"))
        sample = batch_A if name in ("G_A2B", "D_A") else batch_B
        output = model(sample, training=False)
        assert np.all(np.isfinite(output.numpy()))
        if name.startswith("G_"):
            latest = builder(tuple(config["model"]["input_shape"]), name=name)
            latest.load_weights(_checkpoint(config, name, "latest"))
            tf.debugging.assert_near(output, latest(sample, training=False))
            for original, restored in zip(model.weights, latest.weights):
                np.testing.assert_array_equal(original.numpy(), restored.numpy())


@pytest.mark.parametrize("direction", ["A2B", "B2A"])
@pytest.mark.parametrize("checkpoint_source", ["latest", "epoch"])
def test_train_checkpoint_infer_roundtrip(trained_synthetic_integration, direction, checkpoint_source):
    config = trained_synthetic_integration
    domain = "A" if direction == "A2B" else "B"
    held_out = np.load(Path(config["paths"][f"test_{domain.lower()}"]) / f"test{domain}_data.npy")[0]
    input_path = Path(config["paths"]["generated"]).parent / f"synthetic_held_out_{checkpoint_source}_{domain}.npy"
    np.save(input_path, held_out)
    name = f"G_{direction}"
    checkpoint_path = _checkpoint(config, name, checkpoint_source)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    expected_model = build_generator(tuple(config["model"]["input_shape"]), name=name)
    expected_model.load_weights(checkpoint_path)
    expected = expected_model(held_out[None], training=False).numpy()[0]

    # Ejecutar la función original sin modificar ni sustituir su generador/predict.
    explicit_checkpoint = str(checkpoint_path) if checkpoint_source == "epoch" else None
    infer(config, str(input_path), direction, checkpoint_path=explicit_checkpoint)
    output_path = Path(config["paths"]["generated"]) / f"{input_path.stem}_{direction}.npy"
    actual = np.load(output_path)
    assert actual.shape == (128, 128, 1)
    assert actual.dtype == np.float32
    assert np.all(np.isfinite(actual))
    assert np.all(np.abs(actual) <= 1)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert output_path.with_suffix(".png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash


def test_training_log_and_plot(trained_synthetic_integration):
    from matplotlib.image import imread

    config = trained_synthetic_integration
    with (Path(config["paths"]["metrics"]) / "training_log.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert int(rows[0]["epoch"]) == 1
    assert int(rows[0]["steps"]) == 2
    assert int(rows[0]["samples_per_domain"]) == 2
    for key in ("G_A2B_loss", "G_B2A_loss", "D_A_loss", "D_B_loss", "cycle_A", "cycle_B", "cycle_loss"):
        assert np.isfinite(float(rows[0][key]))
        assert float(rows[0][key]) >= 0
    np.testing.assert_allclose(float(rows[0]["cycle_loss"]),
                               float(rows[0]["cycle_A"]) + float(rows[0]["cycle_B"]))
    image = imread(Path(config["paths"]["plots"]) / "training_losses.png")
    assert image.ndim == 3
    assert image.shape[0] > 100 and image.shape[1] > 100
    assert np.all(np.isfinite(image))
    assert np.ptp(image) > 0


def test_infer_cli_existing_arguments(trained_synthetic_integration, monkeypatch):
    import src.infer as inference_module

    config = trained_synthetic_integration
    patch = np.load(Path(config["paths"]["test_b"]) / "testB_data.npy")[1]
    input_path = Path(config["paths"]["generated"]).parent / "synthetic_cli_B.npy"
    np.save(input_path, patch)
    checkpoint = _checkpoint(config, "G_B2A", "latest")
    # Solo sustituir lectura de rutas para no crear ni editar archivos YAML.
    # main() sigue analizando sus argumentos y llama a infer() original.
    monkeypatch.setattr(inference_module, "load_config", lambda path: copy.deepcopy(config))
    monkeypatch.setattr("sys.argv", ["src.infer", "--config", "config/config.yaml",
                                    "--input", str(input_path), "--direction", "B2A",
                                    "--checkpoint", str(checkpoint)])
    inference_module.main()
    output = np.load(Path(config["paths"]["generated"]) / "synthetic_cli_B_B2A.npy")
    model = build_generator(name="G_B2A")
    model.load_weights(checkpoint)
    np.testing.assert_allclose(output, model(patch[None], training=False).numpy()[0], rtol=1e-5, atol=1e-6)
