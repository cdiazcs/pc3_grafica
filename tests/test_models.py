"""Pruebas de arquitecturas, pérdidas y actualizaciones reales de CycleGAN."""

import numpy as np
import pytest
import tensorflow as tf

from src.common.config import load_config
from src.models.cyclegan import build_cyclegan
from src.models.generator import build_generator
from src.models.discriminator import build_discriminator


@pytest.fixture(autouse=True)
def reset_keras():
    tf.keras.utils.set_random_seed(42)
    yield
    tf.keras.backend.clear_session()


@pytest.mark.parametrize("name", ["G_A2B", "G_B2A"])
def test_generator_contract(name):
    model = build_generator(name=name)
    assert model.name == name
    assert model.output_shape == (None, 128, 128, 1)
    assert sum(isinstance(layer, tf.keras.layers.Concatenate) for layer in model.layers) == 4
    assert model.layers[-1].activation == tf.keras.activations.tanh
    for batch_size in (1, 2):
        output = model(tf.random.uniform((batch_size, 128, 128, 1), -1, 1))
        assert output.shape == (batch_size, 128, 128, 1)
        assert output.dtype == tf.float32
        tf.debugging.assert_all_finite(output, "Salida del generador no finita")
        assert bool(tf.reduce_all(tf.abs(output) <= 1))


@pytest.mark.parametrize("name", ["D_A", "D_B"])
def test_discriminator_contract(name):
    model = build_discriminator(name=name)
    assert model.name == name
    assert model.output_shape == (None, 16, 16, 1)
    assert model.layers[-1].activation == tf.keras.activations.linear
    for batch_size in (1, 2):
        output = model(tf.random.uniform((batch_size, 128, 128, 1), -1, 1))
        assert output.shape == (batch_size, 16, 16, 1)
        tf.debugging.assert_all_finite(output, "Logits no finitos")


def test_adversarial_gradients():
    generator = build_generator()
    discriminator = build_discriminator()
    inputs = tf.random.uniform((1, 128, 128, 1), -1, 1)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        generated = generator(inputs, training=True)
        logits = discriminator(generated, training=True)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(logits), logits)
    variables = generator.trainable_variables + discriminator.trainable_variables
    gradients = tape.gradient(loss, [inputs] + variables)
    tf.debugging.assert_all_finite(loss, "Pérdida no finita")
    for gradient in gradients:
        assert gradient is not None
        tf.debugging.assert_all_finite(gradient, "Gradiente no finito")
        assert bool(tf.reduce_any(gradient != 0))


@pytest.mark.parametrize("builder", [build_generator, build_discriminator])
def test_keras_serialization(builder):
    model = builder()
    restored = tf.keras.models.model_from_json(model.to_json())
    restored.set_weights(model.get_weights())
    inputs = tf.random.uniform((1, 128, 128, 1), -1, 1)
    tf.debugging.assert_near(model(inputs), restored(inputs))


@pytest.mark.parametrize("builder", [build_generator, build_discriminator])
@pytest.mark.parametrize("shape", [(128, 128, 3), (16, 16, 1)])
def test_invalid_shape(builder, shape):
    with pytest.raises(ValueError):
        builder(input_shape=shape)


@pytest.mark.parametrize("lambda_identity", [0.0, 0.7])
def test_cyclegan_real_step(lambda_identity, monkeypatch):
    config = load_config("config/config.yaml")
    config["training"]["lambda_identity"] = lambda_identity
    gan = build_cyclegan(config)
    names = ("G_A2B", "G_B2A", "D_A", "D_B")
    before = {name: [v.numpy().copy() for v in getattr(gan, name).trainable_variables]
              for name in names}
    recorded = {}

    def capture_gradients(name, apply):
        def wrapped(pairs, *args, **kwargs):
            pairs = list(pairs)
            recorded[name] = [gradient.numpy().copy() for gradient, _ in pairs]
            return apply(pairs, *args, **kwargs)
        return wrapped

    for name in names:
        assert getattr(gan, name).name == name
        optimizer = gan.optimizers[name]
        assert isinstance(optimizer, tf.keras.optimizers.Adam)
        assert float(optimizer.learning_rate.numpy()) == pytest.approx(config["training"]["learning_rate"])
        assert optimizer.beta_1 == 0.5
        assert optimizer.beta_2 == 0.999
        monkeypatch.setattr(optimizer, "apply_gradients",
                            capture_gradients(name, optimizer.apply_gradients))

    batch_A = tf.random.uniform((1, 128, 128, 1), -1, 1)
    batch_B = tf.random.uniform((1, 128, 128, 1), -1, 1)
    expected_cycle_A = tf.reduce_mean(tf.abs(batch_A - gan.G_B2A(gan.G_A2B(batch_A))))
    expected_cycle_B = tf.reduce_mean(tf.abs(batch_B - gan.G_A2B(gan.G_B2A(batch_B))))
    if lambda_identity:
        expected_identity_A = tf.reduce_mean(tf.abs(batch_A - gan.G_B2A(batch_A)))
        expected_identity_B = tf.reduce_mean(tf.abs(batch_B - gan.G_A2B(batch_B)))

    metrics = gan.train_step(batch_A, batch_B)
    for value in metrics.values():
        assert value.shape == ()
        assert np.isfinite(value.numpy())
        assert float(value) >= 0
    np.testing.assert_allclose(metrics["cycle_A"], expected_cycle_A, rtol=1e-5)
    np.testing.assert_allclose(metrics["cycle_B"], expected_cycle_B, rtol=1e-5)
    np.testing.assert_allclose(metrics["cycle_loss"], expected_cycle_A + expected_cycle_B)
    if lambda_identity:
        np.testing.assert_allclose(metrics["identity_A"], expected_identity_A, rtol=1e-5)
        np.testing.assert_allclose(metrics["identity_B"], expected_identity_B, rtol=1e-5)
        np.testing.assert_allclose(metrics["identity_loss"], expected_identity_A + expected_identity_B)
    else:
        assert not any(key.startswith("identity") for key in metrics)

    for direction, domain in (("A2B", "B"), ("B2A", "A")):
        expected = metrics[f"G_{direction}_adversarial"] + config["training"]["lambda_cycle"] * metrics["cycle_loss"]
        if lambda_identity:
            expected += lambda_identity * metrics[f"identity_{domain}"]
        np.testing.assert_allclose(metrics[f"G_{direction}_loss"], expected)

    for name in names:
        variables = getattr(gan, name).trainable_variables
        assert len(recorded[name]) == len(variables)
        for gradient, previous, current in zip(recorded[name], before[name], variables):
            assert np.all(np.isfinite(gradient))
            assert np.any(gradient != 0)
            assert np.all(np.isfinite(current.numpy()))
            assert np.any(previous != current.numpy())
        assert int(gan.optimizers[name].iterations.numpy()) == 1


def test_cyclegan_loss_values():
    gan = build_cyclegan(load_config("config/config.yaml"))
    real = tf.constant([[[[2.0], [-3.0]]]])
    fake = tf.constant([[[[-1.0], [4.0]]]])
    expected_generator = tf.reduce_mean(tf.nn.softplus(-fake))
    expected_discriminator = 0.5 * tf.reduce_mean(tf.nn.softplus(-real) + tf.nn.softplus(fake))
    np.testing.assert_allclose(gan.generator_loss(fake), expected_generator, rtol=1e-6)
    np.testing.assert_allclose(gan.discriminator_loss(real, fake), expected_discriminator, rtol=1e-6)
    np.testing.assert_allclose(gan.l1_loss(real, fake), 5.0)


def test_cyclegan_graph_steps():
    config = load_config("config/config.yaml")
    config["training"].update(lambda_identity=0.5, beta_1=0.4, beta_2=0.98)
    gan = build_cyclegan(config)
    step = tf.function(gan.train_step)
    for batch_size in (1, 2):
        metrics = step(tf.random.uniform((batch_size, 128, 128, 1), -1, 1),
                       tf.random.uniform((batch_size, 128, 128, 1), -1, 1))
        for value in metrics.values():
            assert np.isfinite(value.numpy())
    for optimizer in gan.optimizers.values():
        assert int(optimizer.iterations.numpy()) == 2
        assert optimizer.beta_1 == 0.4
        assert optimizer.beta_2 == 0.98


@pytest.fixture
def synthetic_training_config(tmp_path):
    """Datos técnicos sintéticos; nunca escribe resultados en las rutas reales."""
    config = load_config("config/config.yaml")
    synthetic_root = tmp_path / "synthetic_technical_only"
    for name in config["paths"]:
        config["paths"][name] = str(synthetic_root / name)
    config["training"].update(epochs=3, batch_size=1, checkpoint_every=2)
    rng = np.random.default_rng(123)
    for domain in ("A", "B"):
        path = synthetic_root / f"train_{domain.lower()}"
        path.mkdir(parents=True)
        np.save(path / f"train{domain}_data.npy",
                rng.uniform(-1, 1, (2, 128, 128, 1)).astype(np.float32))
    return config


def test_train_synthetic_end_to_end(synthetic_training_config, monkeypatch):
    """Entrenamiento real sobre ruido sintético: épocas, CSV, curvas y pesos."""
    import copy
    import csv
    from pathlib import Path
    import src.train as training_module

    config = synthetic_training_config
    original_config = copy.deepcopy(config)
    gan = build_cyclegan(config)
    names = ("G_A2B", "G_B2A", "D_A", "D_B")
    before = {name: [v.numpy().copy() for v in getattr(gan, name).trainable_variables]
              for name in names}
    observed = []
    original_step = gan.train_step

    def observe_step(batch_A, batch_B):
        metrics = original_step(batch_A, batch_B)
        observed.append({name: float(value.numpy()) for name, value in metrics.items()})
        return metrics

    monkeypatch.setattr(gan, "train_step", observe_step)
    monkeypatch.setattr(training_module, "build_cyclegan", lambda cfg: gan)
    training_module.train(config)
    assert config == original_config
    assert len(observed) == 6
    with (Path(config["paths"]["metrics"]) / "training_log.csv").open() as log:
        rows = list(csv.DictReader(log))
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3]
    for epoch, row in enumerate(rows):
        assert int(row["steps"]) == 2
        assert int(row["samples_per_domain"]) == 2
        for name in observed[0]:
            expected = np.mean([step[name] for step in observed[epoch * 2:epoch * 2 + 2]])
            assert float(row[name]) == pytest.approx(expected)
    plot = Path(config["paths"]["plots"]) / "training_losses.png"
    assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    checkpoint_dir = Path(config["paths"]["checkpoints"])
    assert sorted(p.name for p in checkpoint_dir.iterdir() if p.is_dir()) == ["epoch_0002", "epoch_0003"]
    for epoch in (2, 3):
        for name in names:
            path = checkpoint_dir / f"epoch_{epoch:04d}" / f"{name}.weights.h5"
            assert path.is_file()
            restored = tf.keras.models.clone_model(getattr(gan, name))
            restored.load_weights(path)
    for name in names:
        model = getattr(gan, name)
        assert any(np.any(previous != current.numpy())
                   for previous, current in zip(before[name], model.trainable_variables))
        assert int(gan.optimizers[name].iterations.numpy()) == 6
        if name.startswith("G_"):
            restored = build_generator(name=name)
            restored.load_weights(checkpoint_dir / f"{name}.weights.h5")
            sample = tf.random.uniform((1, 128, 128, 1), -1, 1)
            tf.debugging.assert_near(model(sample), restored(sample))
            for weight, reloaded in zip(model.weights, restored.weights):
                np.testing.assert_array_equal(weight.numpy(), reloaded.numpy())


@pytest.mark.parametrize("case", ["missing", "empty_domain", "no_full_batch", "float64", "shape", "nan", "low", "high"])
def test_train_rejects_invalid_data(synthetic_training_config, monkeypatch, case):
    from pathlib import Path
    import src.train as training_module

    config = synthetic_training_config
    path = Path(config["paths"]["train_a"]) / "trainA_data.npy"
    data = np.load(path)
    if case == "missing":
        path.unlink()
    elif case == "empty_domain":
        np.save(path, data[:0])
    elif case == "no_full_batch":
        config["training"]["batch_size"] = 3
    elif case == "float64":
        np.save(path, data.astype(np.float64))
    elif case == "shape":
        np.save(path, data[:, :64])
    else:
        data[:] = {"nan": np.nan, "low": -1.1, "high": 1.1}[case]
        np.save(path, data)

    def unexpected_build(config):
        pytest.fail("No debe construir modelos con el primer batch inválido o ausente")

    monkeypatch.setattr(training_module, "build_cyclegan", unexpected_build)
    match = {"missing": "Faltan datos", "empty_domain": "vacío", "no_full_batch": "vacío",
             "float64": "float32", "shape": "shape", "nan": "no finitos",
             "low": "rango", "high": "rango"}[case]
    with pytest.raises((ValueError, FileNotFoundError, tf.errors.InvalidArgumentError), match=match):
        training_module.train(config)
    assert not Path(config["paths"]["metrics"]).exists()
    assert not Path(config["paths"]["checkpoints"]).exists()


def test_train_validates_later_batches(synthetic_training_config, monkeypatch):
    from pathlib import Path
    import src.train as training_module

    config = synthetic_training_config
    good = tf.zeros((1, 128, 128, 1), dtype=tf.float32)
    bad = tf.fill((1, 128, 128, 1), 2.0)
    dataset = tf.data.Dataset.from_tensor_slices((tf.stack([good, bad]), tf.stack([good, good])))
    monkeypatch.setattr(training_module, "load_unpaired_dataset", lambda cfg, split: dataset)
    gan = build_cyclegan(config)
    monkeypatch.setattr(training_module, "build_cyclegan", lambda cfg: gan)
    with pytest.raises(tf.errors.InvalidArgumentError, match="rango"):
        training_module.train(config)
    for optimizer in gan.optimizers.values():
        assert int(optimizer.iterations.numpy()) == 1
    assert not (Path(config["paths"]["checkpoints"]) / "G_A2B.weights.h5").exists()
