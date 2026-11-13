"""Generador U-Net monocanal para espectrogramas normalizados en [-1, 1]."""

import tensorflow as tf


def build_generator(input_shape: tuple[int, int, int] = (128, 128, 1), name: str = "G_A2B") -> tf.keras.Model:
    """Construye una U-Net; las dimensiones espaciales deben ser múltiplos de 32."""
    if len(input_shape) != 3 or input_shape[-1] != 1 or any(
        size is None or size < 32 or size % 32 for size in input_shape[:2]
    ):
        raise ValueError("input_shape debe ser (alto, ancho, 1), con alto/ancho múltiplos de 32")

    layers = tf.keras.layers
    inputs = layers.Input(shape=input_shape)
    x = inputs
    skips = []
    for index, filters in enumerate((32, 64, 128, 256, 512)):
        x = layers.Conv2D(
            filters, 4, strides=2, padding="same", use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
        )(x)
        if index:
            # Un grupo por canal equivale a normalización de instancia.
            x = layers.GroupNormalization(groups=-1, axis=-1, epsilon=1e-5)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        skips.append(x)

    for filters, skip in zip((256, 128, 64, 32), reversed(skips[:-1])):
        x = layers.Conv2DTranspose(
            filters, 4, strides=2, padding="same", use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
        )(x)
        x = layers.GroupNormalization(groups=-1, axis=-1, epsilon=1e-5)(x)
        x = layers.ReLU()(x)
        x = layers.Concatenate(axis=-1)([x, skip])

    outputs = layers.Conv2DTranspose(
        1, 4, strides=2, padding="same", activation="tanh", dtype="float32",
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
    )(x)
    return tf.keras.Model(inputs, outputs, name=name)
