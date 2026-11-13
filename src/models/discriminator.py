"""Discriminador PatchGAN monocanal con salida de logits."""

import tensorflow as tf


def build_discriminator(input_shape: tuple[int, int, int] = (128, 128, 1), name: str = "D_A") -> tf.keras.Model:
    """Construye un PatchGAN de parches 70x70 (mapa 16x16 para entrada 128x128)."""
    if len(input_shape) != 3 or input_shape[-1] != 1 or any(
        size is None or size < 32 for size in input_shape[:2]
    ):
        raise ValueError("input_shape debe ser (alto, ancho, 1), con alto/ancho >= 32")

    layers = tf.keras.layers
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for index, (filters, stride) in enumerate(((32, 2), (64, 2), (128, 2), (256, 1))):
        x = layers.Conv2D(
            filters, 4, strides=stride, padding="same", use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
        )(x)
        if index:
            x = layers.GroupNormalization(groups=-1, axis=-1, epsilon=1e-5)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
    outputs = layers.Conv2D(
        1, 4, strides=1, padding="same", dtype="float32",
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
    )(x)
    return tf.keras.Model(inputs, outputs, name=name)
