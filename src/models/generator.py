"""Integrante 2: generador tipo U-Net adaptado a 1 canal.

Contrato: build_generator(input_shape=(128, 128, 1), name='G_A2B')
-> tf.keras.Model, con salida de igual shape, float32 y tanh [-1, 1].
Partir del notebook de clase, revisar InstanceNormalization y Keras 3.
"""


def build_generator(input_shape: tuple[int, int, int] = (128, 128, 1), name: str = "G_A2B"):
    """TODO (Integrante 2): devolver un generador Keras."""
    raise NotImplementedError("Integrante 2: implementar build_generator()")
