"""Integrante 2: discriminador PatchGAN de espectrogramas.

Contrato: build_discriminator(input_shape=(128, 128, 1), name='D_A')
-> tf.keras.Model; devuelve un mapa de logits (sin sigmoid si
la perdida usa BinaryCrossentropy(from_logits=True)).
"""


def build_discriminator(input_shape: tuple[int, int, int] = (128, 128, 1), name: str = "D_A"):
    """TODO (Integrante 2): devolver un discriminador Keras."""
    raise NotImplementedError("Integrante 2: implementar build_discriminator()")
