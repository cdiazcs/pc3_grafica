"""Integrante 1: datasets no emparejados A y B.

Contrato: load_unpaired_dataset(config, split) -> tf.data.Dataset
con cada elemento (A, B), tensores float32 con shape
(batch, 128, 128, 1), valores en [-1, 1].
No asumir correspondencia semantica entre muestras de A y B.
"""


def load_unpaired_dataset(config: dict, split: str = "train"):
    """TODO (Integrante 1): cargar `.npy` y devolver dataset TensorFlow."""
    if split not in ("train", "test"):
        raise ValueError("split debe ser 'train' o 'test'")
    raise NotImplementedError("Integrante 1: implementar load_unpaired_dataset()")
