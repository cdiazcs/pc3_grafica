"""Integrante 1: datasets no emparejados A y B."""
import tensorflow as tf
import numpy as np
from pathlib import Path

def load_unpaired_dataset(config: dict, split: str = "train"):
    """Carga `.npy` y devuelve dataset TensorFlow."""
    if split not in ("train", "test"):
        raise ValueError("split debe ser 'train' o 'test'")
    
    path_a = Path(config['paths'][f'{split}_a']) / f"{split}A_data.npy"
    path_b = Path(config['paths'][f'{split}_b']) / f"{split}B_data.npy"
    
    # Cargar arrays en memoria (float32)
    data_a = np.load(path_a)
    data_b = np.load(path_b)
    
    batch_size = config['training']['batch_size']
    
    # Crear datasets individuales
    ds_a = tf.data.Dataset.from_tensor_slices(data_a)
    ds_b = tf.data.Dataset.from_tensor_slices(data_b)
    
    # Mezclar independientemente (shuffle)
    buffer_size = min(1000, len(data_a), len(data_b))
    ds_a = ds_a.shuffle(buffer_size=buffer_size, seed=config['project']['seed'])
    ds_b = ds_b.shuffle(buffer_size=buffer_size, seed=config['project']['seed'])
    
    # Agrupar (batch)
    ds_a = ds_a.batch(batch_size, drop_remainder=True)
    ds_b = ds_b.batch(batch_size, drop_remainder=True)
    
    # Unir en tuplas (batch_A, batch_B)
    dataset = tf.data.Dataset.zip((ds_a, ds_b))
    
    # Prefetch para optimizar rendimiento
    return dataset.prefetch(tf.data.AUTOTUNE)