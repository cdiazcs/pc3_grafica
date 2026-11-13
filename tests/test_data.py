"""Integrante 1: Pruebas para el pipeline de datos."""
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from src.common.config import load_config
from src.data.dataset import load_unpaired_dataset

@pytest.fixture
def config():
    return load_config("config/config.yaml")

def test_dataset_shape_and_range(config):
    """Prueba que el dataset devuelva las formas y rangos acordados."""
    ds = load_unpaired_dataset(config, split="train")
    
    # Tomar un batch
    for batch_a, batch_b in ds.take(1):
        # 1. Verificar shapes
        expected_shape = (config['training']['batch_size'], 128, 128, 1)
        assert batch_a.shape == expected_shape, f"Shape A incorrecto: {batch_a.shape}"
        assert batch_b.shape == expected_shape, f"Shape B incorrecto: {batch_b.shape}"
        
        # 2. Verificar tipo de dato
        assert batch_a.dtype == tf.float32, "El tipo de dato de A no es float32"
        assert batch_b.dtype == tf.float32, "El tipo de dato de B no es float32"
        
        # 3. Verificar rango [-1, 1]
        min_a, max_a = tf.reduce_min(batch_a).numpy(), tf.reduce_max(batch_a).numpy()
        assert min_a >= -1.0 and max_a <= 1.0, f"Valores fuera de rango en A: [{min_a}, {max_a}]"
        
        min_b, max_b = tf.reduce_min(batch_b).numpy(), tf.reduce_max(batch_b).numpy()
        assert min_b >= -1.0 and max_b <= 1.0, f"Valores fuera de rango en B: [{min_b}, {max_b}]"

def test_dataset_splits(config):
    """Prueba que existan los splits de train y test."""
    try:
        train_ds = load_unpaired_dataset(config, split="train")
        test_ds = load_unpaired_dataset(config, split="test")
        assert train_ds is not None
        assert test_ds is not None
    except Exception as e:
        pytest.fail(f"Error al cargar los splits: {e}")

def test_invalid_split(config):
    """Prueba que lance error con un split inválido."""
    with pytest.raises(ValueError, match="split debe ser 'train' o 'test'"):
        load_unpaired_dataset(config, split="validation")