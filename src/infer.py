"""Integrante 1: inferencia sin volver a entrenar (pesos guardados)."""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from src.common.config import load_config
# IMPORTANTE: Esta función la debe crear el Integrante 2 en src/models/generator.py
from src.models.generator import build_generator

def infer(config: dict, input_path: str, direction: str, checkpoint_path: str | None = None) -> None:
    """Cargar solo generador necesario y guardar .npy/PNG."""
    name = "G_A2B" if direction == "A2B" else "G_B2A"
    
    # 1. Instanciar generador
    generator = build_generator(input_shape=tuple(config['model']['input_shape']), name=name)
    
    # 2. Cargar pesos
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config['paths']['checkpoints'], f"{name}.weights.h5")
    
    generator.load_weights(checkpoint_path)
    print(f"Pesos cargados de: {checkpoint_path}")
    
    # 3. Cargar input
    patch = np.load(input_path) # Asume shape (128, 128, 1)
    patch_batch = np.expand_dims(patch, axis=0) # (1, 128, 128, 1)
    
    # 4. Predecir
    prediction = generator.predict(patch_batch)[0] # (128, 128, 1)
    
    # 5. Guardar resultado
    output_dir = config['paths']['generated']
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path).replace('.npy', '')
    out_npy = os.path.join(output_dir, f"{base_name}_{direction}.npy")
    np.save(out_npy, prediction)
    print(f"Traducción guardada en: {out_npy}")
    
    # 6. Guardar visualización rápida
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(patch.squeeze(), cmap='magma', aspect='auto')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(prediction.squeeze(), cmap='magma', aspect='auto')
    plt.title(f"Traducido ({direction})")
    plt.savefig(out_npy.replace('.npy', '.png'))
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia de espectrogramas CycleGAN")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", required=True, help="Ruta a patch .npy")
    parser.add_argument("--direction", choices=("A2B", "B2A"), default="A2B")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    infer(load_config(args.config), args.input, args.direction, args.checkpoint)

if __name__ == "__main__":
    main()