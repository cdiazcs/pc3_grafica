"""Integrante 1: metricas y comparativas A -> B -> A y B -> A -> B."""
import argparse
import os
import numpy as np
import pandas as pd
from src.common.config import load_config
from src.data.dataset import load_unpaired_dataset
from src.models.generator import build_generator
from src.visualization import save_comparison

def evaluate(config: dict) -> None:
    """Implementar evaluacion reproducible."""
    # 1. Cargar dataset de test
    test_ds = load_unpaired_dataset(config, split="test")
    
    # 2. Cargar modelos
    g_a2b = build_generator(tuple(config['model']['input_shape']), name="G_A2B")
    g_b2a = build_generator(tuple(config['model']['input_shape']), name="G_B2A")
    g_a2b.load_weights(os.path.join(config['paths']['checkpoints'], "G_A2B.weights.h5"))
    g_b2a.load_weights(os.path.join(config['paths']['checkpoints'], "G_B2A.weights.h5"))
    
    results = []
    out_dir = config['paths']['comparisons']
    os.makedirs(out_dir, exist_ok=True)
    
    # Evaluar 5 batches para tener una muestra representativa
    for i, (batch_a, batch_b) in enumerate(test_ds.take(5)):
        # Predicciones
        fake_b = g_a2b.predict(batch_a, verbose=0)
        rec_a = g_b2a.predict(fake_b, verbose=0)
        
        fake_a = g_b2a.predict(batch_b, verbose=0)
        rec_b = g_a2b.predict(fake_a, verbose=0)
        
        # Calcular Cycle Consistency Loss (Error absoluto medio L1)
        loss_a2b2a = np.mean(np.abs(batch_a - rec_a))
        loss_b2a2b = np.mean(np.abs(batch_b - rec_b))
        
        results.append({
            "batch": i,
            "cycle_loss_A2B2A": loss_a2b2a,
            "cycle_loss_B2A2B": loss_b2a2b
        })
        
        # Guardar una imagen de comparación por batch (usando el primer elemento)
        if i == 0: 
            save_comparison(
                batch_a[0], fake_b[0], rec_a[0], batch_b[0],
                os.path.join(out_dir, f"comparison_batch_{i}.png")
            )
            
    # Guardar metricas en CSV
    df = pd.DataFrame(results)
    metrics_path = os.path.join(config['paths']['metrics'], "evaluation_metrics.csv")
    os.makedirs(config['paths']['metrics'], exist_ok=True)
    df.to_csv(metrics_path, index=False)
    print(f"Evaluación completa. Métricas guardadas en {metrics_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluacion CycleGAN")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    evaluate(load_config(args.config))

if __name__ == "__main__":
    main()