"""Integrante 1: graficos de espectrogramas y perdidas."""
import matplotlib.pyplot as plt

def save_comparison(original, translated, reconstructed, reference, output_path: str) -> None:
    """Guardar comparacion PNG."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    titles = ["Original (A)", "Traducido (A2B)", "Reconstruido (A2B2A)", "Referencia Real (B)"]
    images = [original, translated, reconstructed, reference]
    
    for ax, img, title in zip(axes, images, titles):
        # Remover dimensiones de batch o canal si existen
        img_2d = img.squeeze()
        ax.imshow(img_2d, cmap='magma', aspect='auto', vmin=-1, vmax=1)
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()