"""Integrante 1: audios (30 s) -> espectrogramas Mel -> patches .npy."""
import argparse
import os
import numpy as np
import librosa
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.common.config import load_config


def normalize_melspec(melspec):
    """Normaliza un espectrograma al rango [-1, 1]."""
    min_val = np.min(melspec)
    max_val = np.max(melspec)
    if max_val - min_val == 0:
        return np.zeros_like(melspec) - 1.0
    norm = (melspec - min_val) / (max_val - min_val)
    return (norm * 2.0) - 1.0


def process_song(file_path, config):
    """Extrae parches (128, 128, 1) de una sola canción."""
    sr = config['audio']['sample_rate']
    n_fft = config['audio']['n_fft']
    hop_length = config['audio']['hop_length']
    n_mels = config['audio']['n_mels']
    patch_frames = config['audio']['patch_frames']
    
    # Cargar audio
    y, _ = librosa.load(file_path, sr=sr)
    
    # Generar Espectrograma Mel (en dB)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Normalizar TODO el audio primero (para mantener consistencia por canción)
    S_norm = normalize_melspec(S_db)
    
    patches = []
    # Extraer parches sin solapamiento (o con solapamiento si lo prefieres)
    for i in range(0, S_norm.shape[1] - patch_frames + 1, patch_frames):
        patch = S_norm[:, i:i+patch_frames]
        patch = np.expand_dims(patch, axis=-1) # Shape: (128, 128, 1)
        patches.append(patch)
        
    return np.array(patches, dtype=np.float32)


def prepare_data(config: dict) -> None:
    """Implementa audio, espectrogramas y particion por cancion."""
    raw_a = Path(config['paths']['raw_a'])
    raw_b = Path(config['paths']['raw_b'])
    
    # Buscar audios (.wav, .mp3, etc.)
    files_a = [f for f in raw_a.glob('*') if f.suffix in ['.wav', '.mp3', '.flac']]
    files_b = [f for f in raw_b.glob('*') if f.suffix in ['.wav', '.mp3', '.flac']]
    
    test_frac = config['audio']['test_fraction_by_song']
    
    # 1. DIVISIÓN POR CANCIÓN (Evita fuga de datos)
    train_a_files, test_a_files = train_test_split(files_a, test_size=test_frac, random_state=config['project']['seed'])
    train_b_files, test_b_files = train_test_split(files_b, test_size=test_frac, random_state=config['project']['seed'])
    
    splits = {
        'trainA': train_a_files, 'trainB': train_b_files,
        'testA': test_a_files, 'testB': test_b_files
    }
    
    paths = {
        'trainA': config['paths']['train_a'], 'trainB': config['paths']['train_b'],
        'testA': config['paths']['test_a'], 'testB': config['paths']['test_b']
    }
    
    metadata = {}
    
    # 2. Procesar y guardar
    for split_name, file_list in splits.items():
        all_patches = []
        for f in file_list:
            patches = process_song(f, config)
            all_patches.append(patches)
            metadata[str(f.name)] = {"split": split_name, "patches_generated": len(patches)}
            
        if all_patches:
            final_array = np.concatenate(all_patches, axis=0)
            os.makedirs(paths[split_name], exist_ok=True)
            np.save(Path(paths[split_name]) / f"{split_name}_data.npy", final_array)
            print(f"Guardado {split_name}: {final_array.shape} en {paths[split_name]}")
            
    # Guardar metadatos
    with open("data/processed/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesar archivos musicales para CycleGAN")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    prepare_data(load_config(args.config))


if __name__ == "__main__":
    main()