import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import json
import random
from tqdm import tqdm

# COSTANTI
AUDIO_SAMPLE_RATE = 48000   # Target Sample Rate
GT_SAMPLE_RATE = 20         # Frequenza aggiornamento ground truth (50ms)
CONTEXT_WINDOW_MS = 200     # Finestra audio passata al modello (200ms)
NUM_MICROPHONES = 4

# Parametri STFT (Devono coincidere con quelli usati in inference/training)
N_FFT = 1024                # Numero di campioni per finestra (21.3ms a 48kHz) 
WIN_LENGTH = 1024           # Lunghezza della finestra (stesso di N_FFT per finestre non sovrapposte)
HOP_LENGTH = 160            # Passo di avanzamento (160 campioni = 3.3ms a 48kHz, per avere ~30 frame per finestra di 100ms)

# Ottimizzazione
MAX_FREQ_BINS = 128 # 128 bin * (48000Hz / 1024) = ~6000Hz. Copre fondamentali e armoniche della sirena (500Hz-4kHz), scartando frequenze inutili

# Percorsi
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
TRAIN_SPLIT_DIR = os.path.join(DATA_DIR, 'train_split')
VAL_SPLIT_DIR   = os.path.join(DATA_DIR, 'val_split')
TEST_SPLIT_DIR  = os.path.join(DATA_DIR, 'test_split')
AUDIO_SUBDIR = 'sound'
GT_FILE = 'gt.csv'
MIC_FILE = 'microphones.csv'

# Parametri split train/val/test
VAL_RATIO  = 0.2
TEST_RATIO = 0.1
SEED = 420


def process_sequence(seq_name):
    """Processa una sequenza e restituisce i dati grezzi (inv_dist non ancora normalizzato)."""
    seq_path = os.path.join(RAW_DATA_DIR, seq_name)
    audio_path = os.path.join(seq_path, AUDIO_SUBDIR)
    
    # Carica ground truth e posizioni microfoni
    try:
        gt_df = pd.read_csv(os.path.join(seq_path, GT_FILE))
        mics_df = pd.read_csv(os.path.join(seq_path, MIC_FILE))
    except FileNotFoundError as e:
        print(f"Skipping {seq_name}: Metadata not found ({e})")
        return None

    mic_coords = torch.tensor(mics_df[['mx', 'my', 'mz']].values, dtype=torch.float32)

    # Carica i 4 file audio dei microfoni
    waveforms = []
    for i in range(1, NUM_MICROPHONES + 1):
        af = os.path.join(audio_path, f'microphone_{i}.wav')
        if not os.path.exists(af):
            print(f"Skipping {seq_name}: {af} not found.")
            return None
        waveform, sr = torchaudio.load(af)
        waveforms.append(waveform[0]) # Prendiamo solo il canale mono

    # Stack: (4, Total_Samples)
    full_audio = torch.stack(waveforms)
    
    # Finestra di Hann per la STFT
    window_fn = torch.hann_window(WIN_LENGTH)

    # Quanti campioni audio servono per la finestra temporale di contesto (es. 200ms)?
    samples_per_window = int((CONTEXT_WINDOW_MS / 1000.0) * AUDIO_SAMPLE_RATE) # es. 9600 campioni a 48kHz
    
    # OTTIMIZZAZIONE: Calcoliamo la STFT sull'intero file audio in un colpo solo.
    # Poi, nel ciclo dei timestep GT, ci limitiamo ad "affettare" (slicing) la matrice globale.
    global_stft = torch.stft(
        full_audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window_fn,
        return_complex=True,
        center=False, 
        normalized=False
    )
    
    # Filtro passa basso globale (eliminiamo frequenze doppie oltre i ~6kHz)
    global_stft = global_stft[:, :MAX_FREQ_BINS, :]
    
    # Pre-concateniamo canali Reali e Immaginari globalmente -> (8, F, Total_TimeFrames)
    spec_global = torch.cat([global_stft.real, global_stft.imag], dim=0)
    
    # Quanti frame temporali di STFT formano una contesto di 200ms?
    frames_per_window = (samples_per_window - WIN_LENGTH) // HOP_LENGTH + 1
    
    spectrogram_list = []
    gt_list = []

    # Itera i campioni di gt.csv e processa l'audio corrispondente
    for _, row in gt_df.iterrows():
        t_end_sec = row['time_s']
        
        # Indici nel file audio (usiamo round per le approssimazioni float)
        end_idx = int(round(t_end_sec * AUDIO_SAMPLE_RATE))
        start_idx = end_idx - samples_per_window

        # Se siamo all'inizio della registrazione e non abbiamo tutto lo storico, saltiamo
        if start_idx < 0:
            continue
            
        # Controllo fine file
        if end_idx > full_audio.shape[1]:
            continue
        
        # L'offset temporale è garantito essere un multiplo perfetto dell'HOP_LENGTH 
        # (es. a 20Hz e 48kHz il balzo è di 2400 campioni, multiplo di 160). 
        # Troviamo l'indice del frame di partenza nella STFT globale.
        f_start = start_idx // HOP_LENGTH
        
        # Taglio la fetta temporale esatta dallo spettrogramma pre-calcolato
        spec_tensor = spec_global[:, :, f_start : f_start + frames_per_window]
        
        # Estrae i target di distanza, angolo e attività
        dist = row['dist']
        angle_deg = row['angle']
        is_active = float(row['is_active'])

        # Inversa della distanza (solo per campioni attivi; 0 per inattivi, verrà ignorato in training)
        if is_active > 0.5 and dist > 0:
            inv_dist = 1.0 / dist
        else:
            inv_dist = 0.0

        # Calcolo sin(theta) e cos(theta)
        angle_rad = np.deg2rad(angle_deg)
        sin_angle = np.sin(angle_rad)
        cos_angle = np.cos(angle_rad)
        
        # (N, 4) con colonne (inv_distance, cos(angle), sin(angle), is_active)
        # NOTA: inv_distance non è ancora normalizzato — la normalizzazione avviene dopo, usando le statistiche del train.
        gt_tensor = torch.tensor([inv_dist, cos_angle, sin_angle, is_active], dtype=torch.float32)

        spectrogram_list.append(spec_tensor)
        gt_list.append(gt_tensor)

    # Restituisce il dizionario grezzo (inv_dist non normalizzato)
    if len(spectrogram_list) > 0:
        return {
            "spectrograms": torch.stack(spectrogram_list), # Shape: (N, 8, F, T)
            "gt": torch.stack(gt_list),                    # Shape: (N, 4)
            "microphones": mic_coords,
        }
    return None


def normalize_and_save(data, out_path, mean_inv_dist, std_inv_dist):
    """Normalizza la colonna inv_dist con le statistiche del train e salva il file .pt."""
    gt = data['gt'].clone()
    is_active = gt[:, 3] > 0.5
    # Normalizzazione solo sui campioni attivi: (inv_dist - mean) / std
    gt[is_active, 0] = (gt[is_active, 0] - mean_inv_dist) / std_inv_dist
    torch.save({
        "spectrograms": data["spectrograms"],
        "gt": gt,
        "microphones": data["microphones"],
    }, out_path)


if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"Directory {RAW_DATA_DIR} not found. Run simulation first.")

    sequences = sorted([d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))])

    # --- Split deterministico delle sequenze ---
    random.seed(SEED)
    shuffled = sequences.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    test_count  = int(total * TEST_RATIO)
    val_count   = int(total * VAL_RATIO)
    train_count = total - test_count - val_count

    train_seqs = shuffled[:train_count]
    val_seqs   = shuffled[train_count:train_count + val_count]
    test_seqs  = shuffled[train_count + val_count:]

    print(f"Found {total} sequences → Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")

    # --- Passo 1: preprocessing delle sequenze di training (in memoria) ---
    print("\nStep 1/4 — Processing training sequences...")
    train_data_list = []
    for seq in tqdm(train_seqs, desc="Train", unit="seq"):
        data = process_sequence(seq)
        if data is not None:
            train_data_list.append((seq, data))

    # --- Passo 2: calcolo media e std di inv_dist SOLO sui campioni attivi del train ---
    print("\nStep 2/4 — Computing normalization statistics from training data...")
    all_inv_dist = []
    for _, data in train_data_list:
        gt = data['gt']
        is_active = gt[:, 3] > 0.5
        if is_active.any():
            all_inv_dist.append(gt[is_active, 0])

    if not all_inv_dist:
        raise ValueError("No active samples found in training data! Cannot compute normalization stats.")

    all_inv_dist_tensor = torch.cat(all_inv_dist)
    mean_inv_dist = all_inv_dist_tensor.mean().item()
    std_inv_dist  = all_inv_dist_tensor.std().item()

    print(f"  inv_dist stats:  mean = {mean_inv_dist:.6f},  std = {std_inv_dist:.6f}")

    # --- Passo 3: normalizzazione e salvataggio di tutti e tre gli split ---
    print("\nStep 3/4 — Normalizing and saving splits...")

    os.makedirs(TRAIN_SPLIT_DIR, exist_ok=True)
    for seq, data in tqdm(train_data_list, desc="Saving train", unit="seq"):
        normalize_and_save(data, os.path.join(TRAIN_SPLIT_DIR, f"{seq}.pt"), mean_inv_dist, std_inv_dist)

    os.makedirs(VAL_SPLIT_DIR, exist_ok=True)
    for seq in tqdm(val_seqs, desc="Processing val", unit="seq"):
        data = process_sequence(seq)
        if data is not None:
            normalize_and_save(data, os.path.join(VAL_SPLIT_DIR, f"{seq}.pt"), mean_inv_dist, std_inv_dist)

    os.makedirs(TEST_SPLIT_DIR, exist_ok=True)
    for seq in tqdm(test_seqs, desc="Processing test", unit="seq"):
        data = process_sequence(seq)
        if data is not None:
            normalize_and_save(data, os.path.join(TEST_SPLIT_DIR, f"{seq}.pt"), mean_inv_dist, std_inv_dist)

    # --- Passo 4: salvataggio dei parametri di preprocessing ---
    print("\nStep 4/4 — Saving preprocessing parameters...")

    frames_per_window = (int((CONTEXT_WINDOW_MS / 1000.0) * AUDIO_SAMPLE_RATE) - WIN_LENGTH) // HOP_LENGTH + 1
    params = {
        "audio_sample_rate":  AUDIO_SAMPLE_RATE,
        "gt_sample_rate":     GT_SAMPLE_RATE,
        "context_window_ms":  CONTEXT_WINDOW_MS,
        "num_microphones":    NUM_MICROPHONES,
        "n_fft":              N_FFT,
        "win_length":         WIN_LENGTH,
        "hop_length":         HOP_LENGTH,
        "max_freq_bins":      MAX_FREQ_BINS,
        "frames_per_window":  frames_per_window,
        "normalization": {
            "target":         "inv_dist",
            "formula":        "gt[:, 0] = (1/dist - mean_inv_dist) / std_inv_dist  [solo campioni attivi]",
            "mean_inv_dist":  mean_inv_dist,
            "std_inv_dist":   std_inv_dist,
        },
    }

    params_path = os.path.join(DATA_DIR, 'preprocessing_params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"  Preprocessing params saved to: {params_path}")
    print("\nProcessing completed.")
