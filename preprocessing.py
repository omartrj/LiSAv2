"""
Questo file è pensato per essere eseguito una sola volta, prima di far partire il training.
Il suo scopo è quello di processare le sequenze grezze che si trovano in `data/raw` e di salvarle in `data/processed` in un formato più comodo per il training.
Ogni sequenza *raw* è composta da:
- 4 file audio (dentro la cartella `sound`), uno per ogni microfono, in formato .wav
- `gt.csv`, con header `time_s,sx,sy,dist,angle`, che contiene le coordinate (sx, sy), la distanza e l'angolo della sorgente sonora rispetto all'origine del sistema di riferimento, campionati ogni 0.05 secondi (20Hz)
- `microphones.csv`, con header `mic_id,mx,my,mz`, che contiene le coordinate (mx, my, mz) di ogni microfono, in metri, rispetto all'origine del sistema di riferimento.
In particolare, il processamento consiste in:
- Caricare i 4 file audio
- Caricare i file gt.csv e microphones.csv
- Per ogni campione di gt.csv:
    * Estrarre gli **ultimi** CONTEXT_WINDOW_MS millisecondi di audio da ogni microfono. Se non sono disponibili (inizio della registrazione), scarto il campione
    * Calcolare lo spettrogramma complesso di ogni finestra audio -> ottengo 4 spettrogrammi complessi
    * Dato theta (l'angolo in gradi in gt.csv), calcolare sin(theta) e cos(theta) e salvarli come target invece di theta, per evitare problemi di discontinuità circolare
- Una volta processati i campioni di una sequenza, costruire un dizionario con le seguenti chiavi:
    * `spectrograms`: Tensor di forma (N, 8, F, T), dove N è il numero di campioni in gt.csv, 8 è il numero di canali complessi (4 microfoni x 2 canali: reale e immaginario), F è il numero di bande di frequenza dello spettrogramma, T è il numero di frame temporali dello spettrogramma
    * `gt`: Tensor di forma (N, 4), con colonne (distance, sin(angle), cos(angle), is_active)
    * `microphones`: Tensor di forma (NUM_MICROPHONES, 3), contenente le coordinate (x, y, z) di ogni microfono
- Salvare il dizionario in un file `.pt`.
"""
import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import json
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
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
AUDIO_SUBDIR = 'sound'
GT_FILE = 'gt.csv'
MIC_FILE = 'microphones.csv'


def process_sequence(seq_name):
    seq_path = os.path.join(RAW_DATA_DIR, seq_name)
    audio_path = os.path.join(seq_path, AUDIO_SUBDIR)
    
    # Carica ground truth e posizioni microfoni
    try:
        gt_df = pd.read_csv(os.path.join(seq_path, GT_FILE))
        mics_df = pd.read_csv(os.path.join(seq_path, MIC_FILE))
    except FileNotFoundError as e:
        print(f"Skipping {seq_name}: Metadata not found ({e})")
        return

    mic_coords = torch.tensor(mics_df[['mx', 'my', 'mz']].values, dtype=torch.float32)

    # Carica i 4 file audio dei microfoni
    waveforms = []
    for i in range(1, NUM_MICROPHONES + 1):
        af = os.path.join(audio_path, f'microphone_{i}.wav')
        if not os.path.exists(af):
            print(f"Skipping {seq_name}: {af} not found.")
            return
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

        # Calcolo sin(theta) e cos(theta)
        angle_rad = np.deg2rad(angle_deg)
        sin_angle = np.sin(angle_rad)
        cos_angle = np.cos(angle_rad)
        
        # (N, 4) con colonne (distance, sin(angle), cos(angle), is_active)
        gt_tensor = torch.tensor([dist, sin_angle, cos_angle, is_active], dtype=torch.float32)

        spectrogram_list.append(spec_tensor)
        gt_list.append(gt_tensor)

    # Salva
    if len(spectrogram_list) > 0:
        final_specs = torch.stack(spectrogram_list) # Shape: (N, 8, F, T)
        final_gt = torch.stack(gt_list)             # Shape: (N, 4)
        
        processed_dict = {
            "spectrograms": final_specs,
            "gt": final_gt,
            "microphones": mic_coords,
        }

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DATA_DIR, f"{seq_name}.pt")
        torch.save(processed_dict, out_path)

if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"Directory {RAW_DATA_DIR} not found. Run simulation first.")

    sequences = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    sequences.sort()
    
    print(f"Found {len(sequences)} sequences to process.")
    print("Starting preprocessing...\n")
    
    for seq in tqdm(sequences, desc="Processing sequences", unit="seq"):
        process_sequence(seq)
    
    print("\nProcessing completed.")
