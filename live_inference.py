import torch
import torchaudio
import numpy as np
import os
import json
import argparse
import time
import pandas as pd
import csv
import matplotlib.pyplot as plt
import signal
from datetime import datetime
from model import LiSANet, LiSALSTMNet
from postprocessing import PostProcessor
from utils import create_trajectory_plot, save_statistics_report

# Stesse costanti di preprocessing per coerenza con training/inference
AUDIO_SAMPLE_RATE = 48000
CONTEXT_WINDOW_MS = 200
UPDATE_INTERVAL_MS = 50
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 1024
MAX_FREQ_BINS = 128
NUM_MICS = 4

class AudioStreamSimulator:

    WAV_FILE_PREFIX = 'microphone_'
    #WAV_FILE_PREFIX = 'mic'

    def __init__(self, seq_dir):
        self.audio_data = []
        
        for i in range(1, NUM_MICS + 1):
            path = os.path.join(seq_dir, 'sound', f'{self.WAV_FILE_PREFIX}{i}.wav')
            waveform, sr = torchaudio.load(path)
            if sr != AUDIO_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE)
                waveform = resampler(waveform)
            self.audio_data.append(waveform[0])
            
        self.audio_data = torch.stack(self.audio_data)
        self.current_idx = 0
        self.total_samples = self.audio_data.shape[1]
        self.chunk_size = int((UPDATE_INTERVAL_MS / 1000.0) * AUDIO_SAMPLE_RATE)
        
    def get_next_chunk(self):
        # Calculate current timestamp before updating index
        timestamp = self.current_idx / AUDIO_SAMPLE_RATE
        
        end_idx = self.current_idx + self.chunk_size
        
        if end_idx > self.total_samples:
            remaining = self.total_samples - self.current_idx
            part1 = self.audio_data[:, self.current_idx:]
            part2 = self.audio_data[:, :self.chunk_size - remaining]
            chunk = torch.cat([part1, part2], dim=1)
            self.current_idx = self.chunk_size - remaining
            # Reset timestamp if looped (optional, depends on needs)
        else:
            chunk = self.audio_data[:, self.current_idx:end_idx]
            self.current_idx = end_idx
            
        return chunk, timestamp

class OnlinePreprocessor:
    def __init__(self):
        self.buffer_size = int((CONTEXT_WINDOW_MS / 1000.0) * AUDIO_SAMPLE_RATE)
        self.buffer = torch.zeros(NUM_MICS, self.buffer_size)
        self.window_fn = torch.hann_window(WIN_LENGTH)
        
    def process(self, new_chunk):
        chunk_len = new_chunk.shape[1]
        self.buffer = torch.roll(self.buffer, -chunk_len, dims=1)
        self.buffer[:, -chunk_len:] = new_chunk
        
        stft = torch.stft(
            self.buffer,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window_fn,
            return_complex=True,
            center=False,
            normalized=False
        )
        
        stft = stft[:, :MAX_FREQ_BINS, :]
        spec = torch.cat([stft.real, stft.imag], dim=0)
        return spec.unsqueeze(0).unsqueeze(0)

def get_interpolated_gt(gt_df, current_time):
    # Find the row with the closest timestamp
    closest_idx = (gt_df['time_s'] - current_time).abs().idxmin()
    row = gt_df.iloc[closest_idx]
    is_active = int(row['is_active']) if 'is_active' in row else 0
    return row['dist'], row['angle'], is_active

def generate_plots_and_statistics(csv_path, output_dir, has_gt=True, postprocess_info=None, mic_coords=None):
    """
    Wrapper che genera grafici e statistiche dal CSV di live inference usando utils.
    """
    if not os.path.exists(csv_path):
        return
    
    df = pd.read_csv(csv_path)

    out_plot = os.path.join(output_dir, 'trajectory.png')
    out_stats = os.path.join(output_dir, 'statistics.txt')

    create_trajectory_plot(df, gt_data=None, pred_dists=None, pred_angles=None, 
                           output_path=out_plot, mic_coords=mic_coords, has_gt=has_gt)
                           
    save_statistics_report(df, out_stats, has_gt=has_gt, postprocess_info=postprocess_info)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", type=str, required=True, help="Directory of the raw sequence (e.g., data/raw/seq00/)")
    # La `seq_dir` deve contenere:
    # - una cartella `sound/` con i file `microphone_1.wav`, ..., `microphone_4.wav`
    # - un file `microphones.csv` con le coordinate dei microfoni
    # - un file `gt.csv` con la ground truth (opzionale, per valutazione)
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to the trained model")
    parser.add_argument("--data_root", type=str, default="data", help="Root data directory (deve contenere preprocessing_params.json)")
    parser.add_argument("--output_dir", type=str, default="live_inference_results", help="Directory to save inference outputs")
    
    # Post-processing options
    parser.add_argument("--postprocess", action='store_true', help="Enable Kalman filter post-processing")
    parser.add_argument("--rnn_type", type=str, default='gru', choices=['gru', 'lstm'], help="RNN architecture to use")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica parametri di normalizzazione
    params_path = os.path.join(args.data_root, 'preprocessing_params.json')
    with open(params_path) as f:
        preproc = json.load(f)
    mean_inv_dist = preproc['normalization']['mean_inv_dist']
    std_inv_dist  = preproc['normalization']['std_inv_dist']
    print(f"Normalization stats: mean_inv_dist={mean_inv_dist:.6f}, std_inv_dist={std_inv_dist:.6f}")

    # Load Microphones
    mic_path = os.path.join(args.seq_dir, 'microphones.csv')
    mics_df = pd.read_csv(mic_path)
    mic_coords = torch.tensor(mics_df[['mx', 'my', 'mz']].values, dtype=torch.float32).to(device)
    mic_coords = mic_coords.unsqueeze(0)

    # Load Ground Truth (optional)
    gt_path = os.path.join(args.seq_dir, 'gt.csv')
    has_gt = os.path.exists(gt_path)
    
    if has_gt:
        gt_df = pd.read_csv(gt_path)
        print("Ground truth found - evaluation mode enabled")
    else:
        gt_df = None
        print("No ground truth found - prediction-only mode")

    # Load Model
    print(f"Initializing Model (RNN Type: {args.rnn_type.upper()})...")
    if args.rnn_type == 'gru':
        model = LiSANet(input_channels=8, gru_hidden_size=256, num_gru_layers=2)
    else:
        model = LiSALSTMNet(input_channels=8, lstm_hidden_size=256, num_lstm_layers=2)
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    streamer = AudioStreamSimulator(args.seq_dir)
    preprocessor = OnlinePreprocessor()
    hidden_state = None
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create output directory with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp_str)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CSV Logger
    csv_path = os.path.join(output_dir, 'inference_log.csv')
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    
    if has_gt:
        csv_writer.writerow(['time_s', 'pred_dist', 'gt_dist', 'pred_angle', 'gt_angle',
                             'pred_active_prob', 'gt_active', 'latency_ms'])
    else:
        csv_writer.writerow(['time_s', 'pred_dist', 'pred_angle', 'pred_active_prob', 'latency_ms'])

    print(f"Logging to {csv_path}...")
    print(f"Output directory: {output_dir}")
    
    print("Press Ctrl+C to stop...")

    # Durata target di ogni ciclo in secondi (es. 0.05s)
    target_loop_time = UPDATE_INTERVAL_MS / 1000.0
    
    # Imposta il "clock" iniziale
    next_deadline = time.time() + target_loop_time

    # Iniziallizza il post-processor se richiesto
    if args.postprocess:
        smoother = PostProcessor()
        print(f"Post-processing enabled.")

    try:
        # Run for the length of the file (stop at loop)
        while True:
            start_proc = time.time()
            
            chunk, timestamp = streamer.get_next_chunk()
            
            # Stop if we looped (timestamp reset) or exceeded max time
            if timestamp > streamer.total_samples / AUDIO_SAMPLE_RATE:
                break
            
            input_tensor = preprocessor.process(chunk).to(device)
            
            with torch.no_grad():
                pred_dist, pred_accdoa, hidden_state = model(input_tensor, mic_coords, hidden_state)
            
            sin_val = pred_accdoa[0, 0, 0].item()
            cos_val = pred_accdoa[0, 0, 1].item()
            # ACCDOA: norma vettoriale è la probabilità
            pred_active_prob = np.sqrt(sin_val**2 + cos_val**2)
            
            if pred_active_prob < 0.5:
                dist_m = 0.0
                angle_deg = 0.0
            else:
                # Denormalizza inv_dist normalizzata → distanza reale in metri
                inv_dist = pred_dist.item() * std_inv_dist + mean_inv_dist
                inv_dist = max(inv_dist, 1e-6)
                dist_m = 1.0 / inv_dist
                angle_deg = np.degrees(np.arctan2(sin_val, cos_val))

            # Post-process if requested
            if args.postprocess:
                dist_m, angle_deg = smoother.update(dist_m, angle_deg, is_active=(pred_active_prob >= 0.5))
            latency = (time.time() - start_proc) * 1000
            
            # Write to log
            if has_gt:
                # Get Ground Truth
                gt_d, gt_a, gt_active = get_interpolated_gt(gt_df, timestamp)
                csv_writer.writerow([
                    f"{timestamp:.3f}",
                    f"{dist_m:.3f}",
                    f"{gt_d:.3f}",
                    f"{angle_deg:.3f}",
                    f"{gt_a:.3f}",
                    f"{pred_active_prob:.4f}",
                    f"{gt_active}",
                    f"{latency:.1f}"
                ])
            else:
                csv_writer.writerow([
                    f"{timestamp:.3f}",
                    f"{dist_m:.3f}",
                    f"{angle_deg:.3f}",
                    f"{pred_active_prob:.4f}",
                    f"{latency:.1f}"
                ])
            
            # Flush CSV periodically to ensure data is written
            csv_file.flush()

            # 2. GESTIONE TEMPO REALE
            now = time.time()
            time_to_wait = next_deadline - now
            
            if time_to_wait > 0:
                # Se siamo in anticipo, dormiamo il giusto per sincronizzarci
                time.sleep(time_to_wait)
            else:
                # Se time_to_wait è negativo, significa che il modello è LENTO!
                # Stiamo violando il vincolo real-time.
                lag = abs(time_to_wait) * 1000
                print(f"WARNING: System lag! Processing took too long. Behind by {lag:.1f}ms")
                
                # Opzionale: Resettiamo la deadline per non accumulare ritardo sul prossimo frame
                # (Skip frame logic)
                next_deadline = time.time()

            # Imposta la sveglia per il prossimo frame
            next_deadline += target_loop_time
            
            # Optional: Print every 10 steps to avoid clutter
            if int(timestamp * 1000) % 500 == 0:  # Every 0.5 seconds
                print(f"T: {timestamp:.2f}s | Dist: {dist_m:.1f}m | Processing Load: {(1 - time_to_wait/target_loop_time)*100:.0f}%")

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        csv_file.close()
        
        # Generate plots (always) and statistics (only if has_gt)
        postprocess_info = {
            'enabled': args.postprocess,
            'method': args.smooth_method if args.postprocess else None,
            'history': args.history if args.postprocess else None
        }
        generate_plots_and_statistics(csv_path, output_dir, has_gt, postprocess_info,
                                       mic_coords=mic_coords.squeeze(0).cpu().numpy())
        
        print(f"Results saved to: {output_dir}/")
        print(f"  - inference_log.csv")
        print(f"  - trajectory.png")
        if has_gt:
            print(f"  - statistics.txt")

if __name__ == "__main__":
    main()