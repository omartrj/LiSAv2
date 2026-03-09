import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import json
import argparse
from tqdm import tqdm
from model import LiSANet, LiSALSTMNet
from postprocessing import PostProcessor
from utils import create_trajectory_plot, save_statistics_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_preprocessing_params(data_root):
    params_path = os.path.join(data_root, 'preprocessing_params.json')
    with open(params_path) as f:
        params = json.load(f)
    mean_inv_dist = params['normalization']['mean_inv_dist']
    std_inv_dist  = params['normalization']['std_inv_dist']
    return mean_inv_dist, std_inv_dist

def denorm_to_dist(norm_inv_dist, mean_inv_dist, std_inv_dist):
    """Converte inv_dist normalizzato → distanza reale in metri."""
    inv_dist = norm_inv_dist * std_inv_dist + mean_inv_dist
    inv_dist = max(inv_dist, 1e-6)
    return 1.0 / inv_dist

def load_sequence(seq_name, split_dir):
    seq_path = os.path.join(split_dir, f"{seq_name}.pt")
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Sequence file not found: {seq_path}")
    return torch.load(seq_path, mmap=False, weights_only=False)

def inference_on_sequence(model, data, device, mean_inv_dist, std_inv_dist):
    """
    Esegue l'inferenza su tutti i frame di una sequenza.
    Restituisce distanze in metri reali (denormalizzate).
    """
    # Estrai dati
    spectrograms = data['spectrograms'].to(device)  # (N, 8, F, T)
    gt_data = data['gt']  # (N, 4) -> [dist, sin(angle), cos(angle), is_active]
    mic_coords = data['microphones'].unsqueeze(0).to(device)  # (1, 4, 3)
    
    num_frames = spectrograms.shape[0]
    
    # Storage per risultati
    pred_dists = []
    pred_angles = []
    pred_actives = []
    
    # Hidden state della GRU (stateful)
    hidden_state = None
    
    model.eval()
    print(f"Running inference on {num_frames} frames...")
    
    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            # Prepara input: (1, 1, 8, F, T)
            spec_frame = spectrograms[i].unsqueeze(0).unsqueeze(0)  # Batch=1, Seq=1
            
            # Forward pass
            pred_dist, pred_accdoa, hidden_state = model(spec_frame, mic_coords, hidden_state)
            
            # ACCDOA: norm = probabilità di attività
            pred_active_prob = torch.norm(pred_accdoa[0, 0], p=2).item()
            
            # Estrai sin e cos (NON normalizzati, per l'atan2 conta solo il rapporto)
            pred_sin = pred_accdoa[0, 0, 0]
            pred_cos = pred_accdoa[0, 0, 1]
            
            # Se la sirena non è attiva, azzeriamo distanza e angolo
            if pred_active_prob < 0.5:
                pred_dist_val = 0.0
                pred_angle_deg = 0.0
            else:
                # Denormalizza inv_dist → distanza reale in metri
                pred_dist_val = denorm_to_dist(pred_dist.item(), mean_inv_dist, std_inv_dist)
                pred_angle_rad = torch.atan2(pred_sin, pred_cos)
                pred_angle_deg = torch.rad2deg(pred_angle_rad).item()
            
            # Estrai valori
            pred_dists.append(pred_dist_val)
            pred_angles.append(pred_angle_deg)
            pred_actives.append(pred_active_prob)
    
    return np.array(pred_dists), np.array(pred_angles), np.array(pred_actives), gt_data[:].numpy()

def apply_postprocessing(pred_dists, pred_angles, pred_actives, method='median', history=5):
    """
    Applica post-processing (smoothing) alle predizioni di distanza e angolo.
    La probabilità di attività non viene smoothata (già stabile grazie alla GRU).
    """
    print(f"Applying post-processing ({method}, history={history})...")
    processor = PostProcessor(history_length=history, method=method)
    
    smooth_dists = []
    smooth_angles = []
    
    for d, a, prob in zip(pred_dists, pred_angles, pred_actives):
        sd, sa = processor.update(d, a, is_active=(prob >= 0.5))
        smooth_dists.append(sd)
        smooth_angles.append(sa)
    
    return np.array(smooth_dists), np.array(smooth_angles), pred_actives

def save_predictions_csv(gt_data, pred_dists, pred_angles, pred_actives, output_path,
                         mean_inv_dist=None, std_inv_dist=None, sample_rate=20):
    """
    Salva le predizioni e il ground truth in un CSV.
    gt_data: (N, 4) con [norm_inv_dist, sin(angle), cos(angle), is_active]
    pred_dists: distanze reali già denormalizzate (in metri)
    """
    num_samples = len(pred_dists)
    timestamps = np.arange(num_samples) / sample_rate

    # Denormalizza gt inv_dist → distanza reale in metri
    gt_norm_inv = gt_data[:, 0]
    gt_active = gt_data[:, 3].astype(int)
    if mean_inv_dist is not None and std_inv_dist is not None:
        gt_inv = gt_norm_inv * std_inv_dist + mean_inv_dist
        gt_inv = np.maximum(gt_inv, 1e-6)
        gt_dist_m = np.where(gt_active.astype(bool), 1.0 / gt_inv, 0.0)
    else:
        gt_dist_m = gt_norm_inv

    # Converti gt sin/cos in angoli (gradi)
    gt_angle_rad = np.arctan2(gt_data[:, 1], gt_data[:, 2])
    gt_angle_deg = np.rad2deg(gt_angle_rad)

    pred_active_binary = (pred_actives >= 0.5).astype(int)
    
    df = pd.DataFrame({
        'time_s': timestamps,
        'gt_dist': gt_dist_m,
        'gt_angle': gt_angle_deg,
        'gt_active': gt_active,
        'pred_dist': pred_dists,
        'pred_angle': pred_angles,
        'pred_active_prob': pred_actives,
        'pred_active': pred_active_binary,
        'error_dist': np.abs(gt_dist_m - pred_dists),
        'error_angle': np.abs((gt_angle_deg - pred_angles + 180) % 360 - 180),
        'error_active': (gt_active != pred_active_binary).astype(int)
    })
    
    # Arrotonda a 2 decimali
    df = df.round(2)
    
    df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"Predictions saved to: {output_path}")
    return df



def main(args):
    # 0. Carica parametri di preprocessing
    mean_inv_dist, std_inv_dist = load_preprocessing_params(args.data_root)

    test_dir = os.path.join(args.data_root, 'test_split')

    # 1. Scegli la sequenza
    if args.seq:
        seq_name = args.seq
    else:
        available = [f[:-3] for f in os.listdir(test_dir) if f.endswith('.pt')]
        if not available:
            raise RuntimeError(f"No sequences found in {test_dir}")
        seq_name = random.choice(available)
        print(f"No sequence specified — randomly selected: {seq_name}")

    print(f"Loading sequence: {seq_name}")
    data = load_sequence(seq_name, test_dir)

    # 2. Load Model
    print(f"Loading model from: {args.model_path} (RNN Type: {args.rnn_type.upper()})")
    if args.rnn_type == 'gru':
        model = LiSANet(input_channels=8, gru_hidden_size=256, num_gru_layers=2).to(DEVICE)
    else:
        model = LiSALSTMNet(input_channels=8, lstm_hidden_size=256, num_lstm_layers=2).to(DEVICE)
    
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 3. Run Inference
    pred_dists, pred_angles, pred_actives, gt_data = inference_on_sequence(
        model, data, DEVICE, mean_inv_dist, std_inv_dist
    )
    
    # 4. Applica Post-Processing (opzionale)
    postprocess_info = {
        'enabled': args.postprocess,
        'method': 'kalman' if args.postprocess else None,
        'history': None
    }
    
    if args.postprocess:
        processor = PostProcessor()
        
        smooth_dists = []
        smooth_angles = []
        
        for d, a, prob in zip(pred_dists, pred_angles, pred_actives):
            sd, sa = processor.update(d, a, is_active=(prob >= 0.5))
            smooth_dists.append(sd)
            smooth_angles.append(sa)
            
        pred_dists = np.array(smooth_dists)
        pred_angles = np.array(smooth_angles)
        
    # 5. Crea directory di output
    output_dir = os.path.join(args.output_dir, seq_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 6. Salva CSV e Plot
    csv_path = os.path.join(output_dir, 'predictions.csv')
    df = save_predictions_csv(gt_data, pred_dists, pred_angles, pred_actives, csv_path,
                              mean_inv_dist=mean_inv_dist, std_inv_dist=std_inv_dist, sample_rate=20)
    
    plot_check_path = os.path.join(output_dir, 'trajectory.png')
    create_trajectory_plot(df, gt_data, pred_dists, pred_angles, plot_check_path, mic_coords=data['microphones'].numpy())
    
    stats_path = os.path.join(output_dir, 'statistics.txt')
    save_statistics_report(df, stats_path, postprocess_info=postprocess_info)
    
    print(f"\nAll outputs saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegue inferenza su una singola sequenza del test set.")
    parser.add_argument("--seq", type=str, default=None, help="Sequence name (e.g., seq00). Se omesso, sceglie una sequenza random dal test set.")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to the trained model")
    parser.add_argument("--data_root", type=str, default="data", help="Root data directory (deve contenere test_split/ e preprocessing_params.json)")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Directory to save inference outputs")

    # Post-processing options
    parser.add_argument("--postprocess", action='store_true', help="Enable Kalman filter post-processing")
    parser.add_argument("--rnn_type", type=str, default='gru', choices=['gru', 'lstm'], help="RNN architecture to use")
    
    args = parser.parse_args()
    main(args)
