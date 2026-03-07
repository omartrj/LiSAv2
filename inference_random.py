import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import random
import time
from tqdm import tqdm
from model import LiSANet, LiSALSTMNet
from postprocessing import PostProcessor
from utils import create_trajectory_plot, save_statistics_report

# Importiamo le funzioni core dallo script inference per non duplicare codice!
from inference import load_sequence, inference_on_sequence, apply_postprocessing, save_predictions_csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_random_test_sequence(processed_dir='data/processed', val_split=0.2, test_split=0.1, seed=420):
    """
    Replica la logica di divisione in split di dataset.py per ottenere la lista
    precisa dei file di test, e ne estrae uno in modo casuale.
    """
    all_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
    
    # Sorting per garantire ripetibilità degli split a prescindere dall'OS
    all_files.sort()
    
    # Seed usato dal dataset per generare val e test sets
    random.seed(seed)
    random.shuffle(all_files)
    
    total_files = len(all_files)
    test_count = int(total_files * test_split)
    val_count = int(total_files * val_split)
    train_count = total_files - test_count - val_count
    
    test_files = all_files[train_count+val_count:]
    
    if len(test_files) == 0:
        raise ValueError("No test sequences found! Dataset too small.")
    
    # Ora usiamo un seed basato sul tempo per scegliere una sequenza a caso tra quelle di test
    random.seed(time.time())
    seq_file = random.choice(test_files)
    seq_name = seq_file.replace('.pt', '')
    return seq_name

def main(args):
    # 1. Trova una sequenza di test casuale
    seq_name = get_random_test_sequence(args.processed_dir)
    print(f"Selected random test sequence: {seq_name}\n")
    
    # 2. Carica Sequenza
    print(f"Loading sequence: {seq_name}")
    data = load_sequence(seq_name, args.processed_dir)
    
    # 3. Carica Modello
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
    
    # 4. Esegui Inferenza (importata da inference.py)
    pred_dists, pred_angles, pred_actives, gt_data = inference_on_sequence(model, data, DEVICE)
    
    # 5. Applica Post-Processing (opzionale)
    postprocess_info = {
        'enabled': args.postprocess,
        'method': 'kalman' if args.postprocess else None, # Default to kalman if enabled
        'history': None # History is no longer a parameter
    }
    
    if args.postprocess:
        postprocessor = PostProcessor()
        
        smooth_dists = []
        smooth_angles = []
        
        for d, a, prob in zip(pred_dists, pred_angles, pred_actives):
            sd, sa = postprocessor.update(d, a, is_active=(prob >= 0.5))
            smooth_dists.append(sd)
            smooth_angles.append(sa)
        
        pred_dists = np.array(smooth_dists)
        pred_angles = np.array(smooth_angles)
    
    # 6. Crea cartelle di output
    output_dir = os.path.join(args.output_dir, seq_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 7. Salva Predizioni CSV
    csv_path = os.path.join(output_dir, 'predictions.csv')
    df = save_predictions_csv(gt_data, pred_dists, pred_angles, pred_actives, csv_path, sample_rate=20)
    
    # 8. Genera Grafici Traiettoria
    plot_check_path = os.path.join(output_dir, 'trajectory.png')
    create_trajectory_plot(df, gt_data, pred_dists, pred_angles, plot_check_path, mic_coords=data['microphones'].numpy())
    
    # 9. Genera e Salva il Report Statistico
    stats_path = os.path.join(output_dir, 'statistics.txt')
    save_statistics_report(df, stats_path, postprocess_info=postprocess_info)
    
    print(f"\nAll outputs saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a random TEST sequence and visualize results."
    )
    # Da notare l'assenza del paremtro obbligatorio --seq
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to the trained model")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Directory with processed dataset")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Directory to save inference outputs")

    # Post-processing options
    parser.add_argument("--postprocess", action='store_true', help="Enable Kalman filter post-processing")
    parser.add_argument("--rnn_type", type=str, default='gru', choices=['gru', 'lstm'], help="RNN architecture to use")
    
    args = parser.parse_args()
    main(args)
