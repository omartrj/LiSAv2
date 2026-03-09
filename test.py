import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from dataset import get_dataloaders
from model import LiSANet, LiSALSTMNet
from postprocessing import PostProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_errors(pred_dist, target_dist, pred_angle_deg, target_angle_deg):
    err_dist_abs = np.abs(pred_dist - target_dist)
    err_dist_pct = (err_dist_abs / (target_dist + 1e-6)) * 100
    
    diff = pred_angle_deg - target_angle_deg
    diff = (diff + 180) % 360 - 180
    err_angle = np.abs(diff)
    
    return err_dist_abs, err_dist_pct, err_angle

def plot_error_by_distance(target_dist, pred_dist, target_angle, pred_angle, output_dir):
    err_dist = np.abs(pred_dist - target_dist)
    diff_angle = (pred_angle - target_angle + 180) % 360 - 180
    err_angle = np.abs(diff_angle)
    
    bins = np.arange(0, np.max(target_dist) + 5, 5)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    dist_mae = []
    angle_mae = []
    
    for i in range(len(bins)-1):
        mask = (target_dist >= bins[i]) & (target_dist < bins[i+1])
        if np.any(mask):
            dist_mae.append(np.mean(err_dist[mask]))
            angle_mae.append(np.mean(err_angle[mask]))
        else:
            dist_mae.append(np.nan)
            angle_mae.append(np.nan)
            
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(bin_centers, dist_mae, marker='o', linestyle='-', color='b')
    plt.xlabel("True Distance (m)")
    plt.ylabel("Distance MAE (m)")
    plt.title("Distance Error vs True Distance")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(bin_centers, angle_mae, marker='o', linestyle='-', color='r')
    plt.xlabel("True Distance (m)")
    plt.ylabel("Angle MAE (°)")
    plt.title("Angle Error vs True Distance")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_vs_distance.png"))
    plt.close()

def evaluate_model(model, loader, mean_inv_dist, std_inv_dist):
    model.eval()
    all_pred_dist, all_target_dist = [], []
    all_pred_angle, all_target_angle = [], []
    all_pred_active, all_target_active = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            specs      = batch['spectrogram'].to(DEVICE)
            gt_dist    = batch['gt_dist'].to(DEVICE)
            gt_angle   = batch['gt_angle'].to(DEVICE)   # (B, Seq, 2) [sin, cos]
            gt_active  = batch['gt_active'].to(DEVICE)  # (B, Seq)
            mic_coords = batch['microphones'].to(DEVICE)
            
            # Forward: nuovo output con ACCDOA
            pred_dist, pred_accdoa, _ = model(specs, mic_coords, hidden_state=None)
            
            # Denormalizza pred_dist (inv_dist norm → metri)
            # Solo sui frame attivi; gli inattivi verranno azzerati comunque
            pred_inv = pred_dist * std_inv_dist + mean_inv_dist
            pred_inv = torch.clamp(pred_inv, min=1e-6)
            pred_dist_m = 1.0 / pred_inv

            # Denormalizza gt_dist allo stesso modo
            gt_inv = gt_dist * std_inv_dist + mean_inv_dist
            gt_inv = torch.clamp(gt_inv, min=1e-6)
            gt_dist_m = 1.0 / gt_inv

            # ACCDOA (B, Seq, 2)
            pred_cos = pred_accdoa[:, :, 0]
            pred_sin = pred_accdoa[:, :, 1]
            gt_cos   = gt_angle[:, :, 0]
            gt_sin   = gt_angle[:, :, 1]
            
            # Probabilità attività (norma del vettore)
            pred_active_prob = torch.norm(pred_accdoa, p=2, dim=-1)
            
            # Azzeriamo dist e angle se non c'è detection (prob < 0.5)
            idle_mask = pred_active_prob < 0.5
            pred_dist_m[idle_mask] = 0.0
            
            pred_angle_rad = torch.atan2(pred_sin, pred_cos)
            pred_angle_rad[idle_mask] = 0.0
            pred_angle_deg = torch.rad2deg(pred_angle_rad)
            
            target_angle_rad = torch.atan2(gt_sin, gt_cos)
            target_angle_deg = torch.rad2deg(target_angle_rad)
            
            all_pred_dist.extend(pred_dist_m.cpu().numpy().flatten())
            all_target_dist.extend(gt_dist_m.cpu().numpy().flatten())
            all_pred_angle.extend(pred_angle_deg.cpu().numpy().flatten())
            all_target_angle.extend(target_angle_deg.cpu().numpy().flatten())
            all_pred_active.extend(pred_active_prob.cpu().numpy().flatten())
            all_target_active.extend(gt_active.cpu().numpy().flatten())
            
    return (np.array(all_pred_dist), np.array(all_target_dist),
            np.array(all_pred_angle), np.array(all_target_angle),
            np.array(all_pred_active), np.array(all_target_active))

def main(args):
    # Carica parametri di normalizzazione
    params_path = os.path.join(args.data_root, 'preprocessing_params.json')
    with open(params_path) as f:
        preproc = json.load(f)
    mean_inv_dist = preproc['normalization']['mean_inv_dist']
    std_inv_dist  = preproc['normalization']['std_inv_dist']
    print(f"Normalization stats: mean_inv_dist={mean_inv_dist:.6f}, std_inv_dist={std_inv_dist:.6f}")

    _, _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        train_dir=os.path.join(args.data_root, 'train_split'),
        val_dir=os.path.join(args.data_root, 'val_split'),
        test_dir=os.path.join(args.data_root, 'test_split'),
        seq_len=args.seq_len
    )
    if args.rnn_type == 'gru':
        model = LiSANet(input_channels=8, gru_hidden_size=256, num_gru_layers=2).to(DEVICE)
    else:
        model = LiSALSTMNet(input_channels=8, lstm_hidden_size=256, num_lstm_layers=2).to(DEVICE)
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        
    p_dist, t_dist, p_angle, t_angle, p_active_prob, t_active = evaluate_model(
        model, test_loader, mean_inv_dist, std_inv_dist
    )
    
    if args.postprocess:
        processor = PostProcessor() # Instantiated without history_length or method
        p_dist_smooth, p_angle_smooth = [], []
        for d, a, prob in zip(p_dist, p_angle, p_active_prob):
            sd, sa = processor.update(d, a, is_active=(prob >= 0.5))
            p_dist_smooth.append(sd)
            p_angle_smooth.append(sa)
        p_dist, p_angle = np.array(p_dist_smooth), np.array(p_angle_smooth)
    
    # Maschera frame attivi per dist/angle
    active_mask = t_active.astype(bool)
    p_active_bin = (p_active_prob >= 0.5).astype(int)
    t_active_int = t_active.astype(int)

    # Detection metrics
    tp = int(((t_active_int == 1) & (p_active_bin == 1)).sum())
    fp = int(((t_active_int == 0) & (p_active_bin == 1)).sum())
    fn = int(((t_active_int == 1) & (p_active_bin == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float('nan')
    det_acc = float((t_active_int == p_active_bin).mean())

    # Dist/angle errors (active frames only)
    if active_mask.any():
        err_dist_abs, err_dist_pct, err_angle = compute_errors(
            p_dist[active_mask], t_dist[active_mask],
            p_angle[active_mask], t_angle[active_mask]
        )
    else:
        err_dist_abs = err_dist_pct = err_angle = np.array([])
    
    metrics = {
        "count": int(len(p_dist)),
        "active_frames": int(active_mask.sum()),
        "postprocess_info": {
            "enabled": args.postprocess,
            "method": "kalman" if args.postprocess else "none",
            "history": "none"
        },
        "detection": {
            "accuracy": det_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "angle": {
            "mae":    float(np.mean(err_angle))    if len(err_angle) else float('nan'),
            "median": float(np.median(err_angle))  if len(err_angle) else float('nan'),
            "rmse":   float(np.sqrt(np.mean(err_angle**2))) if len(err_angle) else float('nan')
        },
        "distance": {
            "mae":    float(np.mean(err_dist_abs))  if len(err_dist_abs) else float('nan'),
            "median": float(np.median(err_dist_abs)) if len(err_dist_abs) else float('nan'),
            "rmse":   float(np.sqrt(np.mean(err_dist_abs**2))) if len(err_dist_abs) else float('nan'),
            "mape":   float(np.mean(err_dist_pct)) if len(err_dist_pct) else float('nan')
        }
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Detection Acc:  {det_acc*100:.1f}%  |  Precision: {precision*100:.1f}%  |  Recall: {recall*100:.1f}%  |  F1: {f1*100:.1f}%")
    print(f"Angle MAE:      {metrics['angle']['mae']:.2f}°  (active frames only)")
    print(f"Dist MAE:       {metrics['distance']['mae']:.2f} m  (active frames only)")
        
    if active_mask.any():
        plot_error_by_distance(t_dist[active_mask], p_dist[active_mask],
                               t_angle[active_mask], p_angle[active_mask], args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to the trained model")
    parser.add_argument("--data_root", type=str, default="data", help="Root data directory (deve contenere train_split/, val_split/, test_split/ e preprocessing_params.json)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length in frames (e.g., 50 steps = 2.5s)")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save test outputs")

    # Post-processing options
    parser.add_argument("--postprocess", action='store_true', help="Enable Kalman filter post-processing")
    parser.add_argument("--rnn_type", type=str, default='gru', choices=['gru', 'lstm'], help="RNN architecture to use")
    args = parser.parse_args()
    main(args)