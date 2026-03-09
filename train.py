import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
import os
import math
import argparse
import json
from tqdm import tqdm

from dataset import get_dataloaders
from model import LiSANet, LiSALSTMNet
from utils import MetricTracker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WeightedMultiLoss(nn.Module):
    def __init__(self, w_dist=1.0, w_accdoa=5.0, w_smooth=0.5, use_smooth_loss=False):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.huber = nn.HuberLoss(reduction='none')
        self.w_dist = w_dist
        self.w_accdoa = w_accdoa
        self.w_smooth = w_smooth
        self.use_smooth_loss = use_smooth_loss

    def forward(self, pred_dist, target_dist, pred_accdoa, target_angle, target_active):
        # Maschera: distanza si calcola SOLO quando la sirena è attiva
        mask = target_active.bool()
        
        # 1. DISTANCE LOSS (MSE su inv_dist normalizzata)
        if mask.any():
            loss_dist = self.mse(pred_dist[mask], target_dist[mask]).mean()
        else:
            loss_dist = torch.tensor(0.0, device=pred_dist.device)

        # 2. ACCDOA LOSS
        target_active_expanded = target_active.unsqueeze(-1)
        target_accdoa = target_angle * target_active_expanded
        loss_accdoa = self.mse(pred_accdoa, target_accdoa).mean()
        
        # 3. TEMPORAL SMOOTHNESS LOSS
        loss_smooth = torch.tensor(0.0, device=pred_dist.device)
        mask_smooth = mask[:, 1:] & mask[:, :-1]
        
        if self.use_smooth_loss and mask_smooth.any():
            # Derivata temporale ACCDOA
            diff_accdoa = pred_accdoa[:, 1:, :] - pred_accdoa[:, :-1, :]
            loss_smooth_accdoa = self.mse(diff_accdoa[mask_smooth], torch.zeros_like(diff_accdoa[mask_smooth])).mean()
            
            # Derivata temporale Distanza
            diff_dist = pred_dist[:, 1:] - pred_dist[:, :-1]
            loss_smooth_dist = self.huber(diff_dist[mask_smooth], torch.zeros_like(diff_dist[mask_smooth])).mean()
            
            loss_smooth = loss_smooth_accdoa + loss_smooth_dist

        total_loss = self.w_dist * loss_dist + self.w_accdoa * loss_accdoa + self.w_smooth * loss_smooth
        return {
            "loss":        total_loss,
            "loss_dist":   loss_dist.detach(),
            "loss_accdoa": loss_accdoa.detach(),
            "loss_smooth": loss_smooth.detach(),
        }
    
class NewWeightedMultiLoss(nn.Module):
    def __init__(self, w_dist=1.0, w_accdoa=5.0, w_smooth=0.5, use_smooth_loss=False):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.huber = nn.HuberLoss(reduction='none')
        self.w_dist = w_dist
        self.w_accdoa = w_accdoa
        self.w_smooth = w_smooth
        self.use_smooth_loss = use_smooth_loss

    def forward(self, pred_dist, target_dist, pred_accdoa, target_angle, target_active):
        mask = target_active.bool()
        
        # 1. DISTANCE LOSS (Masked)
        # Scale: Z-score (mean 0, std 1)
        if mask.any():
            loss_dist = self.mse(pred_dist[mask], target_dist[mask]).mean()
        else:
            loss_dist = torch.tensor(0.0, device=pred_dist.device)

        # 2. ACCDOA - LOCALIZATION LOSS (Masked)
        # Scale: Cartesian coordinates [-1, 1]
        # This aligns perfectly with your Angle MAE metric
        if mask.any():
            loss_accdoa_loc = self.mse(pred_accdoa[mask], target_angle[mask]).mean()
        else:
            loss_accdoa_loc = torch.tensor(0.0, device=pred_dist.device)

        # 3. ACCDOA - DETECTION LOSS (Unmasked)
        # This ensures the model learns when the source is silent
        target_active_expanded = target_active.unsqueeze(-1)
        target_accdoa_det = target_angle * target_active_expanded
        loss_accdoa_det = self.mse(pred_accdoa, target_accdoa_det).mean()
        
        # 4. TEMPORAL SMOOTHNESS LOSS
        loss_smooth = torch.tensor(0.0, device=pred_dist.device)
        if self.use_smooth_loss:
            mask_smooth = mask[:, 1:] & mask[:, :-1]
            if mask_smooth.any():
                diff_accdoa = pred_accdoa[:, 1:, :] - pred_accdoa[:, :-1, :]
                loss_smooth_accdoa = self.mse(diff_accdoa[mask_smooth], torch.zeros_like(diff_accdoa[mask_smooth])).mean()
                diff_dist = pred_dist[:, 1:] - pred_dist[:, :-1]
                loss_smooth_dist = self.huber(diff_dist[mask_smooth], torch.zeros_like(diff_dist[mask_smooth])).mean()
                loss_smooth = loss_smooth_accdoa + loss_smooth_dist

        # Combined Loss
        # Localization task: loss_dist + loss_accdoa_loc
        # Detection task: loss_accdoa_det
        total_loss = (self.w_dist * loss_dist) + \
                     (self.w_accdoa * (loss_accdoa_loc + loss_accdoa_det)) + \
                     (self.w_smooth * loss_smooth)

        return {
            "loss":        total_loss,
            "loss_dist":   loss_dist.detach(),
            "loss_accdoa": (loss_accdoa_loc + loss_accdoa_det).detach(),
            "loss_smooth": loss_smooth.detach(),
        }

    
def compute_metrics(pred_dist, pred_accdoa, target_dist, target_angle, target_active,
                    mean_inv_dist=None, std_inv_dist=None):
    with torch.no_grad():
        pred_active_prob = torch.norm(pred_accdoa, p=2, dim=-1)
        pred_active = (pred_active_prob >= 0.5).cpu().numpy().flatten()
        target_active_np = target_active.cpu().numpy().flatten()

        res = {}

        # Detection F1-Score
        res['f1_det'] = f1_score(target_active_np, pred_active, zero_division=0) * 100.0
        
        mask = target_active.bool()
        
        if mask.any():
            # Denormalizzazione di inv_dist → distanza reale in metri
            if mean_inv_dist is not None and std_inv_dist is not None:
                p_inv = pred_dist[mask] * std_inv_dist + mean_inv_dist
                t_inv = target_dist[mask] * std_inv_dist + mean_inv_dist
                # Clamp per evitare divisioni per zero in caso di predizioni molto lontane dalla media
                p_inv = torch.clamp(p_inv, min=1e-6)
                t_inv = torch.clamp(t_inv, min=1e-6)
                p_dist_m = 1.0 / p_inv
                t_dist_m = 1.0 / t_inv
            else:
                p_dist_m = pred_dist[mask]
                t_dist_m = target_dist[mask]
            errors_dist = torch.abs(p_dist_m - t_dist_m)

            # Distanza (in metri)
            res['dist_mape'] = (errors_dist / t_dist_m).mean().item() * 100
            res['dist_mae'] = errors_dist.mean().item()
            res['dist_rmse'] = torch.sqrt((errors_dist ** 2).mean()).item()

            # Angolo
            pred_angle_rad = torch.atan2(pred_accdoa[:, :, 1], pred_accdoa[:, :, 0])
            target_angle_rad = torch.atan2(target_angle[:, :, 1], target_angle[:, :, 0])
            angle_diff = torch.abs(pred_angle_rad[mask] - target_angle_rad[mask])
            angle_diff = torch.min(angle_diff, 2 * math.pi - angle_diff)
            angle_deg = torch.rad2deg(angle_diff)
            res['angle_mae'] = angle_deg.mean().item()
            res['angle_mse'] = (angle_deg ** 2).mean().item()
            res['angle_acc_10deg'] = (angle_deg < 10.0).to(torch.float32).mean().item() * 100
            res['angle_acc_15deg'] = (angle_deg < 15.0).to(torch.float32).mean().item() * 100
        else:
            res = {
                'f1_det': 0.0,
                'dist_mape': 0.0,
                'dist_mae': 0.0,
                'dist_rmse': 0.0,
                'angle_mae': 0.0,
                'angle_mse': 0.0,
                'angle_acc_10deg': 0.0,
                'angle_acc_15deg': 0.0
            }

    return res

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        specs      = batch['spectrogram'].to(DEVICE)  # (B, Seq, 8, F, T)
        gt_dist    = batch['gt_dist'].to(DEVICE)      # (B, Seq)
        gt_angle   = batch['gt_angle'].to(DEVICE)     # (B, Seq, 2) [cos, sin]
        gt_active  = batch['gt_active'].to(DEVICE)    # (B, Seq)
        mic_coords = batch['microphones'].to(DEVICE)  # (B, 4, 3)
        
        optimizer.zero_grad()
        
        # Forward Pass
        pred_dist, pred_accdoa, _ = model(specs, mic_coords, hidden_state=None)
        
        # Calcolo Loss
        out = criterion(
            pred_dist, gt_dist, pred_accdoa, gt_angle, gt_active
        )
        
        # Backward Pass
        out["loss"].backward()
        
        # Gradient Clipping per la GRU/LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Log
        metrics = compute_metrics(pred_dist, pred_accdoa, gt_dist, gt_angle, gt_active,
                                  MEAN_INV_DIST, STD_INV_DIST)
        
        tracker.update({
            'loss': out["loss"].item(),
            'loss_dist': out["loss_dist"].item(),
            'loss_accdoa': out["loss_accdoa"].item(),
            'loss_smooth': out["loss_smooth"].item(),
            **metrics
        })
        
        pbar.set_postfix({'Loss': f'{out["loss"].item():.4f}'})
        
    return tracker.average()

def validate(model, loader, criterion):
    model.eval()
    tracker = MetricTracker()
    
    with torch.no_grad():
        for batch in loader:
            specs      = batch['spectrogram'].to(DEVICE)
            gt_dist    = batch['gt_dist'].to(DEVICE)
            gt_angle   = batch['gt_angle'].to(DEVICE)
            gt_active  = batch['gt_active'].to(DEVICE)
            mic_coords = batch['microphones'].to(DEVICE)
            
            # Forward
            pred_dist, pred_accdoa, _ = model(specs, mic_coords, hidden_state=None)
            
            # Loss
            out = criterion(
                pred_dist, gt_dist, pred_accdoa, gt_angle, gt_active
            )
            
            # Metrics
            metrics = compute_metrics(
                pred_dist, pred_accdoa, gt_dist, gt_angle, gt_active,
                MEAN_INV_DIST, STD_INV_DIST
            )
            
            tracker.update({
                'loss': out["loss"].item(),
                'loss_dist': out["loss_dist"].item(),
                'loss_accdoa': out["loss_accdoa"].item(),
                'loss_smooth': out["loss_smooth"].item(),
                **metrics
            })
            
    return tracker.average()

def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"Starting Training with the following configuration:")
    print(f"Device: {DEVICE}")
    print(f"Seq Length: {args.seq_len}")
    print(f"Batch Size: {args.batch_size}")
    
    # Carica parametri di preprocessing (media/std per denormalizzazione)
    global MEAN_INV_DIST, STD_INV_DIST
    preproc_path = os.path.join(args.data_root, 'preprocessing_params.json')
    with open(preproc_path) as f:
        preproc = json.load(f)
    MEAN_INV_DIST = preproc['normalization']['mean_inv_dist']
    STD_INV_DIST  = preproc['normalization']['std_inv_dist']
    print(f"Normalization stats loaded: mean_inv_dist={MEAN_INV_DIST:.4f}, std_inv_dist={STD_INV_DIST:.4f}")

    # Dataset
    print("Loading Datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        train_dir=os.path.join(args.data_root, 'train_split'),
        val_dir=os.path.join(args.data_root, 'val_split'),
        test_dir=os.path.join(args.data_root, 'test_split'),
        seq_len=args.seq_len
    )
    
    # Model
    print(f"Initializing Model (RNN Type: {args.rnn_type.upper()})...")
    if args.rnn_type == 'gru':
        model = LiSANet(input_channels=8, gru_hidden_size=256, num_gru_layers=2).to(DEVICE)
    else:
        model = LiSALSTMNet(input_channels=8, lstm_hidden_size=256, num_lstm_layers=2).to(DEVICE)

    #criterion = WeightedMultiLoss(use_smooth_loss=args.smooth, w_smooth=args.w_smooth)
    criterion = NewWeightedMultiLoss(use_smooth_loss=args.smooth, w_smooth=args.w_smooth)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Reduce on Plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )
    
    # Paths
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    last_model_path = os.path.join(args.checkpoint_dir, "last_model.pth")
    
    best_val_loss = float('inf')
    early_stop_counter = 0

    start_epoch = 0

    # Resume from checkpoint if requested
    if args.resume and os.path.exists(last_model_path):
        print(f"Resuming training from checkpoint: {last_model_path}")
        checkpoint = torch.load(last_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
    elif args.resume:
        print(f"No checkpoint found at {last_model_path}. Starting fresh training.")

    print("Starting Training Loop...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = validate(model, val_loader, criterion)
        
        smooth_str = f", L_smooth: {train_metrics['loss_smooth']:.4f}" if args.smooth else ""
        print(f"TRAIN | Loss: {train_metrics['loss']:.4f} (L_dist: {train_metrics['loss_dist']:.4f}, L_accdoa: {train_metrics['loss_accdoa']:.4f}{smooth_str} ) | "
              f"Dist MAPE: {train_metrics['dist_mape']:.2f}% | Dist RMSE: {train_metrics['dist_rmse']:.2f}m | "
              f"Angle MAE: {train_metrics['angle_mae']:.2f}° | Angle Acc@15°: {train_metrics['angle_acc_15deg']:.1f}% | "
              f"F1 Det: {train_metrics['f1_det']:.2f}%")

        smooth_str = f", L_smooth: {val_metrics['loss_smooth']:.4f}" if args.smooth else ""
        print(f"VAL   | Loss: {val_metrics['loss']:.4f} (L_dist: {val_metrics['loss_dist']:.4f}, L_accdoa: {val_metrics['loss_accdoa']:.4f}{smooth_str} ) | "
              f"Dist MAPE: {val_metrics['dist_mape']:.2f}% | Dist RMSE: {val_metrics['dist_rmse']:.2f}m | "
              f"Angle MAE: {val_metrics['angle_mae']:.2f}° | Angle Acc@15°: {val_metrics['angle_acc_15deg']:.1f}% | "
              f"F1 Det: {val_metrics['f1_det']:.2f}%")
        
        scheduler.step(val_metrics['loss'])

        # Checkpointing
        val_loss = val_metrics['loss']
        if val_loss < best_val_loss:
            print(f"--> New Best Model! (Loss: {best_val_loss:.4f} -> {val_loss:.4f})")
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
                'loss': val_loss
            }, best_model_path)
        else:
            early_stop_counter += 1
            print(f"No improvement. Early Stop Counter: {early_stop_counter}/{args.patience}")
            
        # Save Last
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'loss': val_metrics['loss']
        }, last_model_path)
        
        # Early Stopping
        if early_stop_counter >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs of no improvement.")
            break
            
    print("\nTraining Finished.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length in frames (e.g., 50 steps = 2.5s)")
    parser.add_argument("--lr", type=float, default=0.0025, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Epoch patience for Early Stopping")
    parser.add_argument("--resume", action='store_true', help="Resume training from latest checkpoint")
    parser.add_argument("--rnn_type", type=str, default='gru', choices=['gru', 'lstm'], help="RNN architecture to use")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--data_root", type=str, default="data", help="Root data directory (must contain train_split/, val_split/, test_split/ and preprocessing_params.json)")
    parser.add_argument("--smooth", action='store_true', help="Enable temporal smoothness loss")
    parser.add_argument("--w_smooth", type=float, default=0.5, help="Weight for smoothness loss (usato solo se --smooth è attivo)")
    
    args = parser.parse_args()
    main(args)