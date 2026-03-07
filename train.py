import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import math
import argparse
from tqdm import tqdm

from dataset import get_dataloaders
from model import LiSANet, LiSALSTMNet
from utils import MetricTracker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
DATA_DIR = 'data'

class MultiLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10.0, gamma=5.0):
        """Loss personalizzata che combina errore di distanza, ACCDOA e temporal smoothness.

        alpha: peso per la componente di distanza
        beta:  peso per la componente ACCDOA (angolo + attività combinati)
        gamma: peso per la componente di costanza/smoothness temporale
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.huber = nn.HuberLoss(reduction='none')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred_dist, target_dist, pred_accdoa, target_angle, target_active):
        # Maschera: distanza si calcola SOLO quando la sirena è attiva.
        mask = target_active.bool()  # (B, Seq)
        
        # 1. LOSS DISTANZA (Weighted MSE, solo su frame attivi)
        # Penalizza pesantemente errori a corto raggio, e perdona errori a lungo raggio.
        error_dist_sq = (pred_dist - target_dist) ** 2
        weight_dist = 1.0 / (target_dist + 5.0)  # Pesa di più quando vicino, salva dai NaN
        loss_dist_weighted = error_dist_sq * weight_dist
        
        if mask.any():
            loss_dist = loss_dist_weighted[mask].mean()
        else:
            loss_dist = loss_dist_weighted.mean() * 0.0  # zero gradient-friendly
        
        # 2. LOSS ACCDOA (MSE su tutti i frame)
        # target_accdoa = vector (sin, cos) * is_active
        target_active_expanded = target_active.unsqueeze(-1) # (B, Seq, 1)
        target_accdoa = target_angle * target_active_expanded
        
        loss_accdoa_raw = self.mse(pred_accdoa, target_accdoa) # (B, Seq, 2)
        loss_accdoa = loss_accdoa_raw.mean()
        
        # 3. LOSS TEMPORAL SMOOTHNESS (Solo sui frame consecutivi "attivi")
        # Identifica dove sia t che t-1 sono attivi per non penalizzare l'accensione
        mask_smooth = mask[:, 1:] & mask[:, :-1] # (B, Seq-1)
        
        if mask_smooth.any():
            # Derivata temporale ACCDOA
            diff_accdoa = pred_accdoa[:, 1:, :] - pred_accdoa[:, :-1, :]
            diff_accdoa_masked = diff_accdoa[mask_smooth]
            loss_smooth_accdoa = self.mse(diff_accdoa_masked, torch.zeros_like(diff_accdoa_masked)).mean()
            
            # Derivata temporale Distanza
            diff_dist = pred_dist[:, 1:] - pred_dist[:, :-1]
            diff_dist_masked = diff_dist[mask_smooth]
            loss_smooth_dist = self.huber(diff_dist_masked, torch.zeros_like(diff_dist_masked)).mean()
            
            loss_smooth = loss_smooth_accdoa + loss_smooth_dist
        else:
            loss_smooth = torch.tensor(0.0, device=pred_dist.device)
        
        total_loss = self.alpha * loss_dist + self.beta * loss_accdoa + self.gamma * loss_smooth
        
        # Restituiamo 4 valori (total, dist, accdoa, smooth)
        return total_loss, self.alpha * loss_dist, self.beta * loss_accdoa, self.gamma * loss_smooth
    
def compute_metrics(pred_dist, target_dist, pred_accdoa, target_angle, target_active):
    """
    Calcola metriche interpretabili per l'uomo (Metri, Gradi, Accuracy rilevamento).
    Dist e Angle vengono calcolati solo sui frame in cui la sirena è attiva (ground truth).
    """
    mask = target_active.bool()  # (B, Seq)

    # 1. Errore Distanza (MAE, solo su frame attivi)
def compute_metrics(pred_dist, pred_accdoa, target_dist, target_angle, target_active):
    """
    Calcola le metriche di valutazione sul batch corrente.
    Restituisce un dizionario con MAE Distanza, MAE Angolare, e Accuratezza Detection.
    """
    with torch.no_grad():
        pred_active_prob = torch.norm(pred_accdoa, p=2, dim=-1)
        pred_active = (pred_active_prob >= 0.5)

        pred_angle_rad = torch.atan2(pred_accdoa[:, :, 0], pred_accdoa[:, :, 1])
        target_angle_rad = torch.atan2(target_angle[:, :, 0], target_angle[:, :, 1])
        
        mask = target_active.bool()
        
        if mask.any():
            dist_mae = torch.abs(pred_dist[mask] - target_dist[mask]).mean().item()
            angle_diff = torch.abs(pred_angle_rad[mask] - target_angle_rad[mask])
            angle_diff = torch.min(angle_diff, 2 * math.pi - angle_diff)
            angle_mae = torch.rad2deg(angle_diff).mean().item()
        else:
            dist_mae = 0.0
            angle_mae = 0.0
            
        det_acc = (pred_active == target_active.bool()).to(torch.float32).mean().item() * 100
        
    return {
        'dist_mae': dist_mae,
        'angle_mae': angle_mae,
        'det_acc': det_acc
    }
    
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        specs      = batch['spectrogram'].to(DEVICE)  # (B, Seq, 8, F, T)
        gt_dist    = batch['gt_dist'].to(DEVICE)      # (B, Seq)
        gt_angle   = batch['gt_angle'].to(DEVICE)     # (B, Seq, 2) [sin, cos]
        gt_active  = batch['gt_active'].to(DEVICE)    # (B, Seq)
        mic_coords = batch['microphones'].to(DEVICE)  # (B, 4, 3)
        
        optimizer.zero_grad()
        
        # Forward Pass
        pred_dist, pred_accdoa, _ = model(specs, mic_coords, hidden_state=None)
        
        # Calcolo Loss
        loss, loss_dist, loss_accdoa, loss_smooth = criterion(
            pred_dist, gt_dist, pred_accdoa, gt_angle, gt_active
        )
        
        # Backward Pass
        loss.backward()
        
        # Gradient Clipping per la GRU
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Log
        metrics = compute_metrics(pred_dist, pred_accdoa, gt_dist, gt_angle, gt_active)
        
        tracker.update({
            'loss': loss.item(),
            'loss_dist': loss_dist.item(),
            'loss_accdoa': loss_accdoa.item(),
            'loss_smooth': loss_smooth.item(),
            **metrics
        })
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
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
            loss, loss_dist, loss_accdoa, loss_smooth = criterion(
                pred_dist, gt_dist, pred_accdoa, gt_angle, gt_active
            )
            
            # Metrics
            metrics = compute_metrics(
                pred_dist, pred_accdoa, gt_dist, gt_angle, gt_active
            )
            
            tracker.update({
                'loss': loss.item(),
                'loss_dist': loss_dist.item(),
                'loss_accdoa': loss_accdoa.item(),
                'loss_smooth': loss_smooth.item(),
                **metrics
            })
            
    return tracker.average()

def main(args):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"Starting Training with the following configuration:")
    print(f"Device: {DEVICE}")
    print(f"Seq Length: {args.seq_len}")
    print(f"Batch Size: {args.batch_size}")
    
    # Dataset
    print("Loading Datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        processed_dir=f"{DATA_DIR}/processed",
        seq_len=args.seq_len
    )
    
    # Model
    print(f"Initializing Model (RNN Type: {args.rnn_type.upper()})...")
    if args.rnn_type == 'gru':
        model = LiSANet(input_channels=8, gru_hidden_size=256, num_gru_layers=2).to(DEVICE)
    else:
        model = LiSALSTMNet(input_channels=8, lstm_hidden_size=256, num_lstm_layers=2).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = MultiLoss(alpha=1.0, beta=10.0, gamma=5.0)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )
    
    # Paths
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    last_model_path = os.path.join(CHECKPOINT_DIR, "last_model.pth")
    
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
        best_val_loss = checkpoint['loss']
    else:
        print("No checkpoint found. Starting training from scratch.")

    print("Starting Training Loop...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = validate(model, val_loader, criterion)
        
        print(f"TRAIN | Loss: {train_metrics['loss']:.4f} (dist={train_metrics['loss_dist']:.4f} "
              f"accdoa={train_metrics['loss_accdoa']:.4f} smooth={train_metrics['loss_smooth']:.4f}) | "
              f"Dist MAE: {train_metrics['dist_mae']:.2f}m | Angle MAE: {train_metrics['angle_mae']:.2f}° | "
              f"Det Acc: {train_metrics['det_acc']:.1f}%")
              
        print(f"VAL   | Loss: {val_metrics['loss']:.4f} (dist={val_metrics['loss_dist']:.4f} "
              f"accdoa={val_metrics['loss_accdoa']:.4f} smooth={val_metrics['loss_smooth']:.4f}) | "
              f"Dist MAE: {val_metrics['dist_mae']:.2f}m | Angle MAE: {val_metrics['angle_mae']:.2f}° | "
              f"Det Acc: {val_metrics['det_acc']:.1f}%")
        
        # Scheduling
        scheduler.step(val_metrics['loss'])
        
        # Checkpointing
        if val_metrics['loss'] < best_val_loss:
            print(f"--> New Best Model! (Loss: {best_val_loss:.4f} -> {val_metrics['loss']:.4f})")
            best_val_loss = val_metrics['loss']
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            print(f"No improvement. Early Stop Counter: {early_stop_counter+1}/{args.patience}")
            early_stop_counter += 1
            
        # Save Last
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_metrics['loss'],
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
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length in frames (e.g., 50 steps = 2.5s)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=11, help="Epoch patience for Early Stopping")
    parser.add_argument("--resume", action='store_true', help="Resume training from latest checkpoint")
    parser.add_argument("--rnn_type", type=str, default='gru', choices=['gru', 'lstm'], help="RNN architecture to use")
    
    args = parser.parse_args()
    main(args)