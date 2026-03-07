import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def create_trajectory_plot(df, gt_data, pred_dists, pred_angles, output_path, mic_coords=None, has_gt=True):
    """
    Crea e salva il plot spaziale e temporale con i risultati dell'inferenza.
    Compatibile sia con metriche offline (inference.py) che online (live_inference.py).
    """
    # Maschere attività
    pred_active = (df['pred_active_prob'].values >= 0.5) if 'pred_active_prob' in df.columns else np.ones(len(df), dtype=bool)

    # Convert Polar (Dist, Angle) to Cartesian (X, Y)
    pred_rad = np.deg2rad(df['pred_angle'].values)
    pred_x = df['pred_dist'].values * np.cos(pred_rad)
    pred_y = df['pred_dist'].values * np.sin(pred_rad)

    if has_gt:
        gt_active = df['gt_active'].astype(bool).values if 'gt_active' in df.columns else np.ones(len(df), dtype=bool)
        # gt_data[N, 4]: dist, sin, cos, active
        if gt_data is not None and gt_data.shape[1] == 4:
            gt_rad = np.arctan2(gt_data[:, 1], gt_data[:, 2])
            gt_dist = gt_data[:, 0]
        else:
            gt_rad = np.deg2rad(df['gt_angle'].values)
            gt_dist = df['gt_dist'].values
            
        gt_x = gt_dist * np.cos(gt_rad)
        gt_y = gt_dist * np.sin(gt_rad)

        # Calcola errori solo su frame attivi
        active_mask = gt_active
        if active_mask.any():
            mae_dist = np.mean(np.abs(gt_dist[active_mask] - df['pred_dist'].values[active_mask]))
            diff_rad = np.arctan2(np.sin(pred_rad[active_mask] - gt_rad[active_mask]),
                                  np.cos(pred_rad[active_mask] - gt_rad[active_mask]))
            mae_angle = np.mean(np.abs(np.degrees(diff_rad)))
        else:
            mae_dist = mae_angle = float('nan')

        pred_active_bin = (df['pred_active_prob'].values >= 0.5).astype(int)
        det_acc = np.mean(pred_active_bin == gt_active.astype(int)) * 100

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Spatial Trajectory
    ax = axes[0, 0]
    if has_gt:
        gt_x_inact = np.where(~gt_active, gt_x, np.nan)
        gt_y_inact = np.where(~gt_active, gt_y, np.nan)
        gt_x_act   = np.where(gt_active,  gt_x, np.nan)
        gt_y_act   = np.where(gt_active,  gt_y, np.nan)
        ax.plot(gt_x_inact, gt_y_inact, color='lightgrey', linewidth=2, label='GT (inactive)')
        ax.plot(gt_x_act,   gt_y_act,   'k--', linewidth=2, alpha=0.8, label='GT (active)')
        ax.set_title(f'Spatial Trajectory\n(MAE dist={mae_dist:.1f}m, angle={mae_angle:.1f}°, Det Acc={det_acc:.1f}%)')
    else:
        ax.set_title('Predicted Source Trajectory')

    pred_x_act = np.where(pred_active, pred_x, np.nan)
    pred_y_act = np.where(pred_active, pred_y, np.nan)
    ax.plot(pred_x_act, pred_y_act, 'r-', linewidth=1.5, alpha=0.8, label='Prediction (active)')
    
    if mic_coords is not None:
        if len(mic_coords.shape) == 3: mic_coords = mic_coords[0]
        ax.scatter(mic_coords[:, 0], mic_coords[:, 1], c='blue', marker='o', s=80, zorder=5, label='Microphones')
        
    ax.scatter(pred_x[0], pred_y[0], c='cyan', marker='o', label='Start')
    ax.scatter(pred_x[-1], pred_y[-1], c='black', marker='x', label='End')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.axis('equal'); ax.grid(True, linestyle=':', alpha=0.6); ax.legend()

    # 2. Distance over Time
    ax = axes[0, 1]
    if has_gt:
        gt_dist_inact = np.where(~gt_active, gt_dist, np.nan)
        gt_dist_act   = np.where(gt_active,  gt_dist, np.nan)
        ax.plot(df['time_s'], gt_dist_inact, color='lightgrey', linewidth=2, label='GT Dist (inactive)')
        ax.plot(df['time_s'], gt_dist_act,   'k--', linewidth=2, alpha=0.8, label='GT Dist (active)')
        
    pred_dist_act = np.where(pred_active, df['pred_dist'].values, np.nan)
    ax.plot(df['time_s'], pred_dist_act, 'r-', label='Pred Dist (active)')
    ax.set_ylabel('Distance [m]'); ax.set_title('Distance over Time')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend()

    # 3. Angle over Time
    ax = axes[1, 1]
    if has_gt:
        gt_angle_deg = np.rad2deg(gt_rad)
        gt_angle_inact = np.where(~gt_active, gt_angle_deg, np.nan)
        gt_angle_act   = np.where(gt_active,  gt_angle_deg, np.nan)
        ax.plot(df['time_s'], gt_angle_inact, color='lightgrey', linewidth=2, label='GT Angle (inactive)')
        ax.plot(df['time_s'], gt_angle_act,   'k--', linewidth=2, alpha=0.8, label='GT Angle (active)')
        
    pred_angle_act = np.where(pred_active, df['pred_angle'].values, np.nan)
    ax.plot(df['time_s'], pred_angle_act, 'r-', label='Pred Angle (active)')
    ax.set_ylabel('Angle [deg]'); ax.set_xlabel('Time [s]')
    ax.set_title('Angle over Time')
    ax.set_ylim(-180, 180); ax.grid(True, linestyle=':', alpha=0.6); ax.legend()

    # 4. Siren Detection
    ax = axes[1, 0]
    if has_gt:
        ax.step(df['time_s'], gt_active.astype(int), 'k--', where='post', label='GT Active', linewidth=2)
    if 'pred_active_prob' in df.columns:
        ax.plot(df['time_s'], df['pred_active_prob'], 'b-', label='Pred Prob', alpha=0.7)
        ax.axhline(0.5, color='orange', linestyle=':', label='Threshold 0.5')
    ax.set_ylabel('Active'); ax.set_xlabel('Time [s]')
    ax.set_title('Siren Detection')
    ax.set_ylim(-0.05, 1.05); ax.grid(True, linestyle=':', alpha=0.6); ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Trajectory plot saved to: {output_path}")

def save_statistics_report(df, output_path, has_gt=True, postprocess_info=None):
    """
    Calcola e salva le statistiche aggregate dell'inferenza in un report formattato.
    """
    if not has_gt:
        return
        
    stats_lines = []
    stats_lines.append("=" * 60)
    stats_lines.append("INFERENCE STATISTICS")
    stats_lines.append("=" * 60)

    if postprocess_info and postprocess_info.get('enabled'):
        stats_lines.append(f"Post-processing:     ENABLED")
        stats_lines.append(f"  Method:            {postprocess_info['method']}")
        stats_lines.append(f"  History:           {postprocess_info['history']}")
        stats_lines.append("-" * 60)

    gt_active = df['gt_active'].astype(bool).values if 'gt_active' in df.columns else np.ones(len(df), dtype=bool)
    
    stats_lines.append(f"Total Frames:        {len(df)}")
    stats_lines.append(f"Duration:            {df['time_s'].iloc[-1]:.2f} seconds")
    stats_lines.append(f"Active Frames (GT):  {int(gt_active.sum())} ({gt_active.mean() * 100:.1f}%)")
    
    if 'latency_ms' in df.columns:
        stats_lines.append(f"Avg Latency:         {df['latency_ms'].mean():.1f} ms")
        stats_lines.append(f"Max Latency:         {df['latency_ms'].max():.1f} ms")
    stats_lines.append("-" * 60)

    pred_active = (df['pred_active_prob'] >= 0.5).astype(int) if 'pred_active_prob' in df.columns else np.ones(len(df), dtype=int)
    tp = int(((gt_active == 1) & (pred_active == 1)).sum())
    fp = int(((gt_active == 0) & (pred_active == 1)).sum())
    fn = int(((gt_active == 1) & (pred_active == 0)).sum())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float('nan')
    det_acc = (gt_active == pred_active).mean() * 100

    stats_lines.append(f"Detection Accuracy:  {det_acc:.1f}%")
    stats_lines.append(f"Detection Precision: {precision * 100:.1f}%")
    stats_lines.append(f"Detection Recall:    {recall * 100:.1f}%")
    stats_lines.append(f"Detection F1:        {f1 * 100:.1f}%")
    stats_lines.append("-" * 60)

    active_mask = gt_active
    if active_mask.any():
        if 'error_dist' in df.columns:
            err_dist_col = df['error_dist'].values[active_mask]
            err_angle_col = df['error_angle'].values[active_mask]
        else:
            err_dist_col = np.abs(df['gt_dist'].values[active_mask] - df['pred_dist'].values[active_mask])
            pred_rad = np.deg2rad(df['pred_angle'].values[active_mask])
            gt_rad = np.deg2rad(df['gt_angle'].values[active_mask])
            diff_rad = np.arctan2(np.sin(pred_rad - gt_rad), np.cos(pred_rad - gt_rad))
            err_angle_col = np.abs(np.degrees(diff_rad))

        stats_lines.append(f"Distance MAE:        {err_dist_col.mean():.2f} m  (active frames only)")
        stats_lines.append(f"Distance Median:     {np.median(err_dist_col):.2f} m")
        stats_lines.append(f"Distance RMSE:       {np.sqrt(np.mean(err_dist_col**2)):.2f} m")
        stats_lines.append("-" * 60)
        stats_lines.append(f"Angle MAE:           {err_angle_col.mean():.2f}°  (active frames only)")
        stats_lines.append(f"Angle Median:        {np.median(err_angle_col):.2f}°")
        stats_lines.append(f"Angle RMSE:          {np.sqrt(np.mean(err_angle_col**2)):.2f}°")
        stats_lines.append(f"Angle Acc <10°:      {(err_angle_col < 10).mean() * 100:.1f}%")
        stats_lines.append(f"Angle Acc <20°:      {(err_angle_col < 20).mean() * 100:.1f}%")
    else:
        stats_lines.append("No active frames in this sequence.")
    
    stats_lines.append("=" * 60)

    with open(output_path, 'w') as f:
        f.write('\n'.join(stats_lines))
    print(f"Statistics saved to: {output_path}")

class MetricTracker:
    """Utility class to track and average metrics during training/validation loops."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.sum = {}
        self.count = 0
        
    def update(self, metrics_dict, n=1):
        self.count += n
        for k, v in metrics_dict.items():
            self.sum[k] = self.sum.get(k, 0) + v * n
            
    def average(self):
        return {k: v / self.count for k, v in self.sum.items()}
