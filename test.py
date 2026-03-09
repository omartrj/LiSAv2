import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
from tqdm import tqdm
from scipy.stats import gaussian_kde
from dataset import get_dataloaders
from model import LiSANet, LiSALSTMNet
from postprocessing import PostProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Raggi (in metri) per le metriche F_i
FI_RADII = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]

def compute_errors(pred_dist, target_dist, pred_angle_deg, target_angle_deg):
    err_dist_abs = np.abs(pred_dist - target_dist)
    err_dist_pct = (err_dist_abs / (target_dist + 1e-6)) * 100
    
    diff = pred_angle_deg - target_angle_deg
    diff = (diff + 180) % 360 - 180
    err_angle = np.abs(diff)
    
    return err_dist_abs, err_dist_pct, err_angle


def compute_position_error(pred_dist, target_dist, pred_angle_deg, target_angle_deg):
    """Errore euclideo 2D in metri (coordinate polari → cartesiane)."""
    pred_rad = np.deg2rad(pred_angle_deg)
    tgt_rad  = np.deg2rad(target_angle_deg)
    pred_x = pred_dist * np.cos(pred_rad)
    pred_y = pred_dist * np.sin(pred_rad)
    gt_x   = target_dist * np.cos(tgt_rad)
    gt_y   = target_dist * np.sin(tgt_rad)
    return np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)


def compute_fi_metrics(pos_errors, radii=None):
    """F_i = frazione di frame attivi con errore posizionale ≤ i metri."""
    if radii is None:
        radii = FI_RADII
    if len(pos_errors) == 0:
        return {r: float('nan') for r in radii}
    return {r: float(np.mean(pos_errors <= r)) for r in radii}

def plot_error_by_distance(target_dist, pred_dist, target_angle, pred_angle, output_dir):
    err_dist  = np.abs(pred_dist - target_dist)
    pos_err   = compute_position_error(pred_dist, target_dist, pred_angle, target_angle)
    diff_angle = (pred_angle - target_angle + 180) % 360 - 180
    err_angle  = np.abs(diff_angle)

    bins = np.arange(0, np.max(target_dist) + 2, 2)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    dist_mae, angle_mae, pos_mae = [], [], []
    for i in range(len(bins) - 1):
        mask = (target_dist >= bins[i]) & (target_dist < bins[i + 1])
        if np.any(mask):
            dist_mae.append(np.mean(err_dist[mask]))
            angle_mae.append(np.mean(err_angle[mask]))
            pos_mae.append(np.mean(pos_err[mask]))
        else:
            dist_mae.append(np.nan)
            angle_mae.append(np.nan)
            pos_mae.append(np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, ylabel, title, color in zip(
        axes,
        [dist_mae, angle_mae, pos_mae],
        ["Distance MAE (m)", "Angle MAE (°)", "2D Position Error MAE (m)"],
        ["Distance Error vs True Distance", "Angle Error vs True Distance", "2D Error vs True Distance"],
        ["royalblue", "tomato", "mediumseagreen"]
    ):
        ax.plot(bin_centers, data, marker='o', linestyle='-', color=color)
        ax.set_xlabel("True Distance (m)"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_vs_distance.png"), dpi=150)
    plt.close()


def plot_fi_curve(pos_errors, output_dir, radii=None):
    """Curva F_i: CDF dell'errore posizionale 2D."""
    if radii is None:
        radii = FI_RADII
    max_r = max(radii) * 1.1
    r_vals  = np.linspace(0, max_r, 600)
    fi_vals = [np.mean(pos_errors <= r) for r in r_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_vals, fi_vals, 'steelblue', linewidth=2.5)

    colors = cm.viridis(np.linspace(0.1, 0.9, len(radii)))
    fi_at_radii = compute_fi_metrics(pos_errors, radii)
    for (r, f), c in zip(fi_at_radii.items(), colors):
        ax.axvline(r, color=c, linestyle='--', alpha=0.55, linewidth=1)
        ax.axhline(f, color=c, linestyle='--', alpha=0.55, linewidth=1)
        ax.scatter([r], [f], color=c, zorder=5, s=70,
                   label=f'F_{r:.1f}m = {f * 100:.1f}%')

    ax.set_xlabel("Raggio r (m)")
    ax.set_ylabel("F(r) — frazione frame corretti")
    ax.set_title("Curva di accuratezza posizionale (CDF dell'errore 2D)")
    ax.set_ylim(0, 1.05); ax.set_xlim(0)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fi_curve.png"), dpi=150)
    plt.close()


def plot_error_distributions(pos_errors, err_angle, err_dist, output_dir):
    """Istogrammi + KDE per i tre tipi di errore."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    datasets = [
        (pos_errors, axes[0], "Errore posizionale 2D (m)", "steelblue"),
        (err_angle,  axes[1], "Errore angolare (°)",       "tomato"),
        (err_dist,   axes[2], "Errore distanza (m)",       "mediumseagreen"),
    ]
    for data, ax, title, color in datasets:
        if len(data) == 0:
            continue
        data = data[np.isfinite(data)]
        ax.hist(data, bins=40, density=True, alpha=0.45, color=color, edgecolor='white')
        if len(data) > 2:
            kde  = gaussian_kde(data)
            clip = np.percentile(data, 99)
            xs   = np.linspace(0, clip, 300)
            ax.plot(xs, kde(xs), color=color, linewidth=2)
        ax.axvline(np.mean(data),   color='navy',       linestyle='--', linewidth=1.5,
                   label=f'Mean = {np.mean(data):.2f}')
        ax.axvline(np.median(data), color='darkorange',  linestyle='--', linewidth=1.5,
                   label=f'Median = {np.median(data):.2f}')
        ax.set_title(title); ax.set_ylabel("Densità")
        ax.grid(True, linestyle=':', alpha=0.4); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_distributions.png"), dpi=150)
    plt.close()


def plot_scatter_2d(pred_dist, target_dist, pred_angle_deg, target_angle_deg, pos_errors, output_dir):
    """Scatter: posizioni GT colorate per entità dell'errore + GT vs Pred."""
    tgt_rad  = np.deg2rad(target_angle_deg)
    pred_rad = np.deg2rad(pred_angle_deg)
    gt_x  = target_dist * np.cos(tgt_rad);  gt_y  = target_dist * np.sin(tgt_rad)
    px    = pred_dist  * np.cos(pred_rad);  py    = pred_dist  * np.sin(pred_rad)

    vmax = np.percentile(pos_errors, 95)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc = axes[0].scatter(gt_x, gt_y, c=pos_errors, cmap='RdYlGn_r',
                         s=8, alpha=0.6, vmin=0, vmax=vmax)
    plt.colorbar(sc, ax=axes[0], label='Errore posizionale (m)')
    axes[0].set_title('Posizioni GT — colorate per errore 2D')
    axes[0].set_xlabel('X [m]'); axes[0].set_ylabel('Y [m]')
    axes[0].axis('equal'); axes[0].grid(True, linestyle=':', alpha=0.4)

    axes[1].scatter(gt_x, gt_y, c='black', s=6, alpha=0.3, label='Ground Truth')
    axes[1].scatter(px,   py,   c='red',   s=6, alpha=0.3, label='Predetto')
    axes[1].set_title('GT vs Predetto (posizioni 2D)')
    axes[1].set_xlabel('X [m]'); axes[1].set_ylabel('Y [m]')
    axes[1].axis('equal'); axes[1].grid(True, linestyle=':', alpha=0.4); axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_2d.png"), dpi=150)
    plt.close()


def plot_detection_confusion(t_active_int, p_active_bin, output_dir):
    """Matrice di confusione per la detection della sirena."""
    tp = int(((t_active_int == 1) & (p_active_bin == 1)).sum())
    fp = int(((t_active_int == 0) & (p_active_bin == 1)).sum())
    fn = int(((t_active_int == 1) & (p_active_bin == 0)).sum())
    tn = int(((t_active_int == 0) & (p_active_bin == 0)).sum())
    conf = np.array([[tn, fp], [fn, tp]])
    total = conf.sum()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(conf, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Inattivo', 'Pred: Attivo'])
    ax.set_yticklabels(['GT: Inattivo', 'GT: Attivo'])
    ax.set_title('Matrice di Confusione — Detection')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{conf[i, j]}\n({100 * conf[i, j] / total:.1f}%)',
                    ha='center', va='center', fontsize=12,
                    color='white' if conf[i, j] > conf.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detection_confusion.png"), dpi=150)
    plt.close()


# Fasce di distanza (m) per l'analisi del degrado
DIST_BANDS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, np.inf)]


def compute_distance_band_metrics(target_dist, pred_dist, target_angle, pred_angle,
                                  pos_errors, bands=None, fi_radii=None):
    """Calcola metriche per fascia di distanza GT.

    Returns:
        list of dicts — uno per fascia, con chiavi: label, count, pos_mae,
        pos_median, pos_p90, angle_mae, dist_mae, fi (dict r->val)
    """
    if bands is None:
        bands = DIST_BANDS
    if fi_radii is None:
        fi_radii = [1.0, 2.0, 5.0]

    results = []
    for lo, hi in bands:
        label = f"{lo:.0f}-{hi:.0f}m" if hi != np.inf else f"{lo:.0f}m+"
        mask  = (target_dist >= lo) & (target_dist < hi)
        n     = int(mask.sum())
        if n == 0:
            results.append({"label": label, "count": 0})
            continue

        pe   = pos_errors[mask]
        ae   = np.abs(((pred_angle[mask] - target_angle[mask] + 180) % 360) - 180)
        de   = np.abs(pred_dist[mask] - target_dist[mask])

        results.append({
            "label":      label,
            "count":      n,
            "pos_mae":    float(np.mean(pe)),
            "pos_median": float(np.median(pe)),
            "pos_p75":    float(np.percentile(pe, 75)),
            "pos_p90":    float(np.percentile(pe, 90)),
            "angle_mae":  float(np.mean(ae)),
            "dist_mae":   float(np.mean(de)),
            "fi":         {r: float(np.mean(pe <= r)) for r in fi_radii},
        })
    return results


def print_distance_band_table(band_metrics):
    """Stampa a terminale la tabella per fascia di distanza."""
    header = f"{'Fascia':>10} {'N':>6} {'PosMAE':>8} {'P50':>8} {'P90':>8} "\
             f"{'AngMAE':>8} {'DistMAE':>8} {'F1m':>7} {'F2m':>7} {'F5m':>7}"
    print(header)
    print('─' * len(header))
    for b in band_metrics:
        if b['count'] == 0:
            print(f"  {b['label']:>8}  {'─':>6}")
            continue
        fi = b.get('fi', {})
        print(
            f"  {b['label']:>8}  {b['count']:>6d}  "
            f"{b['pos_mae']:>7.2f}m  {b['pos_median']:>7.2f}m  {b['pos_p90']:>7.2f}m  "
            f"{b['angle_mae']:>7.2f}°  {b['dist_mae']:>7.2f}m  "
            f"{fi.get(1.0, float('nan'))*100:>6.1f}%  "
            f"{fi.get(2.0, float('nan'))*100:>6.1f}%  "
            f"{fi.get(5.0, float('nan'))*100:>6.1f}%"
        )


def plot_degradation_by_distance(target_dist, pred_dist, target_angle, pred_angle,
                                  pos_errors, output_dir, bands=None):
    """Due subplot che mostrano come degradano le predizioni con la distanza:
    (a) F_i (r=1,2,5m) per fascia di distanza
    (b) Boxplot / ribbon di errore 2D per fascia di distanza
    """
    if bands is None:
        bands = DIST_BANDS

    fi_radii = [1.0, 2.0, 5.0]
    band_data = compute_distance_band_metrics(
        target_dist, pred_dist, target_angle, pred_angle, pos_errors,
        bands=bands, fi_radii=fi_radii
    )
    valid_bands = [b for b in band_data if b.get('count', 0) > 0]
    labels  = [b['label'] for b in valid_bands]
    x_pos   = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Subplot 1: F_i per fascia ────────────────────────────────────────────
    ax = axes[0]
    colors_fi = ['royalblue', 'darkorange', 'mediumseagreen']
    for r, c in zip(fi_radii, colors_fi):
        vals = [b['fi'][r] * 100 for b in valid_bands]
        ax.plot(x_pos, vals, marker='o', linewidth=2, color=c, label=f'F_{r:.0f}m')
        ax.fill_between(x_pos, vals, alpha=0.12, color=c)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Frazione frame corretti (%)")
    ax.set_xlabel("Fascia di distanza GT")
    ax.set_title("Degradazione F_i con la distanza")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    # ── Subplot 2: Percentili errore posizionale per fascia ──────────────────
    ax = axes[1]
    p50  = [b['pos_median'] for b in valid_bands]
    p75  = [b['pos_p75']    for b in valid_bands]
    p90  = [b['pos_p90']    for b in valid_bands]
    mae  = [b['pos_mae']    for b in valid_bands]

    ax.fill_between(x_pos, p75, p90, alpha=0.20, color='tomato',     label='P75–P90')
    ax.fill_between(x_pos, p50, p75, alpha=0.30, color='darkorange',  label='P50–P75')
    ax.plot(x_pos, p90, '--',  color='tomato',     linewidth=1.2)
    ax.plot(x_pos, p75, '--',  color='darkorange',  linewidth=1.2)
    ax.plot(x_pos, p50, '-',   color='navy',        linewidth=2,   label='Mediana (P50)')
    ax.plot(x_pos, mae, 's--', color='mediumseagreen', linewidth=1.5, markersize=6, label='MAE')
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Errore posizionale 2D (m)")
    ax.set_xlabel("Fascia di distanza GT")
    ax.set_title("Distribuzione errore 2D per fascia di distanza")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "degradation_by_distance.png"), dpi=150)
    plt.close()

    return band_data


def evaluate_model(model, loader, mean_inv_dist, std_inv_dist):
    model.eval()
    all_pred_dist, all_target_dist = [], []
    all_pred_angle, all_target_angle = [], []
    all_pred_active, all_target_active = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            specs      = batch['spectrogram'].to(DEVICE)
            gt_dist    = batch['gt_dist'].to(DEVICE)
            gt_angle   = batch['gt_angle'].to(DEVICE)   # (B, Seq, 2) [cos, sin]
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
            "mae":        float(np.mean(err_angle))                      if len(err_angle) else float('nan'),
            "median":     float(np.median(err_angle))                    if len(err_angle) else float('nan'),
            "rmse":       float(np.sqrt(np.mean(err_angle**2)))          if len(err_angle) else float('nan'),
            "within_5":   float(np.mean(err_angle <= 5.0)  * 100)       if len(err_angle) else float('nan'),
            "within_10":  float(np.mean(err_angle <= 10.0) * 100)       if len(err_angle) else float('nan'),
            "within_15":  float(np.mean(err_angle <= 15.0) * 100)       if len(err_angle) else float('nan'),
            "within_30":  float(np.mean(err_angle <= 30.0) * 100)       if len(err_angle) else float('nan'),
        },
        "distance": {
            "mae":    float(np.mean(err_dist_abs))   if len(err_dist_abs) else float('nan'),
            "median": float(np.median(err_dist_abs)) if len(err_dist_abs) else float('nan'),
            "rmse":   float(np.sqrt(np.mean(err_dist_abs**2))) if len(err_dist_abs) else float('nan'),
            "mape":   float(np.mean(err_dist_pct))  if len(err_dist_pct) else float('nan')
        }
    }

    # ── Metriche posizionali 2D ────────────────────────────────────────────────
    if active_mask.any():
        pos_errors = compute_position_error(
            p_dist[active_mask], t_dist[active_mask],
            p_angle[active_mask], t_angle[active_mask]
        )
        fi_metrics = compute_fi_metrics(pos_errors)
        metrics["position_2d"] = {
            "mae":    float(np.mean(pos_errors)),
            "median": float(np.median(pos_errors)),
            "rmse":   float(np.sqrt(np.mean(pos_errors**2))),
            "p90":    float(np.percentile(pos_errors, 90)),
        }
        metrics["fi"] = {f"F_{r}m": f"{v*100:.2f}%" for r, v in fi_metrics.items()}
    else:
        pos_errors = np.array([])
        fi_metrics = {r: float('nan') for r in FI_RADII}
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # ── Stampa riepilogo ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Detection       Acc={det_acc*100:.1f}%  Prec={precision*100:.1f}%  "
          f"Rec={recall*100:.1f}%  F1={f1*100:.1f}%")
    print(f"Angle MAE       {metrics['angle']['mae']:.2f}°  "
          f"(≤5°: {metrics['angle']['within_5']:.1f}%  "
          f"≤10°: {metrics['angle']['within_10']:.1f}%  "
          f"≤15°: {metrics['angle']['within_15']:.1f}%  "
          f"≤30°: {metrics['angle']['within_30']:.1f}%)")
    print(f"Distance MAE    {metrics['distance']['mae']:.2f} m")
    if active_mask.any():
        print(f"Position MAE    {metrics['position_2d']['mae']:.2f} m  "
              f"(median={metrics['position_2d']['median']:.2f} m  "
              f"P90={metrics['position_2d']['p90']:.2f} m)")
        print(f"F_i metrics     " + "  ".join(
            f"F_{r}m={v*100:.1f}%" for r, v in fi_metrics.items()
        ))
    print(f"{'─'*60}\n")

    # ── Plot + analisi per fascia ──────────────────────────────────────────────
    if active_mask.any():
        plot_error_by_distance(t_dist[active_mask], p_dist[active_mask],
                               t_angle[active_mask], p_angle[active_mask], args.output_dir)
        plot_fi_curve(pos_errors, args.output_dir)
        plot_error_distributions(pos_errors, err_angle, err_dist_abs, args.output_dir)
        plot_scatter_2d(p_dist[active_mask], t_dist[active_mask],
                        p_angle[active_mask], t_angle[active_mask],
                        pos_errors, args.output_dir)

        band_metrics = plot_degradation_by_distance(
            t_dist[active_mask], p_dist[active_mask],
            t_angle[active_mask], p_angle[active_mask],
            pos_errors, args.output_dir
        )
        metrics["by_distance_band"] = band_metrics
        # aggiorna il JSON con le metriche per fascia
        with open(os.path.join(args.output_dir, "test_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)

        print("\n── Metriche per fascia di distanza ──")
        print_distance_band_table(band_metrics)
        print()

    plot_detection_confusion(t_active_int, p_active_bin, args.output_dir)

    # ── Report testuale ────────────────────────────────────────────────────────
    import datetime
    report_path = os.path.join(args.output_dir, "report.txt")
    with open(report_path, 'w') as rpt:
        def w(line=""):
            rpt.write(line + "\n")

        w("=" * 65)
        w("  LiSAv2 — Test Report")
        w(f"  Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"  Model     : {args.model_path}")
        w(f"  Device    : {DEVICE}")
        w(f"  RNN type  : {args.rnn_type.upper()}")
        w(f"  Post-proc : {'Kalman' if args.postprocess else 'None'}")
        w("=" * 65)
        w()

        w("── Detection ───────────────────────────────────────────────────")
        w(f"  Total frames   : {metrics['count']}")
        w(f"  Active frames  : {metrics['active_frames']}")
        w(f"  Accuracy       : {metrics['detection']['accuracy']*100:.2f}%")
        w(f"  Precision      : {metrics['detection']['precision']*100:.2f}%")
        w(f"  Recall         : {metrics['detection']['recall']*100:.2f}%")
        w(f"  F1             : {metrics['detection']['f1']*100:.2f}%")
        w()

        w("── Angle Error (active frames) ────────────────────────────────")
        a = metrics['angle']
        w(f"  MAE            : {a['mae']:.3f}°")
        w(f"  Median         : {a['median']:.3f}°")
        w(f"  RMSE           : {a['rmse']:.3f}°")
        w(f"  Within  5°     : {a['within_5']:.1f}%")
        w(f"  Within 10°     : {a['within_10']:.1f}%")
        w(f"  Within 15°     : {a['within_15']:.1f}%")
        w(f"  Within 30°     : {a['within_30']:.1f}%")
        w()

        w("── Distance Error (active frames) ─────────────────────────────")
        d = metrics['distance']
        w(f"  MAE            : {d['mae']:.3f} m")
        w(f"  Median         : {d['median']:.3f} m")
        w(f"  RMSE           : {d['rmse']:.3f} m")
        w(f"  MAPE           : {d['mape']:.2f}%")
        w()

        if "position_2d" in metrics:
            p2 = metrics['position_2d']
            w("── 2D Position Error (active frames) ──────────────────────────")
            w(f"  MAE            : {p2['mae']:.3f} m")
            w(f"  Median         : {p2['median']:.3f} m")
            w(f"  RMSE           : {p2['rmse']:.3f} m")
            w(f"  P90            : {p2['p90']:.3f} m")
            w()

            w("── F_i Metrics (fraction of frames within radius r) ───────────")
            for key, val in metrics['fi'].items():
                w(f"  {key:<12} : {val}")
            w()

        if "by_distance_band" in metrics:
            w("── Metrics by Distance Band ────────────────────────────────────")
            hdr = (f"  {'Band':>8}  {'N':>6}  {'PosMAE':>8}  {'P50':>8}  "
                   f"{'P90':>8}  {'AngMAE':>8}  {'DistMAE':>8}  "
                   f"{'F1m':>7}  {'F2m':>7}  {'F5m':>7}")
            w(hdr)
            w("  " + "─" * (len(hdr) - 2))
            for b in metrics['by_distance_band']:
                if b.get('count', 0) == 0:
                    w(f"  {b['label']:>8}  {'—':>6}")
                    continue
                fi = b.get('fi', {})
                w(
                    f"  {b['label']:>8}  {b['count']:>6d}  "
                    f"{b['pos_mae']:>7.2f}m  {b['pos_median']:>7.2f}m  "
                    f"{b['pos_p90']:>7.2f}m  {b['angle_mae']:>7.2f}°  "
                    f"{b['dist_mae']:>7.2f}m  "
                    f"{fi.get(1.0, float('nan'))*100:>6.1f}%  "
                    f"{fi.get(2.0, float('nan'))*100:>6.1f}%  "
                    f"{fi.get(5.0, float('nan'))*100:>6.1f}%"
                )
            w()

        w("── Output Files ────────────────────────────────────────────────")
        for fname in ["test_metrics.json", "fi_curve.png", "error_vs_distance.png",
                      "error_distributions.png", "scatter_2d.png",
                      "degradation_by_distance.png", "detection_confusion.png"]:
            w(f"  {fname}")
        w("=" * 65)

    print(f"Report salvato in: {report_path}")
    print(f"Plot salvati in  : {args.output_dir}/")

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