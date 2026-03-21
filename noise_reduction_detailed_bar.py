import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/CRISM20"
IMG_PATH = os.path.join(BASE_DIR, "frt0001073b_01_ra156s_trr3.img")
CORRECTED_PATH = os.path.join(BASE_DIR, "Physics_Correction_Results", "crism_physics_corrected.npy")
OUT_DIR  = os.path.join(BASE_DIR, "Physics_Correction_Results")

# ── 1. Load Data ───────────────────────────────────────────────────────────────
def load_raw(img_path):
    LINES, SAMPLES, BANDS = 15, 64, 107
    raw = np.fromfile(img_path, dtype='<f4', count=LINES * BANDS * SAMPLES)
    cube = raw.reshape((LINES, BANDS, SAMPLES))
    cube = np.transpose(cube, (0, 2, 1))
    cube = np.where(np.isfinite(cube), cube, np.nan)
    cube[cube == 65535.0] = np.nan
    return cube

cube_raw = load_raw(IMG_PATH)
cube_corr = np.load(CORRECTED_PATH)

# ── 2. Metric Calculation Function ─────────────────────────────────────────────
def get_noise_metrics(cube):
    BANDS = cube.shape[2]
    c = np.nan_to_num(cube, nan=0.0)
    
    # Striping
    col_mean = np.nanmean(cube, axis=0)
    striping = np.nanmean(np.nanstd(col_mean, axis=0))
    
    # Gaussian
    noise_resid = cube - median_filter(c, size=(3, 3, 1))
    gaussian = np.nanstd(noise_resid)
    
    # Spikes (Fraction)
    spike_thr = 3.5 * gaussian
    spikes = np.nansum(np.abs(noise_resid) > spike_thr) / cube.size
    
    # Atmospheric
    mean_spec = np.nanmean(cube, axis=(0, 1))
    atmo = np.nanstd(np.gradient(np.gradient(mean_spec)))
    
    # Low-SNR (Fraction)
    b_mean = np.nanmean(cube, axis=(0, 1))
    b_std = np.nanstd(cube, axis=(0, 1))
    snr = np.where(b_std > 1e-9, b_mean / b_std, 0.0)
    low_snr = np.sum(snr < 5.0) / BANDS
    
    return [striping, gaussian, spikes, atmo, low_snr]

# ── 3. Generate Chart ──────────────────────────────────────────────────────────
metrics_raw = get_noise_metrics(cube_raw)
metrics_corr = get_noise_metrics(cube_corr)

noise_labels = ["Striping\nNoise", "Gaussian\nNoise", "Spike\nNoise", "Atmospheric\nFeatures", "Low-SNR\nBands"]
x = np.arange(len(noise_labels))
width = 0.35

# Normalize for display (percentage of raw)
norm_raw = [1.0] * len(metrics_raw)
norm_corr = [c / (r + 1e-12) for r, c in zip(metrics_raw, metrics_corr)]

plt.figure(figsize=(12, 7))
plt.bar(x - width/2, norm_raw, width, label='Before (Raw)', color='#E63946', alpha=0.9, edgecolor='white')
plt.bar(x + width/2, norm_corr, width, label='After Physics Denoising', color='#2A9D8F', alpha=0.9, edgecolor='white')

plt.title("Detailed Noise Reduction: Before vs After Physical Denoising", fontsize=15, fontweight='bold')
plt.ylabel("Relative Severity (Normalized to Raw)", fontsize=12)
plt.xticks(x, noise_labels, fontsize=10)
plt.ylim(0, 1.2)
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add improvement percentage labels
for i in range(len(noise_labels)):
    reduction = (1.0 - norm_corr[i]) * 100
    plt.text(x[i], max(1.0, norm_corr[i]) + 0.05, f"−{reduction:.1f}%", 
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1D4E3F')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "detailed_noise_reduction_chart.png")
plt.savefig(out_path, dpi=150)
print(f"Detailed noise reduction chart saved to {out_path}")
