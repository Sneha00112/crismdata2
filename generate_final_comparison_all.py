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
PHYS_PATH = os.path.join(BASE_DIR, "Stabilized_Physics_Results", "crism_stabilized_physics.npy")
ML_PATH = os.path.join(BASE_DIR, "ML_Denoising_Results", "crism_ml_denoised.npy")
OUT_DIR  = os.path.join(BASE_DIR, "Final_Results")
os.makedirs(OUT_DIR, exist_ok=True)

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
cube_phys = np.load(PHYS_PATH)
cube_ml = np.load(ML_PATH)

# ── 2. Comprehensive Metric Calculation ────────────────────────────────────────
def get_noise_metrics(cube):
    BANDS = cube.shape[2]
    c = np.nan_to_num(cube, nan=0.0)
    
    # Striping
    col_mean = np.nanmean(cube, axis=0)
    striping = np.nanmean(np.nanstd(col_mean, axis=0))
    
    # Gaussian
    noise_resid = cube - median_filter(c, size=(3, 3, 1))
    gaussian = np.nanstd(noise_resid)
    
    # Spike (Fraction)
    spike_thr = 3.5 * np.nanstd(cube - median_filter(c, size=(3, 3, 1)))
    spikes = np.nansum(np.abs(cube - median_filter(c, size=(3, 3, 1))) > spike_thr) / cube.size
    
    # Atmospheric
    mean_spec = np.nanmean(cube, axis=(0, 1))
    atmo = np.nanstd(np.gradient(np.gradient(mean_spec)))
    
    # Low-SNR (Fraction)
    b_mean = np.nanmean(cube, axis=(0, 1))
    b_std = np.nanstd(cube, axis=(0, 1)) + 1e-12
    snr_array = b_mean / b_std
    low_snr = np.sum(snr_array < 5.0) / BANDS
    
    return [striping, gaussian, spikes, atmo, low_snr]

m_raw = get_noise_metrics(cube_raw)
m_phys = get_noise_metrics(cube_phys)
m_ml = get_noise_metrics(cube_ml)

# SNR calculation
def get_snr(cube):
    b_mean = np.nanmean(cube, axis=(0, 1))
    b_std = np.nanstd(cube, axis=(0, 1)) + 1e-12
    return np.nanmean(b_mean / b_std)

snrs = [get_snr(cube_raw), get_snr(cube_phys), get_snr(cube_ml)]

# ── 3. Final Comparison Plot ──────────────────────────────────────────────────
labels = ["Striping\nNoise", "Gaussian\nNoise", "Spike\nNoise", "Atmospheric\nFeatures", "Low-SNR\nBands"]
x = np.arange(len(labels))
width = 0.25

# Normalize to Raw = 1.0
norm_raw = [1.0] * 5
norm_phys = [p / (r + 1e-12) for r, p in zip(m_raw, m_phys)]
norm_ml = [m / (r + 1e-12) for r, m in zip(m_raw, m_ml)]

plt.figure(figsize=(14, 8))
plt.bar(x - width, norm_raw, width, label='Raw (Baseline)', color='#E63946', alpha=0.8, edgecolor='black')
plt.bar(x, norm_phys, width, label='After Physics Correction', color='#2A9D8F', alpha=0.8, edgecolor='black')
plt.bar(x + width, norm_ml, width, label='After Final ML Denoising', color='#8338EC', alpha=0.9, edgecolor='black')

plt.title("Comprehensive CRISM Noise Reduction: 3-Stage Comparison", fontsize=15, fontweight='bold')
plt.ylabel("Normalized Noise Severity (Raw = 1.0)", fontsize=12)
plt.xticks(x, labels, fontsize=11)
plt.ylim(0, 1.3)
plt.legend(fontsize=11, loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Annotate improvements
for i in range(len(labels)):
    phys_imp = (1.0 - norm_phys[i]) * 100
    ml_imp = (1.0 - norm_ml[i]) * 100
    plt.text(x[i], norm_ml[i] + 0.05, f"Total: -{ml_imp:.1f}%", ha='center', fontweight='bold', color='#432371')

plt.annotate(f"SNR Trend:\nRaw: {snrs[0]:.1f}\nPhys: {snrs[1]:.1f}\nML:   {snrs[2]:.1f}", 
             xy=(0.02, 0.8), xycoords='axes fraction', bbox=dict(boxstyle='round', fc='white', alpha=0.8), fontsize=10)

plt.tight_layout()
final_plot_path = os.path.join(OUT_DIR, "final_3stage_comparison.png")
plt.savefig(final_plot_path, dpi=150)
print(f"Final 3-stage comparison plot saved to {final_plot_path}")
