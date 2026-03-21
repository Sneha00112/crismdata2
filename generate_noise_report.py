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
OUT_DIR  = os.path.join(BASE_DIR, "Analysis_Results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
def load_crism(img_path):
    LINES, SAMPLES, BANDS = 15, 64, 107
    raw = np.fromfile(img_path, dtype='<f4', count=LINES * BANDS * SAMPLES)
    cube = raw.reshape((LINES, BANDS, SAMPLES))
    cube = np.transpose(cube, (0, 2, 1))
    cube = np.where(np.isfinite(cube), cube, np.nan)
    cube[cube == 65535.0] = np.nan
    return cube

print(f"Loading CRISM dataset for noise analysis...")
cube = load_crism(IMG_PATH)
LINES, SAMPLES, BANDS = cube.shape

# ── 2. Quantify Noise Types ────────────────────────────────────────────────────

# A. Striping Noise (Column-wise detector non-uniformity)
col_mean = np.nanmean(cube, axis=0) # [Samples, Bands]
col_std_per_band = np.nanstd(col_mean, axis=0)
striping_metric = np.nanmean(col_std_per_band)

# B. Gaussian / Random Noise
# Residual = Cube - 3x3 Spatial Median Filtered Cube
temp_cube = np.nan_to_num(cube, nan=0.0)
noise_cube = cube - median_filter(temp_cube, size=(3, 3, 1))
gaussian_metric = np.nanstd(noise_cube)

# C. Spike / Impulse Noise (Outliers)
spike_threshold = 3.5 * gaussian_metric
spike_mask = np.abs(noise_cube) > spike_threshold
spike_metric = np.nansum(spike_mask) / cube.size # Fraction of pixels that are spikes

# D. Low-SNR Bands
band_mean = np.nanmean(cube, axis=(0, 1))
band_std  = np.nanstd(cube, axis=(0, 1))
snr = np.where(band_std > 1e-9, band_mean / band_std, 0.0)
low_snr_bands = np.sum(snr < 5.0)
low_snr_metric = low_snr_bands / BANDS # Fraction of bands with low SNR

# E. Atmospheric / Spectral Distortion
# Dips in mean spectrum detected by the std of the 2nd derivative
mean_spectrum = np.nanmean(cube, axis=(0, 1))
d2_spectrum = np.gradient(np.gradient(mean_spectrum))
atmo_metric = np.nanstd(d2_spectrum)

# ── 3. Normalize for Comparison ────────────────────────────────────────────────
# Raw metrics have different scales, so we normalize them relative to their impact
# These factors are estimated based on typical CRISM noise levels
raw_vals = [striping_metric, gaussian_metric, spike_metric, low_snr_metric, atmo_metric]
# Empirically adjusted normalization factors to bring metrics into a 0-1 range
factors = [1.0, 5.0, 100.0, 1.0, 10.0] 
norm_vals = [rv * f for rv, f in zip(raw_vals, factors)]
# Cap at 1.0 for visualization
norm_vals = [min(1.0, v) for v in norm_vals]

noise_types = ["Striping\nNoise", "Gaussian\nNoise", "Spike\nNoise", "Low-SNR\nBands", "Atmospheric\nFeatures"]
colors = ['#E63946', '#457B9D', '#F4A261', '#2A9D8F', '#8338EC']

# ── 4. Generate Bar Chart ───────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
bars = plt.bar(noise_types, norm_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
plt.title("CRISM Noise Type Severity Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Normalized Severity Index (0 - 1.0)", fontsize=12)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, "noise_severity_chart.png")
plt.savefig(chart_path, dpi=150)
print(f"Noise severity chart saved to {chart_path}")

# ── 5. Detailed Summary ────────────────────────────────────────────────────────
print("\nNoise Analysis Summary:")
for nt, rv, nv in zip(noise_types, raw_vals, norm_vals):
    print(f"  {nt.replace('\\n', ' '):20}: Raw={rv:8.4f}, Severity={nv:4.2f}")
