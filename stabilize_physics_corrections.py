import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/CRISM20"
IMG_PATH = os.path.join(BASE_DIR, "frt0001073b_01_ra156s_trr3.img")
OUT_DIR  = os.path.join(BASE_DIR, "Stabilized_Physics_Results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
def load_crism(img_path):
    LINES, SAMPLES, BANDS = 15, 64, 107
    raw = np.fromfile(img_path, dtype='<f4', count=LINES * BANDS * SAMPLES)
    cube = raw.reshape((LINES, BANDS, SAMPLES))
    cube = np.transpose(cube, (0, 2, 1)) # [L, S, B]
    cube = np.where(np.isfinite(cube), cube, np.nan)
    cube[cube == 65535.0] = np.nan
    return cube

print(f"Loading CRISM dataset for stabilized physical corrections...")
cube_raw = load_crism(IMG_PATH)
LINES, SAMPLES, BANDS = cube_raw.shape
WL = np.linspace(0.362, 1.053, BANDS)

# ── 2. Metric Functions ────────────────────────────────────────────────────────
def compute_metrics(cube, cube_ref=None):
    c = np.nan_to_num(cube, nan=0.0)
    
    # Spike Noise (Deviation from spatial median)
    med = median_filter(c, size=(3, 3, 1))
    diff = np.abs(c - med)
    spike_val = float(np.nanstd(diff))
    
    # Striping Noise
    col_mean = np.nanmean(cube, axis=0)
    strip_val = float(np.nanmean(np.nanstd(col_mean, axis=0)))
    
    # Atmospheric / Spectral Jitter
    mean_spec = np.nanmean(cube, axis=(0, 1))
    atmo_val = float(np.nanstd(np.gradient(np.gradient(mean_spec))))
    
    # Gaussian Noise
    gauss_val = float(np.nanstd(c - median_filter(c, size=(3, 3, 1))))
    
    # Low-SNR Bands (Fraction with SNR < 5)
    b_mean = np.nanmean(cube, axis=(0, 1))
    b_std = np.nanstd(cube, axis=(0, 1)) + 1e-12
    snr_array = b_mean / b_std
    low_snr_val = float(np.sum(snr_array < 5.0) / BANDS)
    
    m = {
        'Spike Noise': spike_val,
        'Atmospheric Features': atmo_val,
        'Gaussian Noise': gauss_val,
        'Striping Noise': strip_val,
        'Low-SNR Bands': low_snr_val,
        'Avg SNR': float(np.nanmean(snr_array))
    }
    return m

# ── 3. Stabilization Pipeline ──────────────────────────────────────────────────
cube_work = cube_raw.copy()
initial_metrics = compute_metrics(cube_work)

print("\n--- Applying Sequential Stabilization ---")

# Step 1: Spike Removal (First, to prevent propagating errors into means)
print("[1] Spike Removal (3.5 sigma Clipping)")
med = median_filter(np.nan_to_num(cube_work, nan=0.0), size=(3, 3, 1))
diff = np.abs(cube_work - med)
thr = 3.5 * np.nanstd(diff)
cube_work = np.where(diff > thr, med, cube_work)

# Step 2: Destriping
print("[2] Column-wise Destriping")
col_mean = np.nanmean(cube_work, axis=0, keepdims=True)
scene_mean = np.nanmean(cube_work)
col_correction = col_mean / (scene_mean + 1e-12)
# Stabilize correction to prevent over-amplification
col_correction = np.clip(col_correction, 0.5, 2.0)
cube_work = cube_work / col_correction

# Step 3: Stabilized Log Residual Normalization
print("[3] Stabilized Log Residual Normalization")
# Formula: X / (pixel_mean * band_mean / total_mean)
# Log space: log(X) - mean_p(log(X)) - mean_b(log(X)) + mean_total(log(X))
eps = 1e-8
# We work in linear space for better stability with small values
pixel_mean = np.nanmean(cube_work, axis=2, keepdims=True) # [L, S, 1]
band_mean  = np.nanmean(cube_work, axis=(0, 1), keepdims=True) # [1, 1, B]
total_mean = np.nanmean(cube_work)

denominator = ((pixel_mean + eps) * (band_mean + eps)) / (total_mean + eps)
# Clip denominator to prevent division by near-zero
denominator = np.clip(denominator, 1e-6, np.inf)

cube_work = cube_work / denominator

# Final Clipping to prevent noise exaggeration
q01, q99 = np.nanpercentile(cube_raw, [0.1, 99.9])
# Scale to raw range roughly
cube_work = np.clip(cube_work, 0, 2.0) # Normalized range [0, 2] is common for Log Residual

# Step 4: Selective Savitzky-Golay Smoothing
print("[4] Selective Savitzky-Golay Smoothing (Window 5)")
# Only apply to bands with low SNR if possible, or apply mildly to all
# Here we apply window=5, order=2 as requested.
cube_work = savgol_filter(cube_work, window_length=5, polyorder=2, axis=2)

final_metrics = compute_metrics(cube_work)

# ── 4. Verify & Compare ───────────────────────────────────────────────────────
print("\n--- Metric Comparison ---")
noise_types = ['Striping Noise', 'Gaussian Noise', 'Spike Noise', 'Low-SNR Bands', 'Atmospheric Features']
all_labels = noise_types + ['Avg SNR']
for l in all_labels:
    before = initial_metrics[l]
    after = final_metrics[l]
    change = (after - before) / (before + 1e-12) * 100
    if l == 'Avg SNR':
        status = "PASSED (Improved)" if after > before else "FAILED (Decreased)"
    else:
        status = "PASSED (Reduced)" if after < before else "FAILED (Increased)"
    print(f"{l:22s}: Raw={before:.4f} -> Corr={after:.6f} ({change:+.1f}%) | {status}")

# ── 5. Generate Visualizations ────────────────────────────────────────────────
# A. Noise Level Bar Chart (Focus on 5 Noise Types)
plt.figure(figsize=(12, 7))
x = np.arange(len(noise_types))
w = 0.35
# Normalize for display
raw_vals = [initial_metrics[l] for l in noise_types]
corr_vals = [final_metrics[l] for l in noise_types]

# Normalize by raw for better comparison
norm_raw = [1.0] * len(noise_types)
norm_corr = [c / (r + 1e-12) for r, c in zip(raw_vals, corr_vals)]

plt.bar(x - w/2, norm_raw, w, label='Before (Raw)', color='#E63946', alpha=0.8, edgecolor='white')
plt.bar(x + w/2, norm_corr, w, label='After Stabilized Preprocessing', color='#2A9D8F', alpha=0.8, edgecolor='white')
plt.xticks(x, [l.replace(' ', '\n') for l in noise_types])
plt.ylabel("Relative Magnitude (Raw=1.0)", fontsize=12)
plt.title("Detailed Noise Reduction: Before vs After Stabilized Physical Preprocessing", fontsize=14, fontweight='bold')
plt.ylim(0, 1.25)
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Add reduction % labels
for i in range(len(noise_types)):
    reduction = (1.0 - norm_corr[i]) * 100
    plt.text(x[i], max(1.0, norm_corr[i]) + 0.05, f"−{reduction:.1f}%", 
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1D4E3F')
plt.savefig(os.path.join(OUT_DIR, "noise_stability_comparison.png"), dpi=150)

# B. Spectral Plot
plt.figure(figsize=(10, 6))
p = (7, 32)
# Rescale corrected to match raw mean for visual comparison of shape
cube_rescaled = cube_work * (np.nanmean(cube_raw) / np.nanmean(cube_work))
plt.plot(WL, cube_raw[p[0], p[1], :], 'r', alpha=0.3, label='Raw Spectrum')
plt.plot(WL, cube_rescaled[p[0], p[1], :], 'b', lw=1.5, label='Stabilized Spectrum')
plt.title(f"Spectral Stability Comparison (Pixel {p})")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Radiance / Normalized")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "spectral_stability.png"), dpi=150)

# C. Spatial Map (Band 50)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(cube_raw[:, :, 50], cmap='magma', aspect='auto')
axes[0].set_title("Raw Spatial Map (Band 50)")
axes[1].imshow(cube_work[:, :, 50], cmap='magma', aspect='auto')
axes[1].set_title("Stabilized Spatial Map (Band 50)")
plt.savefig(os.path.join(OUT_DIR, "spatial_stability.png"), dpi=150)

# Save corrected cube
np.save(os.path.join(OUT_DIR, "crism_stabilized_physics.npy"), cube_work.astype(np.float32))

print(f"\nStabilization complete. Results in {OUT_DIR}")
