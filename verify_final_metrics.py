import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# Ensure no NaNs for vector math
def clean(cube):
    return np.nan_to_num(cube, nan=0.0)

r = clean(cube_raw)
p = clean(cube_phys)
m = clean(cube_ml)

# ── 2. Metric Calculations ─────────────────────────────────────────────────────

# A. Avg SNR (Mean / Std)
def get_snr(cube):
    b_mean = np.nanmean(cube, axis=(0, 1))
    b_std = np.nanstd(cube, axis=(0, 1)) + 1e-12
    return float(np.nanmean(b_mean / b_std))

snr_raw = get_snr(cube_raw)
snr_phys = get_snr(cube_phys)
snr_ml = get_snr(cube_ml)

# B. Spectral Angle Mapper (SAM) - Average angle in radians between spectra
def get_sam(cube1, cube2):
    # Flatten spatial
    v1 = cube1.reshape(-1, cube1.shape[2])
    v2 = cube2.reshape(-1, cube2.shape[2])
    
    # Normalize vectors
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-12
    
    dot = np.sum((v1/norm1) * (v2/norm2), axis=1)
    # Clip for arccos stability
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(dot)
    return float(np.nanmean(angle))

sam_phys = get_sam(r, p)
sam_ml = get_sam(r, m)

# C. Feature Preservation (Correlation)
def get_corr(cube1, cube2):
    v1 = cube1.flatten()
    v2 = cube2.flatten()
    return float(np.corrcoef(v1, v2)[0, 1])

corr_phys = get_corr(r, p)
corr_ml = get_corr(r, m)

# ── 3. Reporting ──────────────────────────────────────────────────────────────
print("\n--- FINAL QUANTITATIVE VERIFICATION ---")
print(f"{'Metric':25s} | {'Raw':10s} | {'Physics':10s} | {'ML (Final)':10s}")
print("-" * 65)
print(f"{'Avg SNR (signal/noise)':25s} | {snr_raw:10.2f} | {snr_phys:10.2f} | {snr_ml:10.2f}")
print(f"{'Spectral Angle (SAM, rad)':25s} | {'0.00':10s} | {sam_phys:10.4f} | {sam_ml:10.4f}")
print(f"{'Feature Correlation':25s} | {'1.000':10s} | {corr_phys:10.4f} | {corr_ml:10.4f}")

# ── 4. Dashboard Visualization ────────────────────────────────────────────────
labels = ['Avg SNR', 'SAM Distortion (min is better)', 'Feature Preservation']
metrics_raw  = [snr_raw, 0.0, 1.0]
metrics_ml   = [snr_ml, sam_ml, corr_ml]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Subplot 1: SNR Improvement
axes[0].bar(['Raw', 'ML Final'], [snr_raw, snr_ml], color=['#E63946', '#8338EC'])
axes[0].set_title("SNR Improvement (Higher is Better)")
axes[0].set_ylabel("Avg SNR")
improve = ((snr_ml - snr_raw) / snr_raw) * 100
axes[0].text(0.5, snr_ml/2, f"+{improve:.1f}%", ha='center', color='white', fontweight='bold')

# Subplot 2: Spectral Angle (SAM)
axes[1].bar(['Physics', 'ML Final'], [sam_phys, sam_ml], color=['#2A9D8F', '#8338EC'])
axes[1].set_title("Spectral Angle Distortion (Lower is Better)")
axes[1].set_ylabel("SAM (Radians)")

# Subplot 3: Feature Preservation
axes[2].bar(['Physics', 'ML Final'], [corr_phys, corr_ml], color=['#2A9D8F', '#8338EC'])
axes[2].set_title("Feature Correlation (Higher is Better)")
axes[2].set_ylabel("Correlation Coefficient")
axes[2].set_ylim(0.8, 1.05)

plt.suptitle("Final HYBRID Pipeline Quantitative Assessment", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "final_metrics_dashboard.png"), dpi=150)
print(f"\nFinal metrics dashboard saved to {os.path.join(OUT_DIR, 'final_metrics_dashboard.png')}")
