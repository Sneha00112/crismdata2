import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    # Dimensions from .lbl.txt: Lines=15, Samples=64, Bands=107
    LINES, SAMPLES, BANDS = 15, 64, 107
    # Data is LINE_INTERLEAVED (BIL): [line, band, sample]
    # Each sample is PC_REAL (float32, 4 bytes)
    # Total size = 15 * 107 * 64 * 4 = 410,880 bytes
    # The file size is 411,136 bytes. 
    # Based on PDS BIL, each record is 256 bytes (64 samples * 4 bytes).
    # There are 1606 records. Records 1 to 1605 are the image data.
    # Record 1606 is the rownum table.
    
    raw = np.fromfile(img_path, dtype='<f4', count=LINES * BANDS * SAMPLES)
    cube = raw.reshape((LINES, BANDS, SAMPLES))
    # Transpose to [Lines, Samples, Bands] for easier processing
    cube = np.transpose(cube, (0, 2, 1))
    # Replace invalid values (NaNs, infs) and fill value 65535.0
    cube = np.where(np.isfinite(cube), cube, np.nan)
    cube[cube == 65535.0] = np.nan
    return cube

print(f"Loading CRISM dataset from {IMG_PATH}...")
cube = load_crism(IMG_PATH)
LINES, SAMPLES, BANDS = cube.shape
print(f"  Shape (L x S x B): {LINES} x {SAMPLES} x {BANDS}")

# Approximate VNIR wavelengths for sensor-S (0.362–1.053 µm, 107 bands)
wavelengths = np.linspace(0.362, 1.053, BANDS)

# ── 2. Basic Statistics ────────────────────────────────────────────────────────
valid       = cube[np.isfinite(cube)]
global_min  = float(np.nanmin(cube))
global_max  = float(np.nanmax(cube))
global_mean = float(np.nanmean(cube))
global_std  = float(np.nanstd(cube))
nan_frac    = float(np.isnan(cube).sum() / cube.size)

band_mean = np.nanmean(cube, axis=(0, 1))
band_std  = np.nanstd(cube, axis=(0, 1))
band_snr  = np.where(band_std > 1e-9, band_mean / band_std, 0.0)

col_mean  = np.nanmean(cube, axis=0)   # [samples, bands]
row_mean  = np.nanmean(cube, axis=1)   # [lines,   bands]

print(f"  Global min={global_min:.6f}  max={global_max:.6f}  mean={global_mean:.6f}")
print(f"  NaN fraction: {nan_frac:.4%}")
print(f"  Band SNR range: {band_snr.min():.2f} – {band_snr.max():.2f}")

# ── Helper: save plot ──────────────────────────────────────────────────────────
def savefig(name):
    p = os.path.join(OUT_DIR, name)
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved: {name}")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION 1 – Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 10))
fig.suptitle("CRISM Dataset Analysis - frt0001073b", fontsize=16, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Band 50 image
ax = fig.add_subplot(gs[0, 0])
ax.imshow(cube[:, :, 50], cmap='gray', aspect='auto')
ax.set_title(f"Band 50 ({wavelengths[50]:.3f} µm)")
ax.set_xlabel("Samples"); ax.set_ylabel("Lines")

# Band 100 image (VNIR edge)
ax = fig.add_subplot(gs[0, 1])
ax.imshow(cube[:, :, 100], cmap='gray', aspect='auto')
ax.set_title(f"Band 100 ({wavelengths[100]:.3f} µm)")
ax.set_xlabel("Samples"); ax.set_ylabel("Lines")

# Mean spectrum
ax = fig.add_subplot(gs[0, 2])
ax.plot(wavelengths, band_mean, color='blue', lw=1.5)
ax.fill_between(wavelengths, band_mean-band_std, band_mean+band_std, alpha=0.2, color='blue')
ax.set_title("Mean Spectral Profile ± 1σ")
ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Radiance")

# SNR plot
ax = fig.add_subplot(gs[1, 0])
ax.plot(wavelengths, band_snr, color='green', lw=1.5)
ax.set_title("Per-Band SNR")
ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("SNR")

# Histogram
ax = fig.add_subplot(gs[1, 1])
ax.hist(valid, bins=50, color='purple', alpha=0.7)
ax.set_title("Radiance Histogram")
ax.set_xlabel("Radiance"); ax.set_ylabel("Frequency")
ax.set_yscale('log')

# Column-mean (striping)
ax = fig.add_subplot(gs[1, 2])
ax.imshow(col_mean.T, cmap='viridis', aspect='auto')
ax.set_title("Column-Mean per Band (Striping)")
ax.set_xlabel("Sample"); ax.set_ylabel("Band index")

savefig("data_analysis_summary.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Noise Metric Calculation
# ══════════════════════════════════════════════════════════════════════════════
# Striping noise metric: std of column means per band
col_var = np.nanstd(col_mean, axis=0)
striping_metric = float(np.mean(col_var))

# Gaussian noise via difference from 3x3 spatial median
# Handle NaNs in noise calculation
temp_cube = np.nan_to_num(cube, nan=0.0)
noise_cube = cube - median_filter(temp_cube, size=(3, 3, 1))
gaussian_metric = float(np.nanstd(noise_cube))

# Spike noise (outliers > 3.5 sigma)
spike_threshold = 3.5 * np.nanstd(noise_cube)
spike_count = int(np.sum(np.abs(noise_cube) > spike_threshold))

# ══════════════════════════════════════════════════════════════════════════════
# 4. Generate Report
# ══════════════════════════════════════════════════════════════════════════════
report_path = os.path.join(OUT_DIR, "Analysis_Report.md")
with open(report_path, 'w') as f:
    f.write("# CRISM Data Analysis Metrics Summary\n\n")
    f.write(f"**Dataset:** `frt0001073b_01_ra156s_trr3`\n\n")
    
    f.write("## 1. Dimensionality Summary\n\n")
    f.write(f"- **Number of Bands:** {BANDS}\n")
    f.write(f"- **Spectral Range:** {wavelengths[0]:.4f} to {wavelengths[-1]:.4f} µm (estimated)\n")
    f.write(f"- **Spatial Dimensions:** {LINES} lines x {SAMPLES} samples\n")
    f.write(f"- **Total Pixels:** {LINES * SAMPLES}\n\n")
    
    f.write("## 2. Statistical Summary\n\n")
    f.write(f"- **Global Min Radiance:** {global_min:.6f}\n")
    f.write(f"- **Global Max Radiance:** {global_max:.6f}\n")
    f.write(f"- **Global Mean Radiance:** {global_mean:.6f}\n")
    f.write(f"- **Global Std Dev:** {global_std:.6f}\n")
    f.write(f"- **NaN Fraction:** {nan_frac:.4%}\n\n")
    
    f.write("## 3. Quality & Noise Metrics\n\n")
    f.write(f"- **Average Band SNR:** {np.nanmean(band_snr):.2f}\n")
    f.write(f"- **Min Band SNR:** {band_snr.min():.2f}\n")
    f.write(f"- **Max Band SNR:** {band_snr.max():.2f}\n")
    f.write(f"- **Striping Metric (Col Std):** {striping_metric:.6f}\n")
    f.write(f"- **Gaussian Noise Metric:** {gaussian_metric:.6f}\n")
    f.write(f"- **Spike Noise Count:** {spike_count}\n\n")
    
    f.write("## 4. Visualizations\n\n")
    f.write("- **`data_analysis_summary.png`**: Comprehensive dashboard showing images, spectrum, and SNR.\n")

print(f"Analysis complete. Report: {report_path}")
