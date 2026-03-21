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

print(f"Loading CRISM dataset for final visualizations...")
cube = load_crism(IMG_PATH)
LINES, SAMPLES, BANDS = cube.shape
wavelengths = np.linspace(0.362, 1.053, BANDS)

# ── 2. Raw vs Noisy Spectral Plot (Phase 1) ───────────────────────────────────
plt.figure(figsize=(12, 6))
pixel_coords = [(7, 32), (3, 15), (12, 50)] # Some random pixels
colors = ['r', 'g', 'b']

for (r, c), color in zip(pixel_coords, colors):
    spectrum = cube[r, c, :]
    smooth = median_filter(spectrum, size=5)
    plt.plot(wavelengths, spectrum, color=color, alpha=0.3, label=f'Raw Pixel ({r},{c})')
    plt.plot(wavelengths, smooth, color=color, lw=1.5, label=f'Smooth Pixel ({r},{c})')

plt.title("Spectral Quality Analysis: Raw vs. Denoised Spectra (CRISM)", fontsize=14, fontweight='bold')
plt.xlabel("Wavelength (µm)")
plt.ylabel("Radiance")
plt.legend(ncol=3, fontsize=9)
plt.grid(True, alpha=0.3)
spec_comp_path = os.path.join(OUT_DIR, "spectral_noise_comparison.png")
plt.savefig(spec_comp_path, dpi=150)
print(f"Spectral comparison plot saved to {spec_comp_path}")

# ── 3. Categorical Noise Bar Chart (Phase 2) ──────────────────────────────────
# Quantify metrics again for grouping
col_mean = np.nanmean(cube, axis=0)
striping = np.nanmean(np.nanstd(col_mean, axis=0)) * 1.0
gaussian = np.nanstd(cube - median_filter(np.nan_to_num(cube, nan=0.0), size=(3, 3, 1))) * 5.0
spike    = (np.nansum(np.abs(cube - median_filter(np.nan_to_num(cube, nan=0.0), size=(3, 3, 1))) > 3.0) / cube.size) * 50.0
snr_edge = (np.sum(np.nanmean(cube, axis=(0, 1)) / (np.nanstd(cube, axis=(0, 1)) + 1e-6) < 5.0) / BANDS) * 2.0
atmo     = np.nanstd(np.gradient(np.gradient(np.nanmean(cube, axis=(0, 1))))) * 10.0

categories = {
    "Structured": [("Striping", striping)],
    "Random": [("Gaussian / Thermal", gaussian)],
    "Severe": [("Impulse Spikes", spike), ("Low-SNR Edges", snr_edge)],
    "Environmental": [("Atmo Absorption", atmo)]
}

labels = []
values = []
groups = []
cat_colors = {'Structured': '#E63946', 'Random': '#457B9D', 'Severe': '#F4A261', 'Environmental': '#8338EC'}
bar_colors = []

for cat, items in categories.items():
    for label, val in items:
        labels.append(label)
        values.append(min(1.0, val))
        groups.append(cat)
        bar_colors.append(cat_colors[cat])

plt.figure(figsize=(12, 7))
x = np.arange(len(labels))
bars = plt.bar(x, values, color=bar_colors, alpha=0.9, edgecolor='black')
plt.xticks(x, labels, rotation=15, ha='right')
plt.title("Characterized Noise Severities by Category (CRISM)", fontsize=14, fontweight='bold')
plt.ylabel("Relative Severity (0 - 1.0)")
plt.ylim(0, 1.2)

# Legend for categories
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, edgecolor='black', label=cat) for cat, color in cat_colors.items()]
plt.legend(handles=legend_elements, title="Noise Categories", loc='upper right')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
cat_chart_path = os.path.join(OUT_DIR, "categorical_noise_severity.png")
plt.savefig(cat_chart_path, dpi=150)
print(f"Categorical noise chart saved to {cat_chart_path}")
