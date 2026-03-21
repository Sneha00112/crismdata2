import os
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_IMG_PATH = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/CRISM20/frt0001073b_01_ra156s_trr3.img"
HYBRID_CUBE_PATH = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/CRISM20/ML_Denoising_Results/crism_ml_denoised.npy"

# ── 1. Load Data ───────────────────────────────────────────────────────────────
def load_raw_crism(img_path):
    # Dimensions from .lbl.txt: Lines=15, Samples=64, Bands=107
    LINES, SAMPLES, BANDS = 15, 64, 107
    raw = np.fromfile(img_path, dtype='<f4', count=LINES * BANDS * SAMPLES)
    cube = raw.reshape((LINES, BANDS, SAMPLES))
    cube = np.transpose(cube, (0, 2, 1))
    cube = np.where(np.isfinite(cube), cube, np.nan)
    cube[cube == 65535.0] = np.nan
    return cube

def calculate_denoising_improvement(raw_cube, hybrid_cube, region_coords=None):
    """
    Calculates the percentage reduction in spectral variance.
    """
    # Handle NaNs by filling with zeros or interpolating for variance check
    # Variance of NaNs is problematic
    raw_cube = np.nan_to_num(raw_cube, nan=0.0)
    hybrid_cube = np.nan_to_num(hybrid_cube, nan=0.0)
    
    if region_coords:
        y1, y2, x1, x2 = region_coords
        raw_sample = raw_cube[y1:y2, x1:x2, :]
        hybrid_sample = hybrid_cube[y1:y2, x1:x2, :]
    else:
        raw_sample = raw_cube
        hybrid_sample = hybrid_cube

    # 2. Calculate Variance across the spectral dimension
    var_raw = np.var(raw_sample, axis=2).mean()
    var_hybrid = np.var(hybrid_sample, axis=2).mean()

    # 3. Calculate Percentage Reduction
    reduction = ((var_raw - var_hybrid) / var_raw) * 100

    # 4. Calculate Signal-to-Noise Ratio (SNR) Improvement
    snr_gain_db = 10 * np.log10(var_raw / (var_hybrid + 1e-12))

    print("-" * 30)
    print(f"Spectral Variance (Raw):    {var_raw:.6f}")
    print(f"Spectral Variance (Hybrid): {var_hybrid:.6f}")
    print("-" * 30)
    print(f"TOTAL NOISE REDUCTION:      {reduction:.2f}%")
    print(f"APPROXIMATE SNR GAIN:       {snr_gain_db:.2f} dB")
    print("-" * 30)

    return reduction

print("Loading raw and hybrid cubes...")
cube_raw = load_raw_crism(RAW_IMG_PATH)
cube_hybrid = np.load(HYBRID_CUBE_PATH)

print("\n[Global Analysis]")
calculate_denoising_improvement(cube_raw, cube_hybrid)

print("\n[Homogeneous Region Analysis (Center 5x5 patch)]")
calculate_denoising_improvement(cube_raw, cube_hybrid, region_coords=(5, 10, 30, 35))
