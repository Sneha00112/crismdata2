import os
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
DENOISED_CUBE_PATH = "/Users/snehasr/Downloads/crismdata2-main/crism_ml_denoised.npy"
OUT_DIR = "/Users/snehasr/Downloads/crismdata2-main/Mineral_Features"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
cube = np.load(DENOISED_CUBE_PATH)
LINES, SAMPLES, BANDS = cube.shape
WL = np.linspace(0.362, 1.053, BANDS)

# ── 2. Continuum Removal (Convex Hull) ─────────────────────────────────────────
def remove_continuum(spectrum, wavelengths):
    # Mask NaNs
    valid = ~np.isnan(spectrum)
    if np.sum(valid) < 3:
        return np.full_like(spectrum, np.nan), np.full_like(spectrum, np.nan)
    
    s = spectrum[valid]
    w = wavelengths[valid]
    
    # Points for Convex Hull
    points = np.vstack((w, s)).T
    # Add base points far below the spectrum to close the hull from the bottom
    p_min = s.min() - 1.0
    points_with_base = np.vstack((points, [[w[-1], p_min], [w[0], p_min]]))
    
    try:
        hull = ConvexHull(points_with_base)
        # Vertices of the upper hull (excluding the added base points)
        upper_vertices = sorted([v for v in hull.vertices if v < len(w)])
        
        # Interpolate the upper hull to get the continuum
        f_cont = interp1d(w[upper_vertices], s[upper_vertices], kind='linear', fill_value="extrapolate")
        continuum = f_cont(wavelengths)
        
        # Normalized Reflectance
        normalized = spectrum / (continuum + 1e-12)
        return normalized, continuum
    except:
        return np.full_like(spectrum, np.nan), np.full_like(spectrum, np.nan)

# ── 3. Feature Extraction ──────────────────────────────────────────────────────
print(f"[Task 1/3] Extracting Features from {LINES}x{SAMPLES} pixels...")
feature_list = []

for l in range(LINES):
    for s in range(SAMPLES):
        spectrum = cube[l, s, :]
        norm, cont = remove_continuum(spectrum, WL)
        
        if np.isnan(norm).all():
            continue
            
        # Find Dip
        idx = np.argmin(norm)
        depth = 1.0 - norm[idx]
        wavelength = WL[idx]
        
        # Width (FWHM proxy)
        half_max = 1.0 - (depth / 2.0)
        idx_left = np.where(norm[:idx] > half_max)[0]
        idx_right = np.where(norm[idx:] > half_max)[0]
        
        if len(idx_left) > 0 and len(idx_right) > 0:
            width = WL[idx + idx_right[0]] - WL[idx_left[-1]]
        else:
            width = 0.02 # Default narrow
            
        feature_list.append({
            'line': l, 'sample': s,
            'wavelength': wavelength,
            'depth': depth,
            'width': width,
            'area': depth * width * 0.5 # Proxy for area
        })

df_features = pd.DataFrame(feature_list)
df_features.to_csv(os.path.join(OUT_DIR, "feature_table.csv"), index=False)
np.save(os.path.join(OUT_DIR, "feature_vectors.npy"), df_features.values)

print(f"\n--- Feature Extraction Complete. Results in: {OUT_DIR} ---")
