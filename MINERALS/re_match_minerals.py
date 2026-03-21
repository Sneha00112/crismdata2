import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_CSV = "/Users/snehasr/Downloads/crismdata2-main/Mineral_Features/feature_table.csv"
LIB_DIR = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/CRISM20/usgs_splib07/ASCIIdata/ASCIIdata_splib07b_cvAVIRISc2005"
WL_FILE = os.path.join(LIB_DIR, "s07_AV05_AVIRIS_2005_Wavelengths_(um)_224_chans.txt")
OUT_DIR = "/Users/snehasr/Downloads/crismdata2-main/Refined_Mineral_Results"
os.makedirs(OUT_DIR, exist_ok=True)

LINES, SAMPLES, BANDS = 15, 64, 107
WL_CRISM = np.linspace(0.362, 1.053, BANDS)

# ── 1. Load Real USGS Library ──────────────────────────────────────────────────
print("[Task 1] Loading Real USGS Spectral Library (AV05 set)...")

def load_usgs_file(path):
    with open(path, 'r') as f:
        # Skip header and filter out invalid/extremely negative values
        vals = []
        for line in f.readlines()[1:]:
            try:
                v = float(line.strip())
                if v > -1e10: # Filter out -1.23e34 and similar
                    vals.append(v)
                else:
                    vals.append(np.nan)
            except:
                continue
    return np.array(vals)


wl_lib = load_usgs_file(WL_FILE)

import glob

# 14 Minerals Map (Discovering files dynamically in LIB_DIR)
target_names = [
    'Olivine', 'Enstatite', 'Augite', 'Nontronite', 'Montmorillonite', 
    'Kaolinite', 'Gypsum', 'Jarosite', 'Alunite', 'Kieserite', 
    'Calcite', 'Hematite', 'Magnetite', 'Opal'
]

lib_files = {}
for name in target_names:
    # Look in ChapterM, ChapterS, etc.
    pattern = os.path.join(LIB_DIR, "Chapter*", f"s07_AV05_{name}*.txt")
    matches = glob.glob(pattern)
    if matches:
        # Avoid errorbars
        matches = [m for m in matches if 'errorbars' not in m]
        if matches:
            lib_files[name] = matches[0]

if not lib_files:
    print("Critical Error: No library files found! Check LIB_DIR.")
    exit(1)


def extract_features_real(spectrum, wl_src, wl_target):
    # Mask NaNs
    valid = ~np.isnan(spectrum)
    if np.sum(valid) < 3:
        return [0, 0], np.zeros_like(wl_target), np.zeros_like(wl_target)
    
    # Interpolate to CRISM grid
    f = interp1d(wl_src[valid], spectrum[valid], kind='linear', fill_value="extrapolate")
    spec_crism = f(wl_target)
    
    # Continuum Removal
    pts = np.vstack((wl_target, spec_crism)).T
    p_min = spec_crism.min() - 1
    pts_with_base = np.vstack((pts, [[wl_target[-1], p_min], [wl_target[0], p_min]]))
    hull = ConvexHull(pts_with_base)

    vertices = sorted([v for v in hull.vertices if v < len(wl_target)])
    f_cont = interp1d(wl_target[vertices], spec_crism[vertices], kind='linear', fill_value="extrapolate")
    cont = f_cont(wl_target)
    norm = spec_crism / (cont + 1e-12)
    
    idx = np.argmin(norm)
    return [wl_target[idx], 1.0 - norm[idx]], spec_crism, norm

lib_feat = {}
lib_norm = {}
plt.figure(figsize=(12, 6))
for name, fpath in lib_files.items():
    if not os.path.exists(fpath): continue
    spec = load_usgs_file(fpath)
    feat, spec_c, norm = extract_features_real(spec, wl_lib, WL_CRISM)
    lib_feat[name] = feat
    lib_norm[name] = norm
    print(f"Library Mineral: {name:<20} | Center: {feat[0]:.3f} | Depth: {feat[1]:.3f}")
    plt.plot(WL_CRISM, norm, label=name)


plt.title("USGS Scientific Library (Continuum Removed)")
plt.xlabel("Wavelength (µm)")
plt.legend(ncol=3, fontsize=8)
plt.savefig(os.path.join(OUT_DIR, "real_library_spectra.png"), dpi=200)

# ── 2. Refined Matching (Multi-Mineral Heatmaps) ──────────────────────────────
print("[Task 2] Computing Refined Similarity & Heatmaps...")

df_pixel = pd.read_csv(FEATURES_CSV)
pixel_groups = df_pixel.groupby(['line', 'sample'])

# Initialize similarity maps for ALL minerals
sim_maps = {name: np.zeros((LINES, SAMPLES)) for name in lib_feat.keys()}

for (l, s), group in pixel_groups:
    best_dip = group.loc[group['depth'].idxmax()]
    p_feat = [best_dip['wavelength'], best_dip['depth']]
    
    for name, l_feat in lib_feat.items():
        # Distance calculation
        d_wl = abs(p_feat[0] - l_feat[0])
        d_dp = abs(p_feat[1] - l_feat[1])
        
        # New Sensitivity: 0.8 wavelength, 0.2 depth (Wavelength is key for ID)
        dist = 0.8 * d_wl + 0.2 * d_dp
        sim = np.exp(-5 * dist) # Relaxed sensitivity (5) for multi-mineral overlap

        
        # Physics Rules (Rule-based boost)
        if name == 'Nontronite' and 0.6 <= p_feat[0] <= 0.7: sim *= 1.5
        if 'Olivine' in name and 0.95 <= p_feat[0] <= 1.05: sim *= 1.3
        if 'Pyroxene' in name and 0.85 <= p_feat[0] <= 0.95: sim *= 1.3
        
        sim_maps[name][l, s] = sim

# ── 3. Final Outputs & Paper Visuals ──────────────────────────────────────────
print("[Task 3] Generating Heatmaps and Statistics...")

# Layout of heatmaps
n_minerals = len(sim_maps)
rows = (n_minerals + 2) // 3
fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
axes = axes.flatten()

for i, (name, smap) in enumerate(sim_maps.items()):
    im = axes[i].imshow(smap, cmap='magma', interpolation='nearest')
    axes[i].set_title(f"{name} Similarity Heatmap")
    plt.colorbar(im, ax=axes[i])

# Hide empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mineral_similarity_heatmaps.png"), dpi=300)

# Calculate Abundance (Similarity-Weighted)
print("[Task 4] Calculating Weighted Abundance and Exporting Results...")

total_sim = {name: np.sum(smap) for name, smap in sim_maps.items()}
grand_total = sum(total_sim.values())
stats_pct = {name: (val / grand_total) * 100 for name, val in total_sim.items()}

df_stats = pd.DataFrame(list(stats_pct.items()), columns=['Mineral', 'Abundance (%)'])
df_stats = df_stats.sort_values('Abundance (%)', ascending=False)
df_stats.to_csv(os.path.join(OUT_DIR, "refined_mineral_stats.csv"), index=False)


# ── 4. Save Refined Data for ISRU Analysis ──────────────────────────────────
print("[Task 5] Saving Refined Similarity Cube and Labels...")
# Stack similarity maps into a cube (Minerals, Lines, Samples)
sim_names = list(sim_maps.keys())
sim_cube = np.stack([sim_maps[name] for name in sim_names], axis=0)
np.save(os.path.join(OUT_DIR, "refined_sim_cube.npy"), sim_cube)
np.save(os.path.join(OUT_DIR, "refined_sim_names.npy"), np.array(sim_names))

# Create a final ID cube for ISRU (Winner for simplicity in some maps)
id_map_refined = {i: name for i, name in enumerate(sim_names)}
final_ids_numeric = np.zeros((LINES, SAMPLES), dtype=int)
for l in range(LINES):
    for s in range(SAMPLES):
        scores = {i: sim_cube[i, l, s] for i in range(len(sim_names))}
        final_ids_numeric[l, s] = max(scores, key=scores.get)

np.save(os.path.join(OUT_DIR, "refined_id_map.npy"), final_ids_numeric)

print("\n--- Refined Mineral Matching Complete ---")

print(f"Results saved to: {OUT_DIR}")
