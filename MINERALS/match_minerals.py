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
OUT_DIR = "/Users/snehasr/Downloads/crismdata2-main/Mineral_Matching"
os.makedirs(OUT_DIR, exist_ok=True)

LINES, SAMPLES, BANDS = 15, 64, 107
WL = np.linspace(0.362, 1.053, BANDS)

# ── Task 1: Load Spectral Libraries (Science-Informed Virtual models) ────────
print("[Task 1] Building Virtual Spectral Library for 14 Target Minerals...")

# Mineral Definitions (Center wavelength, Width, Depth, Rule Category)
# Note: Centers adapted to VNIR range (0.36-1.05) or typical broad features
minerals_lib = {
    'Olivine':          {'center': 1.00, 'width': 0.15, 'depth': 0.8, 'cat': 'mafic'},
    'Low-Ca Pyroxene':  {'center': 0.90, 'width': 0.10, 'depth': 0.7, 'cat': 'mafic'},
    'High-Ca Pyroxene': {'center': 0.95, 'width': 0.12, 'depth': 0.6, 'cat': 'mafic'},
    'Nontronite':       {'center': 0.65, 'width': 0.05, 'depth': 0.4, 'cat': 'clay'}, # VNIR electronic
    'Montmorillonite':  {'center': 0.45, 'width': 0.04, 'depth': 0.3, 'cat': 'clay'},
    'Kaolinite':        {'center': 0.96, 'width': 0.03, 'depth': 0.2, 'cat': 'clay'},
    'Gypsum':           {'center': 1.00, 'width': 0.03, 'depth': 0.4, 'cat': 'sulfate'},
    'Kieserite':        {'center': 1.01, 'width': 0.04, 'depth': 0.5, 'cat': 'sulfate'},
    'Carbonates':       {'center': 0.70, 'width': 0.05, 'depth': 0.3, 'cat': 'carbonate'},
    'Hematite':         {'center': 0.86, 'width': 0.10, 'depth': 0.9, 'cat': 'oxide'},
    'Magnetite':        {'center': 0.50, 'width': 0.20, 'depth': 0.1, 'cat': 'oxide'}, # Flat dark
    'Ilmenite':         {'center': 0.50, 'width': 0.20, 'depth': 0.1, 'cat': 'oxide'},
    'Hydrated silica':  {'center': 1.00, 'width': 0.05, 'depth': 0.4, 'cat': 'hydration'},
    'Sulfates':         {'center': 0.43, 'width': 0.04, 'depth': 0.3, 'cat': 'sulfate'}
}

def generate_model_spectrum(params, wavelengths):
    # Gaussian absorption dip model
    c, w, d = params['center'], params['width'], params['depth']
    spectrum = 1.0 - d * np.exp(-0.5 * ((wavelengths - c) / w)**2)
    return spectrum

def extract_lib_features(spectrum, wavelengths):
    # Continuum removal for model
    pts = np.vstack((wavelengths, spectrum)).T
    p_min = spectrum.min() - 1
    pts_with_base = np.vstack((pts, [[wavelengths[-1], p_min], [wavelengths[0], p_min]]))
    hull = ConvexHull(pts_with_base)
    vertices = sorted([v for v in hull.vertices if v < len(wavelengths)])
    f_interp = interp1d(wavelengths[vertices], spectrum[vertices], kind='linear', fill_value="extrapolate")
    cont = f_interp(wavelengths)
    norm = spectrum / (cont + 1e-12)
    
    # Feature Vector: [wavelength_of_max_dip, max_depth]
    idx = np.argmin(norm)
    return [wavelengths[idx], 1.0 - norm[idx]]

lib_features = {}
plt.figure(figsize=(12, 6))
for name, params in minerals_lib.items():
    spec = generate_model_spectrum(params, WL)
    feat = extract_lib_features(spec, WL)
    lib_features[name] = feat
    plt.plot(WL, spec, label=name)

plt.title("Virtual Spectral Library Models (Raw)")
plt.xlabel("Wavelength (µm)")
plt.legend(ncol=3, fontsize=8)
plt.savefig(os.path.join(OUT_DIR, "library_spectra.png"), dpi=200)

# ── Task 2: Feature Distance Matching ──────────────────────────────────────────
print("[Task 2] Computing Feature Distance Matching...")

# Pixel features from CSV
df_pixel_all = pd.read_csv(FEATURES_CSV)
# We aggregate features per pixel (take the deepest dip for comparison)
pixel_groups = df_pixel_all.groupby(['line', 'sample'])

similarity_results = []

for (l, s), group in pixel_groups:
    best_dip = group.loc[group['depth'].idxmax()]
    p_feat = [best_dip['wavelength'], best_dip['depth']]
    
    pixel_sims = {}
    for min_name, l_feat in lib_features.items():
        # Weighted Distance: 0.7 wavelength, 0.3 depth
        dist = 0.7 * abs(p_feat[0] - l_feat[0]) + 0.3 * abs(p_feat[1] - l_feat[1])
        sim = np.exp(-10 * dist) # Sensitivity factor 10
        
        # ── Task 3: CRISM Physics Rules ────────────────────────────────────────
        # Rule boosting
        cat = minerals_lib[min_name]['cat']
        if best_dip['wavelength'] > 0.85 and cat == 'mafic':
            sim *= 1.2 # Boost mafic in 0.9-1.0 range
        if best_dip['wavelength'] < 0.5 and cat == 'sulfate':
            sim *= 1.3 # Boost sulfates in UV/Visible edge
        if (0.6 < best_dip['wavelength'] < 0.7) and cat == 'clay':
            sim *= 1.5 # Boost specific clay electronic features
            
        pixel_sims[min_name] = sim
    
    similarity_results.append({'line': l, 'sample': s, **pixel_sims})

df_sim = pd.DataFrame(similarity_results)

# ── Task 4: Combined Analysis & Top Minerals ──────────────────────────────────
print("[Task 4] Identifying Top Minerals & Spatial Maps...")

top_minerals = []
sim_array = df_sim.drop(['line', 'sample'], axis=1).values
mineral_names = list(lib_features.keys())

for idx, row in df_sim.iterrows():
    l, s = int(row['line']), int(row['sample'])
    row_vals = row.drop(['line', 'sample']).values
    best_idx = np.argsort(row_vals)[-3:][::-1] # Top 3
    top_minerals.append({
        'line': l, 'sample': s,
        'mineral_1': mineral_names[best_idx[0]],
        'sim_1': row_vals[best_idx[0]],
        'mineral_2': mineral_names[best_idx[1]],
        'mineral_3': mineral_names[best_idx[2]]
    })

df_top = pd.DataFrame(top_minerals)

# Create Map
mineral_map = np.zeros((LINES, SAMPLES))
# Map names to integers
name_to_id = {name: i for i, name in enumerate(mineral_names)}
for idx, row in df_top.iterrows():
    mineral_map[int(row['line']), int(row['sample'])] = name_to_id[row['mineral_1']]

plt.figure(figsize=(10, 6))
plt.imshow(mineral_map, cmap='tab20', interpolation='nearest')
plt.title("Dominant Mineral Map (Top 1 Similarity)")
cb = plt.colorbar(ticks=range(len(mineral_names)))
cb.ax.set_yticklabels(mineral_names)
plt.savefig(os.path.join(OUT_DIR, "mineral_top_map.png"), dpi=200)

# Top Mineral Histogram
plt.figure(figsize=(10, 5))
df_top['mineral_1'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Mineral Frequency Across Scene")
plt.ylabel("Pixel Count")
plt.savefig(os.path.join(OUT_DIR, "mineral_frequency.png"), dpi=200)

# ── Task 5: Publication Outputs ───────────────────────────────────────────────
print("[Task 5] Saving Tables & Vectors...")

df_sim.to_csv(os.path.join(OUT_DIR, "pixel_similarity.csv"), index=False)
df_top.to_csv(os.path.join(OUT_DIR, "mineral_table.csv"), index=False)

# Export Similarity Cube (Lines, Samples, Minerals)
sim_cube = np.zeros((LINES, SAMPLES, len(mineral_names)))
for idx, row in df_sim.iterrows():
    l, s = int(row['line']), int(row['sample'])
    sim_cube[l, s, :] = row.drop(['line', 'sample']).values
np.save(os.path.join(OUT_DIR, "pixel_similarity.npy"), sim_cube)

print("\n--- Mineral Matching Complete ---")
print(f"Results saved to: {OUT_DIR}")
