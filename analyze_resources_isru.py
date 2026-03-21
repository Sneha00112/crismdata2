import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
REFINED_DIR = "/Users/snehasr/Downloads/crismdata2-main/MINERALS/Refined_Mineral_Results"
SIM_CUBE_PATH = os.path.join(REFINED_DIR, "refined_sim_cube.npy")

SIM_NAMES_PATH = os.path.join(REFINED_DIR, "refined_sim_names.npy")
ID_MAP_PATH = os.path.join(REFINED_DIR, "refined_id_map.npy")
OUT_DIR = "/Users/snehasr/Downloads/crismdata2-main/Journal_Results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
print("[Task 1] Loading Refined Multi-Mineral Data...")
sim_cube = np.load(SIM_CUBE_PATH) # (Minerals, Lines, Samples)
sim_names = np.load(SIM_NAMES_PATH)
mineral_ids = np.load(ID_MAP_PATH) # (Lines, Samples) - Winner-takes-all IDs

name_to_idx = {name: i for i, name in enumerate(sim_names)}
LINES, SAMPLES = mineral_ids.shape

# ── 2. Mineral Abundance Analysis (Journal Bar Chart) ──────────────────────────
print("[Task 2] Quantifying Refined Mineral Abundances...")
# Use weighted abundance for the paper chart
total_sim = np.sum(sim_cube, axis=(1, 2))
abundance_pct = (total_sim / np.sum(total_sim)) * 100

df_abundance = pd.DataFrame({'Mineral': sim_names, 'Percentage': abundance_pct})
df_abundance = df_abundance.sort_values('Percentage', ascending=False)
df_abundance.to_csv(os.path.join(OUT_DIR, "refined_mineral_stats.csv"), index=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=df_abundance, x='Mineral', y='Percentage', palette='magma')
plt.title("Refined Scene Mineral Abundance (%) - High Fidelity USGS Library", fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.ylabel("Weighted Abundance (%)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "refined_abundance_bar_chart.png"), dpi=300)

# ── 3. Resource & Metal Indicator Detection (Refined) ──────────────────────────
print("[Task 3] Mapping Refined Resource Indicators (Fe, Ti, H2O)...")

# Resource Logical Maps (using similarity sums)
# Fe: Hematite + Magnetite + Mafic (Olivine, Enstatite, Augite)
fe_indices = [name_to_idx[n] for n in ['Hematite', 'Magnetite', 'Olivine', 'Enstatite', 'Augite'] if n in name_to_idx]
fe_map = np.sum(sim_cube[fe_indices, :, :], axis=0)

# Ti: Assuming Magnetite/Oxides as a proxy here if Ilmenite wasn't explicitly in the 14
ti_map = sim_cube[name_to_idx['Magnetite'], :, :] if 'Magnetite' in name_to_idx else np.zeros_like(fe_map)

# Hydration: Nontronite, Montmorillonite, Kaolinite, Gypsum, Jarosite, Alunite, Kieserite, Opal
water_indices = [name_to_idx[n] for n in ['Nontronite', 'Montmorillonite', 'Kaolinite', 'Gypsum', 'Jarosite', 'Alunite', 'Kieserite', 'Opal'] if n in name_to_idx]
water_map = np.sum(sim_cube[water_indices, :, :], axis=0)

# Normalize for visualization
fe_map /= (fe_map.max() + 1e-12)
water_map /= (water_map.max() + 1e-12)

# Smooth for "Zones"
fe_zone = gaussian_filter(fe_map, sigma=0.8)
water_zone = gaussian_filter(water_map, sigma=0.8)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
im0 = axes[0].imshow(fe_zone, cmap='Reds'); axes[0].set_title("Iron (Fe) Resource Potential"); plt.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(ti_map, cmap='Greys'); axes[1].set_title("Titanium (Ti) Proxy (Magnetite)"); plt.colorbar(im1, ax=axes[1])
im2 = axes[2].imshow(water_zone, cmap='Blues'); axes[2].set_title("Hydration (H2O) Potential"); plt.colorbar(im2, ax=axes[2])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "refined_resource_indicators_map.png"), dpi=300)

# ── 4. Diverse Mineral Map (Multi-Color) ──────────────────────────────────────
print("[Task 4] Generating Multi-Color Mineral Map...")
plt.figure(figsize=(12, 8))
# Use a discrete colormap for the 14 minerals
cmap = plt.cm.get_cmap('tab20', len(sim_names))
im = plt.imshow(mineral_ids, cmap=cmap)
plt.title("Refined Mineral Identification Map (14 Classes)")
cb = plt.colorbar(im, ticks=range(len(sim_names)))
cb.ax.set_yticklabels(sim_names)
plt.savefig(os.path.join(OUT_DIR, "refined_multi_mineral_map.png"), dpi=300)

# ── 5. ISRU & Mission Suitability Scoring (Refined) ───────────────────────────
print("[Task 5] Computing Refined Mission & ISRU Suitability Scores...")

# ISRU Score: Weight Water (0.7) and Metals (0.3)
isru_score = (water_map * 0.7 + fe_map * 0.3)
# Landing Site Score: High resource overlap + diversity
landing_score = isru_score * (1.0 - (np.max(sim_cube, axis=0) / (np.sum(sim_cube, axis=0) + 1e-12))) # Diversity factor
landing_score = gaussian_filter(landing_score, sigma=0.5)
landing_score /= (landing_score.max() + 1e-12)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
im0 = axes[0].imshow(isru_score, cmap='magma'); axes[0].set_title("Refined ISRU Resource Value Map"); plt.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(landing_score, cmap='YlGn'); axes[1].set_title("Refined Landing Site Suitability Score"); plt.colorbar(im1, ax=axes[1])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "refined_mission_analysis_isru.png"), dpi=300)

print("\n--- Refined ISRU Analysis Complete. Final Figures Generated ---")

print(f"Publication Results in: {OUT_DIR}")
