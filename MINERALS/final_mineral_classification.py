import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_CSV = "/Users/snehasr/Downloads/crismdata2-main/Mineral_Features/feature_table.csv"
SIMILARITY_NPY = "/Users/snehasr/Downloads/crismdata2-main/Mineral_Matching/pixel_similarity.npy"
OUT_DIR = "/Users/snehasr/Downloads/crismdata2-main/Final_Mineral_Map"
os.makedirs(OUT_DIR, exist_ok=True)

LINES, SAMPLES, BANDS = 15, 64, 107
WL = np.linspace(0.362, 1.053, BANDS)

# ── 1. Re-build Library for Training ───────────────────────────────────────────
print("[Task 1] Re-building Virtual Library for SVM Training...")

minerals_lib = {
    'Olivine':          {'center': 1.00, 'width': 0.15, 'depth': 0.8},
    'Low-Ca Pyroxene':  {'center': 0.90, 'width': 0.10, 'depth': 0.7},
    'High-Ca Pyroxene': {'center': 0.95, 'width': 0.12, 'depth': 0.6},
    'Nontronite':       {'center': 0.65, 'width': 0.05, 'depth': 0.4},
    'Montmorillonite':  {'center': 0.45, 'width': 0.04, 'depth': 0.3},
    'Kaolinite':        {'center': 0.96, 'width': 0.03, 'depth': 0.2},
    'Gypsum':           {'center': 1.00, 'width': 0.03, 'depth': 0.4},
    'Kieserite':        {'center': 1.01, 'width': 0.04, 'depth': 0.5},
    'Carbonates':       {'center': 0.70, 'width': 0.05, 'depth': 0.3},
    'Hematite':         {'center': 0.86, 'width': 0.10, 'depth': 0.9},
    'Magnetite':        {'center': 0.50, 'width': 0.20, 'depth': 0.1},
    'Ilmenite':         {'center': 0.50, 'width': 0.20, 'depth': 0.1},
    'Hydrated silica':  {'center': 1.00, 'width': 0.05, 'depth': 0.4},
    'Sulfates':         {'center': 0.43, 'width': 0.04, 'depth': 0.3}
}

mineral_names = list(minerals_lib.keys())
train_X, train_y = [], []

for i, name in enumerate(mineral_names):
    # Augment training data with small variations
    params = minerals_lib[name]
    for _ in range(20):
        c = params['center'] + np.random.normal(0, 0.005)
        d = params['depth'] + np.random.normal(0, 0.05)
        train_X.append([c, d])
        train_y.append(i)

X_train = np.array(train_X)
y_train = np.array(train_y)

# ── 2. Train SVM Model ─────────────────────────────────────────────────────────
print("[Task 2] Training SVM Classifier...")
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm.fit(X_train, y_train)

# ── 3. Hybrid Classification Logic ─────────────────────────────────────────────
print("[Task 3] Running Hybrid Physics + SVM Classification...")

df_pixels = pd.read_csv(FEATURES_CSV)
sim_cube = np.load(SIMILARITY_NPY)

# Map for IDs
label_to_id = {
    'Unknown': 0, 'Olivine': 1, 'Pyroxene': 2, 'Clay': 3,
    'Sulfate': 4, 'Carbonate': 5, 'Oxide': 6, 'Hydrated': 7, 'Mixed': 8
}

def get_class_id(name):
    n = name.lower()
    if 'olivine' in n: return 1
    if 'pyroxene' in n: return 2
    if 'nontronite' in n or 'mont' in n or 'kaol' in n: return 3
    if 'sulfate' in n or 'kieserite' in n or 'gypsum' in n: return 4
    if 'carbonate' in n: return 5
    if 'hematite' in n or 'magnetite' in n or 'ilmenite' in n: return 6
    if 'hydrated' in n or 'silica' in n: return 7
    return 8

final_labels = np.zeros((LINES, SAMPLES))
final_conf = np.zeros((LINES, SAMPLES))
uncertain_mask = np.zeros((LINES, SAMPLES))

results = []

for l in range(LINES):
    for s in range(SAMPLES):
        p_row = df_pixels[(df_pixels['line'] == l) & (df_pixels['sample'] == s)]
        if p_row.empty: continue
        
        best_dip = p_row.loc[p_row['depth'].idxmax()]
        px_feat = np.array([[best_dip['wavelength'], best_dip['depth']]])
        
        # Physics Rule Confidence check
        # e.g., Nontronite rule (2.3um logic adapted)
        rule_score = 0.5
        wv = best_dip['wavelength']
        if 0.9 <= wv <= 1.05: rule_score += 0.3 # Mafic region
        if 0.4 <= wv <= 0.7:  rule_score += 0.3 # Clay/Oxide region
        
        # SVM Prediction
        svm_probs = svm.predict_proba(px_feat)[0]
        svm_idx = np.argmax(svm_probs)
        svm_name = mineral_names[svm_idx]
        svm_conf = svm_probs[svm_idx]
        
        # Similarity from previous step
        sim_scores = sim_cube[l, s, :]
        sim_idx = np.argmax(sim_scores)
        sim_name = mineral_names[sim_idx]
        sim_conf = sim_scores[sim_idx]
        
        # Hybrid Decision
        if sim_conf > 0.8 and rule_score > 0.7:
            label_name = sim_name
            conf = (sim_conf + rule_score) / 2.0
            source = "Physics_Rules"
        elif svm_conf > 0.6:
            label_name = svm_name
            conf = svm_conf
            source = "SVM_ML"
        else:
            label_name = "Mixed"
            conf = max(sim_conf, svm_conf)
            source = "Ambiguous"
            uncertain_mask[l, s] = 1
            
        final_labels[l, s] = get_class_id(label_name)
        final_conf[l, s] = conf
        
        results.append({
            'line': l, 'sample': s, 'mineral': label_name, 
            'confidence': conf, 'source': source, 'wavelength': wv
        })

# ── 4. Generate Publication Maps ──────────────────────────────────────────────
print("[Task 4] Generating Final Mineral Map and Analysis...")

plt.figure(figsize=(10, 6))
plt.imshow(final_labels, cmap='tab10', interpolation='nearest')
plt.title("Final Hybrid Mineral classification (CRISM Baseline)")
cb = plt.colorbar(ticks=range(9))
cb.ax.set_yticklabels(['Unknown', 'Olivine', 'Pyroxene', 'Clay', 'Sulfate', 'Carbonate', 'Oxide', 'Hydrated', 'Mixed'])
plt.savefig(os.path.join(OUT_DIR, "final_mineral_map.png"), dpi=300)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(final_conf, cmap='magma')
plt.title("Classification Confidence")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(uncertain_mask, cmap='gray')
plt.title("Uncertainty Mask (Mixed Pixels)")
plt.savefig(os.path.join(OUT_DIR, "final_confidence_uncertainty.png"), dpi=300)

# Statistics
df_final = pd.DataFrame(results)
stats = df_final['mineral'].value_counts()
stats.to_csv(os.path.join(OUT_DIR, "mineral_statistics.csv"))

plt.figure(figsize=(10, 5))
stats.plot(kind='bar', color='forestgreen')
plt.title("Mineral Distribution (Final Classification)")
plt.ylabel("Number of Pixels")
plt.savefig(os.path.join(OUT_DIR, "mineral_distribution_chart.png"), dpi=300)

# ── 5. Export Final Cube ───────────────────────────────────────────────────────
print("[Task 5] Exporting Final Mineral Cube...")
# Band 1: ID, Band 2: Confidence, Band 3: Uncertainty
final_cube = np.zeros((LINES, SAMPLES, 3))
final_cube[:, :, 0] = final_labels
final_cube[:, :, 1] = final_conf
final_cube[:, :, 2] = uncertain_mask
np.save(os.path.join(OUT_DIR, "final_mineral_cube.npy"), final_cube)
df_final.to_csv(os.path.join(OUT_DIR, "final_mineral_table.csv"), index=False)

print(f"\nHybrid Classification Complete. Results in {OUT_DIR}")
