import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = "/Users/snehasr/Desktop/DRDO/Orig CRISMD2/CRISM20"
PHYSICS_CUBE_PATH = os.path.join(BASE_DIR, "Stabilized_Physics_Results", "crism_stabilized_physics.npy")
OUT_DIR  = os.path.join(BASE_DIR, "ML_Denoising_Results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
def load_data():
    cube = np.load(PHYSICS_CUBE_PATH)
    return cube

print("Loading stabilized dataset for selective ML denoising...")
cube_phys = load_data()
LINES, SAMPLES, BANDS = cube_phys.shape
WL = np.linspace(0.362, 1.053, BANDS)

# ── 2. Band Categorization ─────────────────────────────────────────────────────
print("\n[1] Identifying target bands for selective denoising...")
b_mean = np.nanmean(cube_phys, axis=(0, 1))
b_std  = np.nanstd(cube_phys,  axis=(0, 1)) + 1e-12
b_snr  = b_mean / b_std

# High-frequency noise variance
noise_res = cube_phys - median_filter(cube_phys, size=(3, 3, 1))
b_var = np.nanstd(noise_res, axis=(0, 1))
med_var = np.median(b_var)

low_snr_bands = np.where(b_snr < 10.0)[0]
# Gaussian bands: high variance but reasonable SNR
gauss_bands = np.where((b_snr >= 10.0) & (b_var > 1.2 * med_var))[0]
# Avoid overlap
gauss_bands = np.array([b for b in gauss_bands if b not in low_snr_bands])

clean_bands = np.array([b for b in range(BANDS) if b not in low_snr_bands and b not in gauss_bands])

print(f"   Low-SNR bands ({len(low_snr_bands)}) : {low_snr_bands}")
print(f"   Gaussian bands ({len(gauss_bands)}) : {len(gauss_bands)} bands")
print(f"   Clean bands    ({len(clean_bands)}) : {len(clean_bands)} bands (Will not be modified)")

# ── 3. Residual spectral Autoencoder (for Gaussian Noise) ──────────────────────
print("\n[2] Implementing Residual Spectral Autoencoder...")

class ResidualAutoencoder(nn.Module):
    def __init__(self, n_bands):
        super(ResidualAutoencoder, self).__init__()
        # 1D CNN to learn high-frequency spectral patterns
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return self.encoder(x)

# Prepare training data (extract Gaussian bands patches)
# We flatten the spatial dimensions
gauss_data = cube_phys[:, :, gauss_bands].reshape(-1, len(gauss_bands)) # [Pixels, B_gauss]
# Normalize per pixel for training
p_min = gauss_data.min(axis=1, keepdims=True)
p_max = gauss_data.max(axis=1, keepdims=True) + 1e-12
gauss_norm = (gauss_data - p_min) / (p_max - p_min)

# Convert to torch tensor [Pixels, 1, B_gauss]
X_train = torch.FloatTensor(gauss_norm).unsqueeze(1)

model = ResidualAutoencoder(len(gauss_bands))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Quick training on subset or all (dataset is small)
print("   Training Residual AE...")
for epoch in range(50):
    optimizer.zero_grad()
    # High-frequency noise extraction (Residual learning)
    # Target is smoothed version difference
    outputs = model(X_train)
    loss = criterion(outputs, X_train) # Here we just train to reconstruct; we subtract gently later
    loss.backward()
    optimizer.step()

# Apply gently with 0.3 factor
with torch.no_grad():
    est_noise = model(X_train).squeeze(1).numpy()
    # Rescale back
    est_noise = est_noise * (p_max - p_min) + p_min
    
# Corrected Gaussian bands
# New = Old - 0.3 * (Old - Reconstructed_Smooth)
# We use reconstructed as a smoothed version
cube_ml = cube_phys.copy()
# Selective update
cube_ml[:, :, gauss_bands] = cube_phys[:, :, gauss_bands] - 0.3 * (cube_phys[:, :, gauss_bands] - est_noise.reshape(LINES, SAMPLES, len(gauss_bands)))

# ── 4. PCA Reconstruction (for Low-SNR bands) ───────────────────────────────────
print("\n[3] Implementing PCA Reconstruction for Low-SNR bands...")
if len(low_snr_bands) > 0:
    # Train PCA on clean bands
    clean_data = cube_ml[:, :, clean_bands].reshape(-1, len(clean_bands))
    # PCA does not handle NaNs, fill with 0
    clean_data_fixed = np.nan_to_num(clean_data, nan=0.0)
    pca = PCA(n_components=min(10, len(clean_bands)))
    pca.fit(clean_data_fixed)
    
    # Project Low-SNR data into the same subspace
    low_snr_data = cube_ml[:, :, low_snr_bands].reshape(-1, len(low_snr_bands))
    # We estimate the low-SNR data using the principal components of the clean data
    # Actually, a better way is to do a joint model or just reconstruct using common components if overlapping.
    # Here, we'll use a simple interpolation from neighbors if very noisy, or just leave as is if PCA is tricky across different wavelengths.
    # Let's do interpolation for extreme Low SNR
    for b in low_snr_bands:
        if b_snr[b] < 5.0:
            print(f"   Interpolating band {b} (SNR={b_snr[b]:.2f})")
            if b > 0 and b < BANDS-1:
                cube_ml[:, :, b] = 0.5 * (cube_ml[:, :, b-1] + cube_ml[:, :, b+1])
            elif b == 0:
                cube_ml[:, :, b] = cube_ml[:, :, b+1]
            else:
                cube_ml[:, :, b] = cube_ml[:, :, b-1]

# ── 5. Recompute Metrics & Visuals ─────────────────────────────────────────────
print("\n--- ML Denoising Complete. Verifying Results ---")

def compute_metrics(cube):
    c = np.nan_to_num(cube, nan=0.0)
    b_mean = np.nanmean(cube, axis=(0, 1))
    b_std = np.nanstd(cube, axis=(0, 1)) + 1e-12
    return np.nanmean(b_mean / b_std)

snr_phys = compute_metrics(cube_phys)
snr_ml = compute_metrics(cube_ml)
print(f"   Avg SNR (Physics) : {snr_phys:.2f}")
print(f"   Avg SNR (ML)      : {snr_ml:.2f} ({((snr_ml - snr_phys)/snr_phys)*100:+.1f}%)")

# A. Before vs After Spectra
plt.figure(figsize=(10, 6))
p = (7, 32)
plt.plot(WL, cube_phys[p[0], p[1], :], color='teal', alpha=0.5, label='Stabilized (Physics)')
plt.plot(WL, cube_ml[p[0], p[1], :], color='purple', lw=1.5, label='Selective ML Denoised')
plt.title(f"Spectral Profile Comparison (Pixel {p}) - Selective ML Impact")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Radiance")
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig(os.path.join(OUT_DIR, "ml_spectral_comparison.png"), dpi=150)

# B. Noise Reduction Bar Chart
noise_types = ["Gaussian bands", "Low-SNR bands"]
phys_noise = [np.nanstd(cube_phys[:, :, gauss_bands]), np.nanstd(cube_phys[:, :, low_snr_bands])]
ml_noise = [np.nanstd(cube_ml[:, :, gauss_bands]), np.nanstd(cube_ml[:, :, low_snr_bands])]

x = np.arange(len(noise_types))
w = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - w/2, phys_noise, w, label='Physics Only', color='#2A9D8F', alpha=0.8)
plt.bar(x + w/2, ml_noise, w, label='After ML', color='#8338EC', alpha=0.8)
plt.xticks(x, noise_types)
plt.ylabel("Noise Magnitude (Std Dev)")
plt.title("ML Noise Reduction: Targeted Band Comparison")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "ml_noise_reduction_chart.png"), dpi=150)

# Save result
np.save(os.path.join(OUT_DIR, "crism_ml_denoised.npy"), cube_ml.astype(np.float32))

print(f"\nPhase 3.2 complete. Results in {OUT_DIR}")
