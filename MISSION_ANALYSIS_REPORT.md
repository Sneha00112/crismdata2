# Research Project: Hyperspectral Noise Analysis

This report fulfills Phase 1 and Phase 2 of the research roadmap for the CRISM dataset.

---

## 🔹 PHASE 1: DATASET ACQUISITION (CRISM)

### 1. Dataset Summary Table

| Mission / Dataset | Sensor | Spectral Range (µm) | Spatial Resolution | Samples x Lines | Missing / Corrupted |
|---|---|---|---|---|---|
| **CRISM** (Mars) | VNIR (S) | 0.362 – 1.053 | ~20 m/pixel | 64 x 15 | 7.81% (Fill Values) |

### 2. Known Noise Issues
*   **Striping**: Prominent along-track vertical banding due to detector gain offsets.
*   **Spikes**: High-amplitude impulse noise (cosmic rays).
*   **Low SNR**: Significant signal degradation at the spectral edges (<0.4 µm and >1.0 µm).
*   **Atmospheric Features**: Sharp absorption dips at 0.72, 0.76, and 0.94 µm.

### 3. Noise Behavior Notes
The CRISM VNIR data is characterized by a "push-broom" acquisition style, making it highly susceptible to detector non-uniformity (striping). The Mars atmosphere introduces systematic spectral distortions that can be misidentified as sensor noise.

![Spectral Quality Analysis: Raw vs. Denoised Spectra](/Users/snehasr/.gemini/antigravity/brain/ef73a9d3-3214-4e05-8feb-1bf9497bacf5/spectral_noise_comparison.png)

---

## 🔹 PHASE 2: NOISE CHARACTERIZATION (CRISM)

### 1. Noise Profile Analysis
The noise profile was quantitatively assessed using SNR estimation and spatial/spectral variance analysis.

| Noise Category | Severity | Metric | Primary Source |
|---|---|---|---|
| **Structured** | **High** | 1.049 (Col Std) | Detector Striping |
| **Random** | **High** | 0.554 (Std Dev) | Thermal / Read Noise |
| **Severe degradation** | **Medium** | 0.81 spike count | Cosmic Rays / Bad Pixels |
| **Spectral Distortion** | **High** | 0.514 (d2 Std) | Mars Atmosphere (Dust/Gas) |

### 2. Noise-Environment Linkage
*   **Dust & Atmosphere**: Mars' dusty atmosphere causes significant scattering and absorption features (O2, H2O) in the VNIR range, particularly visible as spectral "dips" which represent the environment rather than sensor artifacts.
*   **Low SNR**: The low solar flux and detector quantum efficiency at the VNIR boundaries contribute to the severe noise floor at band edges.

![Characterized Noise Severities by Category](/Users/snehasr/.gemini/antigravity/brain/ef73a9d3-3214-4e05-8feb-1bf9497bacf5/categorical_noise_severity.png)

### 3. Justification for Hybrid Preprocessing
A standard linear filter is insufficient due to the diversity of noise types:
1.  **Physics-based correction** is required for systematic striping and atmospheric features.
2.  **ML-based denoising** (e.g., Autoencoders) is justified for suppressing complex stochastic (Gaussian) and non-linear spike noise without blurring fine spectral details.

---

## ✅ Phase Outputs
*   [x] Dataset summary table
*   [x] Raw vs noisy spectral plots (contained in `data_analysis_summary.png`)
*   [x] Noise behavior notes
*   [x] Noise profile plots (contained in `noise_severity_chart.png`)
*   [x] Justification for hybrid preprocessing
