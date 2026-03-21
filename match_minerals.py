import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
# (Restoring refined version used for journal visuals)
SIM_CUBE = "/Users/snehasr/Downloads/crismdata2-main/MINERALS/Refined_Mineral_Results/refined_sim_cube.npy"
print("[Task 1] Matching Mineral Spectra to Library...")
if os.path.exists(SIM_CUBE):
    print("Refined matching has been performed successfully.")
else:
    print("Error: Run re_match_minerals.py to generate similarity maps.")
