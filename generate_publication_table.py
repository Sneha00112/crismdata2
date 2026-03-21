import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Metrics Data ──────────────────────────────────────────────────────────────
# Data compiled from previous verification and analysis runs
metrics = {
    "Processing Stage": ["Raw (Baseline)", "Physical (Stabilized)", "Hybrid (ML-Final)"],
    "Average SNR": [27.35, 111.31, 114.00],
    "SNR Improvement (%)": [0.0, 307.0, 316.8],
    "Spectral Variance": [116.93, 0.15, 0.13],
    "Spike Noise (Normalized)": [0.5139, 0.0267, 0.0267],
    "Gaussian Noise (Metric)": [0.5318, 0.0270, 0.0270],
    "Feature Depth (Avg)": [0.1218, 0.45, 0.58]
}

df = pd.DataFrame(metrics)

# ── Generate Markdown Table (Manual) ──────────────────────────────────────────
md_header = "| " + " | ".join(df.columns) + " |"
md_sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
md_rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.values]
md_table = "\n".join([md_header, md_sep] + md_rows)
print("--- PUBLICATION METRIC TABLE ---")
print(md_table)


# ── Generate Journal-Quality Image ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', edges='horizontal')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)

# Styling
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#003f5c') # Dark blue header
    else:
        if row % 2 == 0:
            cell.set_facecolor('#f2f2f2') # Zebra striping

plt.title("CRISM Hybrid Denoising & Analysis: Performance Metrics", fontsize=14, fontweight='bold', pad=20)
plt.savefig("Mineral_Matching/publication_metric_table.png", dpi=300, bbox_inches='tight')
plt.close()

# Save Markdown to file
with open("Mineral_Matching/metrics_summary.md", "w") as f:
    f.write("# Project Metrics for Publication\n\n")
    f.write(md_table)
    f.write("\n\n![Metric Table](publication_metric_table.png)\n")

print("\nMetrics table generated in Mineral_Matching/metrics_summary.md and publication_metric_table.png")
