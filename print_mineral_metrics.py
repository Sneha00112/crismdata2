import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
MINERAL_STATS_CSV = "/Users/snehasr/Downloads/crismdata2-main/MINERALS/Refined_Mineral_Results/refined_mineral_stats.csv"
SITES_CSV = "/Users/snehasr/Downloads/crismdata2-main/Journal_Results/mission_sites.csv"
RESOURCE_CSV = "/Users/snehasr/Downloads/crismdata2-main/Journal_Results/final_resource_table.csv"

def print_metric_table():
    print("\n" + "="*80)
    print(" " * 20 + "CRISM PLANETARY ANALYSIS: FINAL MISSION METRICS")
    print("="*80)
    
    # 1. Mineral Abundance
    print("\n[PART 1: MINEROLOGICAL DISTRIBUTION]")
    print("-" * 50)
    if os.path.exists(MINERAL_STATS_CSV):
        df_stats = pd.read_csv(MINERAL_STATS_CSV)
        # Format for terminal
        print(f"{'Mineral Class':<25} | {'Abundance (%)':<15}")
        print("-" * 50)
        for _, row in df_stats.iterrows():
            print(f"{row['Mineral']:<25} | {row['Abundance (%)']:>12.2f}%")
    else:
        print("Error: refined_mineral_stats.csv not found.")

    # 2. Resource Indicators
    print("\n[PART 2: GLOBAL RESOURCE POTENTIAL (ISRU)]")
    print("-" * 50)
    if os.path.exists(RESOURCE_CSV):
        df_res = pd.read_csv(RESOURCE_CSV)
        print(f"{'Resource Type':<25} | {'Detected Pixels':<15}")
        print("-" * 50)
        for _, row in df_res.iterrows():
            print(f"{row['Resource']:<25} | {int(row['Abundance_Pixels']):>15}")
    else:
        print("Error: final_resource_table.csv not found.")

    # 3. Top Mission Sites
    print("\n[PART 3: RECOMMENDED LANDING SITES (TOP 5)]")
    print("-" * 80)
    if os.path.exists(SITES_CSV):
        df_sites = pd.read_csv(SITES_CSV)
        print(f"{'Rank':<5} | {'Coords (L, S)':<15} | {'Primary Mineral':<20} | {'ISRU Score':<12} | {'Suitability'}")
        print("-" * 80)
        for _, row in df_sites.head(5).iterrows():
            coords = f"({int(row['Line'])}, {int(row['Sample'])})"
            print(f"{int(row['Rank']):<5} | {coords:<15} | {row['Primary_Mineral']:<20} | {row['ISRU_Score']:>12.4f} | {row['Suitability']}")
    else:
        print("Error: mission_sites.csv not found.")

    print("\n" + "="*80)
    print("End of Report")
    print("="*80 + "\n")

if __name__ == "__main__":
    print_metric_table()
