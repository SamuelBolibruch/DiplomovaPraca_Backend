#!/usr/bin/env python3
"""
Skript na vytvorenie grafov najlepších 5 čŕt pre feature group 'combined'
porovnaním general a personal datasetov - správní pre diplomovú prácu.
Vertikálne stĺpcové grafy v slovenčine.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

# Nastavenie fontov pre slovenčinu
plt.rcParams['font.family'] = 'DejaVu Sans'

# ---------------------------------------------------------------------------
# Konfigurácia
# ---------------------------------------------------------------------------

RESULTS_BASE = os.path.join(
    os.path.dirname(__file__),
    "results",
    "exp2_RF_feature_group_comparison"
)

FEATURE_IMPORTANCE_FILE = os.path.join(
    RESULTS_BASE,
    "feature_importance_all_groups_general_vs_personal.csv"
)

OUTPUT_DIR = RESULTS_BASE
TOP_N = 5

# ---------------------------------------------------------------------------
# Načítanie a filtrowanie dát
# ---------------------------------------------------------------------------

print(f"Načítavam dáta z: {FEATURE_IMPORTANCE_FILE}")
df = pd.read_csv(FEATURE_IMPORTANCE_FILE)

# Filtrovanie len pre "combined" feature group
df_combined = df[df["feature_group"] == "combined"].copy()

if df_combined.empty:
    print("Chyba: Žiadne dáta pre 'combined' feature group!")
    exit(1)

# Oddelenie general a personal
df_general = df_combined[df_combined["dataset"] == "general"].copy()
df_personal = df_combined[df_combined["dataset"] == "personal"].copy()

# Získanie top 5 pre každý dataset
top_general = df_general.nlargest(TOP_N, "mean_importance")
top_personal = df_personal.nlargest(TOP_N, "mean_importance")

print(f"\nTop {TOP_N} čŕt pre GENERAL dataset:")
for idx, (_, row) in enumerate(top_general.iterrows(), start=1):
    print(f"  {idx}. {row['feature']}: {row['mean_importance']:.6f} (±{row['std_importance']:.6f})")

print(f"\nTop {TOP_N} čŕt pre PERSONAL dataset:")
for idx, (_, row) in enumerate(top_personal.iterrows(), start=1):
    print(f"  {idx}. {row['feature']}: {row['mean_importance']:.6f} (±{row['std_importance']:.6f})")

# ---------------------------------------------------------------------------
# Graf pre GENERAL dataset
# ---------------------------------------------------------------------------

fig_general, ax_general = plt.subplots(figsize=(10, 7))

x_pos_general = np.arange(len(top_general))
features_general = top_general["feature"].values
means_general = top_general["mean_importance"].values
stds_general = top_general["std_importance"].values

bars_general = ax_general.bar(x_pos_general, means_general, 
                               yerr=stds_general,
                               color="#4C72B0", 
                               capsize=6, 
                               alpha=0.85,
                               edgecolor="black",
                               linewidth=1.5)

ax_general.set_xticks(x_pos_general)
ax_general.set_xticklabels(features_general, fontsize=11, rotation=45, ha='right')
ax_general.set_ylabel("Priemerná dôležitosť čŕt", fontsize=12, fontweight="bold")
ax_general.set_xlabel("Črta", fontsize=12, fontweight="bold")
ax_general.set_title("Top 5 najdôležitejších čŕt\nVšeobecné dáta", 
                     fontsize=13, fontweight="bold", pad=20)
ax_general.yaxis.grid(True, linestyle="--", alpha=0.6)
ax_general.set_axisbelow(True)

# Pridanie hodnôt nad stĺpce
for i, (mean, std) in enumerate(zip(means_general, stds_general)):
    ax_general.text(i, mean + std + 0.0008, f"{mean:.4f}", 
                   ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
output_general = os.path.join(OUTPUT_DIR, "top5_features_combined_GENERAL.png")
plt.savefig(output_general, dpi=300, bbox_inches="tight")
print(f"\nGraf (GENERAL) uložený do: {output_general}")
plt.close()

# ---------------------------------------------------------------------------
# Graf pre PERSONAL dataset
# ---------------------------------------------------------------------------

fig_personal, ax_personal = plt.subplots(figsize=(10, 7))

x_pos_personal = np.arange(len(top_personal))
features_personal = top_personal["feature"].values
means_personal = top_personal["mean_importance"].values
stds_personal = top_personal["std_importance"].values

bars_personal = ax_personal.bar(x_pos_personal, means_personal, 
                                 yerr=stds_personal,
                                 color="#DD8452", 
                                 capsize=6, 
                                 alpha=0.85,
                                 edgecolor="black",
                                 linewidth=1.5)

ax_personal.set_xticks(x_pos_personal)
ax_personal.set_xticklabels(features_personal, fontsize=11, rotation=45, ha='right')
ax_personal.set_ylabel("Priemerná dôležitosť čŕt", fontsize=12, fontweight="bold")
ax_personal.set_xlabel("Črta", fontsize=12, fontweight="bold")
ax_personal.set_title("Top 5 najdôležitejších čŕt\nOsobná veta", 
                      fontsize=13, fontweight="bold", pad=20)
ax_personal.yaxis.grid(True, linestyle="--", alpha=0.6)
ax_personal.set_axisbelow(True)

# Pridanie hodnôt nad stĺpce
for i, (mean, std) in enumerate(zip(means_personal, stds_personal)):
    ax_personal.text(i, mean + std + 0.0008, f"{mean:.4f}", 
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
output_personal = os.path.join(OUTPUT_DIR, "top5_features_combined_PERSONAL.png")
plt.savefig(output_personal, dpi=300, bbox_inches="tight")
print(f"Graf (PERSONAL) uložený do: {output_personal}")
plt.close()

# ---------------------------------------------------------------------------
# Vytvorenie CSV súboru s top 5
# ---------------------------------------------------------------------------

top5_data = []

for idx, (_, row) in enumerate(top_general.iterrows(), start=1):
    top5_data.append({
        "dataset": "general",
        "poradie": idx,
        "črta": row["feature"],
        "priemerná_dôležitosť": row["mean_importance"],
        "smerodajná_odchýlka": row["std_importance"]
    })

for idx, (_, row) in enumerate(top_personal.iterrows(), start=1):
    top5_data.append({
        "dataset": "personal",
        "poradie": idx,
        "črta": row["feature"],
        "priemerná_dôležitosť": row["mean_importance"],
        "smerodajná_odchýlka": row["std_importance"]
    })

top5_df = pd.DataFrame(top5_data)
top5_csv_path = os.path.join(OUTPUT_DIR, "top5_features_combined_porovnanie_SK.csv")
top5_df.to_csv(top5_csv_path, index=False)
print(f"Top 5 tabuľka (slovenčina) uložená do: {top5_csv_path}")

print("\n" + "=" * 80)
print("Profesionálne grafy pre diplomovú prácu vytvorené!")
print("=" * 80)
