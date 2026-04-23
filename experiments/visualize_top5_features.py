#!/usr/bin/env python3
"""
Skript na vytvorenie grafu najlepších 5 čŕt pre feature group 'combined'
porovnaním general a personal datasetov.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

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
# Načítanie a filtrование dát
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
# Vytvorenie grafu
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# General dataset
ax_general = axes[0]
y_pos_general = range(len(top_general))
features_general = top_general["feature"].values[::-1]
means_general = top_general["mean_importance"].values[::-1]
stds_general = top_general["std_importance"].values[::-1]

bars_general = ax_general.barh(y_pos_general, means_general, xerr=stds_general,
                                 color="#4C72B0", capsize=5, alpha=0.8)
ax_general.set_yticks(y_pos_general)
ax_general.set_yticklabels(features_general, fontsize=10)
ax_general.set_xlabel("Mean Importance", fontsize=11)
ax_general.set_title("Top 5 Features – GENERAL Dataset (Combined)", fontsize=12, fontweight="bold")
ax_general.xaxis.grid(True, linestyle="--", alpha=0.5)
ax_general.set_axisbelow(True)

# Pridanie hodnôt na grafy
for i, (mean, std) in enumerate(zip(means_general, stds_general)):
    ax_general.text(mean + std + 0.0005, i, f"{mean:.4f}", va="center", fontsize=9)

# Personal dataset
ax_personal = axes[1]
y_pos_personal = range(len(top_personal))
features_personal = top_personal["feature"].values[::-1]
means_personal = top_personal["mean_importance"].values[::-1]
stds_personal = top_personal["std_importance"].values[::-1]

bars_personal = ax_personal.barh(y_pos_personal, means_personal, xerr=stds_personal,
                                   color="#DD8452", capsize=5, alpha=0.8)
ax_personal.set_yticks(y_pos_personal)
ax_personal.set_yticklabels(features_personal, fontsize=10)
ax_personal.set_xlabel("Mean Importance", fontsize=11)
ax_personal.set_title("Top 5 Features – PERSONAL Dataset (Combined)", fontsize=12, fontweight="bold")
ax_personal.xaxis.grid(True, linestyle="--", alpha=0.5)
ax_personal.set_axisbelow(True)

# Pridanie hodnôt na grafy
for i, (mean, std) in enumerate(zip(means_personal, stds_personal)):
    ax_personal.text(mean + std + 0.0005, i, f"{mean:.4f}", va="center", fontsize=9)

fig.suptitle("Experiment 2 RF – Top 5 Features Comparison: General vs Personal (Combined Group)",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "top5_features_combined_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\nGraf uložený do: {output_path}")
plt.close()

# ---------------------------------------------------------------------------
# Vytvorenie CSV súboru s top 5
# ---------------------------------------------------------------------------

top5_data = []

for idx, (_, row) in enumerate(top_general.iterrows(), start=1):
    top5_data.append({
        "dataset": "general",
        "rank": idx,
        "feature": row["feature"],
        "mean_importance": row["mean_importance"],
        "std_importance": row["std_importance"]
    })

for idx, (_, row) in enumerate(top_personal.iterrows(), start=1):
    top5_data.append({
        "dataset": "personal",
        "rank": idx,
        "feature": row["feature"],
        "mean_importance": row["mean_importance"],
        "std_importance": row["std_importance"]
    })

top5_df = pd.DataFrame(top5_data)
top5_csv_path = os.path.join(OUTPUT_DIR, "top5_features_combined_comparison.csv")
top5_df.to_csv(top5_csv_path, index=False)
print(f"Top 5 tabuľka uložená do: {top5_csv_path}")

print("\n" + "=" * 80)
print("Vizualizácia najlepších 5 čŕt pre 'combined' grupu hotová!")
print("=" * 80)
