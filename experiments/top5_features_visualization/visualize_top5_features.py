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

FEATURE_IMPORTANCE_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "exp2_RF_feature_group_comparison",
    "feature_importance_all_groups_general_vs_personal.csv"
)

OUTPUT_DIR = os.path.dirname(__file__)
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

import numpy as np

n_features = len(df_general)
uniform_pct = (1.0 / n_features) * 100

# Zjednotenie čŕt z oboch datasetov (zachovanie poradia podľa general)
all_features = list(dict.fromkeys(
    list(top_general["feature"]) + list(top_personal["feature"])
))

gen_vals = [
    df_general[df_general["feature"] == f]["mean_importance"].values[0] * 100
    if f in df_general["feature"].values else 0.0
    for f in all_features
]
per_vals = [
    df_personal[df_personal["feature"] == f]["mean_importance"].values[0] * 100
    if f in df_personal["feature"].values else 0.0
    for f in all_features
]

x = np.arange(len(all_features))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))

bars_g = ax.bar(x - width / 2, gen_vals, width, label="General", color="#4C72B0", alpha=0.85)
bars_p = ax.bar(x + width / 2, per_vals, width, label="Personal", color="#DD8452", alpha=0.85)

# Baseline – uniformná dôležitosť
ax.axhline(uniform_pct, color="gray", linestyle="--", linewidth=1.2,
           label=f"Uniformná dôležitosť (1/{n_features} = {uniform_pct:.2f}%)")

# Hodnoty nad stĺpcami
for bar, val in zip(bars_g, gen_vals):
    if val > 0:
        ratio = val / uniform_pct
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.04,
                f"{val:.2f}%\n({ratio:.1f}×)",
                ha="center", va="bottom", fontsize=8, color="#4C72B0", fontweight="bold", linespacing=1.3)

for bar, val in zip(bars_p, per_vals):
    if val > 0:
        ratio = val / uniform_pct
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.04,
                f"{val:.2f}%\n({ratio:.1f}×)",
                ha="center", va="bottom", fontsize=8, color="#DD8452", fontweight="bold", linespacing=1.3)

ax.set_xticks(x)
ax.set_xticklabels(all_features, rotation=25, ha="right", fontsize=10)
ax.set_ylabel("Dôležitosť príznaku (%)", fontsize=11)
ax.set_title("Top 5 príznakov – General vs Personal (Combined group, 154 čŕt celkovo)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

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
