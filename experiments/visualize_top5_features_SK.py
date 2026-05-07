#!/usr/bin/env python3
"""
Vizualizácia feature importance z produkčných Random Forest modelov.
Generuje rovnakú štruktúru ako exp2, ale priamo z modelov v RandomForrest/.
Výstupy idú do experiments/top5_features_visualization/.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'

# ---------------------------------------------------------------------------
# Skupiny čŕt (rovnaké ako v exp2)
# ---------------------------------------------------------------------------

KEYSTROKE_FEATURES = [
    "cps", "wpm", "total_duration", "typing_efficiency", "error_rate",
    "backspace_count", "mean_corr_time", "mean_burst_size", "burst_count",
    "max_burst_size", "n_inserts", "n_dd_valid", "n_ft_valid", "n_uu_valid",
    "dd_valid_ratio", "ft_valid_ratio", "uu_valid_ratio", "backspace_per_char",
    "burst_count_per_char", "long_pause_per_char", "micro_pause_per_char",
    "space_pd", "space_ft", "vowel_pd", "consonant_pd", "special_pd",
    "capital_pd", "long_pause_count", "dd_cv_ratio", "dd_iqr",
    "micro_pause_count", "dd_trend_slope",
    "dd_vc_mean", "dd_vc_std", "dd_vc_median", "dd_vc_count",
    "dd_cv_mean", "dd_cv_std", "dd_cv_median", "dd_cv_count",
    "dd_cc_mean", "dd_cc_std", "dd_cc_median", "dd_cc_count",
    "dd_ls_mean", "dd_ls_std", "dd_ls_median", "dd_ls_count",
    "dd_sl_mean", "dd_sl_std", "dd_sl_median", "dd_sl_count",
    "pd_mean", "pd_std", "pd_median", "pd_min", "pd_max", "pd_skew", "pd_kurt",
    "ft_mean", "ft_std", "ft_median", "ft_min", "ft_max", "ft_skew", "ft_kurt",
    "dd_mean", "dd_std", "dd_median", "dd_min", "dd_max", "dd_skew", "dd_kurt",
    "uu_mean", "uu_std", "uu_median", "uu_min", "uu_max", "uu_skew", "uu_kurt",
]

SENSOR_FEATURES = [
    "acc_cov",
    "acc_mag_mean", "acc_mag_std", "acc_energy", "acc_rms", "acc_jerk_std",
    "acc_mag_median", "acc_mag_iqr", "acc_axis_corr_xy", "acc_axis_corr_xz",
    "acc_axis_corr_yz", "acc_mag_p95", "acc_mag_p99", "acc_mag_mad",
    "acc_mag_trend_slope", "acc_mag_spectral_entropy", "acc_planarity",
    "acc_cv", "acc_n_samples", "acc_duration_sec", "acc_dt_mean",
    "acc_dt_std", "acc_fs_hz",
    "gyro_cov",
    "gyro_mag_mean", "gyro_mag_std", "gyro_energy", "gyro_rms", "gyro_jerk_std",
    "gyro_mag_median", "gyro_mag_iqr", "gyro_axis_corr_xy", "gyro_axis_corr_xz",
    "gyro_axis_corr_yz", "gyro_mag_p95", "gyro_mag_p99", "gyro_mag_mad",
    "gyro_mag_trend_slope", "gyro_mag_spectral_entropy", "gyro_planarity",
    "gyro_cv", "gyro_n_samples", "gyro_duration_sec", "gyro_dt_mean",
    "gyro_dt_std", "gyro_fs_hz",
]

CROSS_MODAL_FEATURES = [
    "acc_auc_per_char_mean", "acc_peak_to_rms_ratio_mean",
    "acc_post_pre_energy_ratio_mean", "acc_rise_time_ms_mean",
    "acc_keystroke_xcorr_lag_ms", "acc_keystroke_xcorr_peak",
    "acc_peak_space_minus_letter", "acc_peak_per_char_mean",
    "acc_jerk_peak_per_char_mean", "acc_energy_per_char",
    "acc_peak_lag_mean_ms", "corr_dd_acc_energy", "fast_slow_acc_ratio",
    "gyro_auc_per_char_mean", "gyro_peak_to_rms_ratio_mean",
    "gyro_post_pre_energy_ratio_mean", "gyro_rise_time_ms_mean",
    "gyro_keystroke_xcorr_lag_ms", "gyro_keystroke_xcorr_peak",
    "gyro_peak_space_minus_letter", "gyro_peak_per_char_mean",
    "gyro_jerk_peak_per_char_mean", "gyro_energy_per_char",
    "gyro_peak_lag_mean_ms", "corr_dd_gyro_energy", "fast_slow_gyro_ratio",
    "corr_acc_gyro_peak_per_char", "gyro_to_acc_energy_ratio_per_char_mean",
]

FEATURE_GROUPS = {
    "keystroke_only": KEYSTROKE_FEATURES,
    "sensor_only": SENSOR_FEATURES,
    "combined": KEYSTROKE_FEATURES + SENSOR_FEATURES + CROSS_MODAL_FEATURES,
}

# ---------------------------------------------------------------------------
# Konfigurácia
# ---------------------------------------------------------------------------

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_GENERAL = os.path.join(ROOT, "RandomForrest", "models")
MODELS_PERSONAL = os.path.join(ROOT, "RandomForrest", "models_personal")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "top5_features_visualization")
TOP_N = 5

# ---------------------------------------------------------------------------
# Načítanie feature importance z produkčných modelov
# ---------------------------------------------------------------------------

def load_importances_per_feature(models_dir: str) -> dict:
    """Vráti dict {feature: [importance_user1, importance_user2, ...]}."""
    records = {}
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    print(f"Načítavam {len(model_files)} modelov z: {models_dir}")
    for fname in model_files:
        data = joblib.load(os.path.join(models_dir, fname))
        for feat, imp in zip(data["features"], data["model"].feature_importances_):
            records.setdefault(feat, []).append(imp)
    return records


def importance_df_for_group(records: dict, features: list) -> pd.DataFrame:
    """Vytvorí DataFrame s mean/std importance pre danú skupinu čŕt."""
    rows = []
    for feat in features:
        vals = records.get(feat, [])
        if vals:
            rows.append({
                "feature": feat,
                "mean_importance": np.mean(vals),
                "std_importance": np.std(vals),
            })
    df = pd.DataFrame(rows).sort_values("mean_importance", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def save_importance_graph(df: pd.DataFrame, title: str, out_path: str, top_n: int = 20):
    """Horizontálny bar chart feature importance (štýl exp2)."""
    top = df.head(top_n)
    fig_h = max(5, top_n * 0.35)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y_pos = np.arange(len(top))
    means = top["mean_importance"].values[::-1]
    stds = top["std_importance"].values[::-1]
    names = top["feature"].values[::-1]
    ax.barh(y_pos, means, xerr=stds, align="center", color="#4C72B0", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Priemerná dôležitosť (Gini)")
    ax.set_title(title)
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def process_dataset(models_dir: str, dataset_name: str):
    records = load_importances_per_feature(models_dir)

    all_rows = []
    group_dfs = {}

    for group_name, features in FEATURE_GROUPS.items():
        df = importance_df_for_group(records, features)
        group_dfs[group_name] = df

        csv_path = os.path.join(OUTPUT_DIR, f"feature_importance_{group_name}_{dataset_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  CSV uložené: {csv_path}")

        png_path = os.path.join(OUTPUT_DIR, f"feature_importance_{group_name}_{dataset_name}.png")
        save_importance_graph(df, f"Top 20 čŕt [{group_name}] – {dataset_name}", png_path)
        print(f"  Graf uložený: {png_path}")

        all_df = df.copy()
        all_df.insert(0, "feature_group", group_name)
        all_df.insert(0, "dataset", dataset_name)
        all_rows.append(all_df)

    all_groups_df = pd.concat(all_rows, ignore_index=True)
    all_csv = os.path.join(OUTPUT_DIR, f"feature_importance_all_groups_{dataset_name}.csv")
    all_groups_df.to_csv(all_csv, index=False)
    print(f"  All-groups CSV: {all_csv}")

    return group_dfs["combined"], all_groups_df


# ---------------------------------------------------------------------------
# Spracovanie oboch datasetov
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n=== GENERAL ===")
df_general_combined, all_general = process_dataset(MODELS_GENERAL, "general")

print("\n=== PERSONAL ===")
df_personal_combined, all_personal = process_dataset(MODELS_PERSONAL, "personal")

# Súhrnný CSV cez oba datasety
all_vs = pd.concat([all_general, all_personal], ignore_index=True)
all_vs.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_all_groups_general_vs_personal.csv"), index=False)

# ---------------------------------------------------------------------------
# Top 5 grafy (vertikálne, slovenčina) – combined skupina
# ---------------------------------------------------------------------------

top_general = df_general_combined.head(TOP_N)
top_personal = df_personal_combined.head(TOP_N)

print(f"\nTop {TOP_N} čŕt pre GENERAL dataset:")
for _, row in top_general.iterrows():
    print(f"  {int(row['rank'])}. {row['feature']}: {row['mean_importance']:.6f} (±{row['std_importance']:.6f})")

print(f"\nTop {TOP_N} čŕt pre PERSONAL dataset:")
for _, row in top_personal.iterrows():
    print(f"  {int(row['rank'])}. {row['feature']}: {row['mean_importance']:.6f} (±{row['std_importance']:.6f})")


def plot_top5_vertical(top_df: pd.DataFrame, title: str, color: str, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 7))
    x_pos = np.arange(len(top_df))
    means = top_df["mean_importance"].values
    stds = top_df["std_importance"].values
    features = top_df["feature"].values
    ax.bar(x_pos, means, yerr=stds, color=color, capsize=6, alpha=0.85,
           edgecolor="black", linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features, fontsize=11, rotation=45, ha='right')
    ax.set_ylabel("Priemerná dôležitosť čŕt", fontsize=12, fontweight="bold")
    ax.set_xlabel("Črta", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    for i, mean in enumerate(means):
        ax.text(i, mean / 2, f"{mean:.4f}",
                ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


plot_top5_vertical(top_general, "Top 5 najdôležitejších čŕt\nVšeobecné dáta", "#4C72B0",
                   os.path.join(OUTPUT_DIR, "top5_features_combined_GENERAL.png"))
print(f"\nGraf (GENERAL) uložený.")

plot_top5_vertical(top_personal, "Top 5 najdôležitejších čŕt\nOsobná veta", "#DD8452",
                   os.path.join(OUTPUT_DIR, "top5_features_combined_PERSONAL.png"))
print(f"Graf (PERSONAL) uložený.")

# ---------------------------------------------------------------------------
# Kombinovaný porovnávací graf (general vs personal vedľa seba)
# ---------------------------------------------------------------------------

n_features = len(df_general_combined)
uniform_pct = (1.0 / n_features) * 100

all_features = list(dict.fromkeys(
    list(top_general["feature"]) + list(top_personal["feature"])
))

gen_vals = [
    df_general_combined[df_general_combined["feature"] == f]["mean_importance"].values[0] * 100
    if f in df_general_combined["feature"].values else 0.0
    for f in all_features
]
per_vals = [
    df_personal_combined[df_personal_combined["feature"] == f]["mean_importance"].values[0] * 100
    if f in df_personal_combined["feature"].values else 0.0
    for f in all_features
]

x = np.arange(len(all_features))
width = 0.35
fig, ax = plt.subplots(figsize=(13, 6))
bars_g = ax.bar(x - width / 2, gen_vals, width, label="Všeobecné dáta", color="#4C72B0", alpha=0.85)
bars_p = ax.bar(x + width / 2, per_vals, width, label="Osobná veta", color="#DD8452", alpha=0.85)
ax.axhline(uniform_pct, color="gray", linestyle="--", linewidth=1.2,
           label=f"Uniformná dôležitosť (1/{n_features} = {uniform_pct:.2f}%)")

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
ax.set_title(f"Top 5 príznakov – Všeobecné vs Osobné dáta ({n_features} čŕt celkovo)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top5_features_combined_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Graf (porovnanie) uložený.")

# ---------------------------------------------------------------------------
# CSV – top 5
# ---------------------------------------------------------------------------

top5_data = []
for _, row in top_general.iterrows():
    top5_data.append({"dataset": "general", "rank": int(row["rank"]), "feature": row["feature"],
                      "mean_importance": row["mean_importance"], "std_importance": row["std_importance"]})
for _, row in top_personal.iterrows():
    top5_data.append({"dataset": "personal", "rank": int(row["rank"]), "feature": row["feature"],
                      "mean_importance": row["mean_importance"], "std_importance": row["std_importance"]})

pd.DataFrame(top5_data).to_csv(os.path.join(OUTPUT_DIR, "top5_features_combined_comparison.csv"), index=False)

print("\n" + "=" * 80)
print("Všetky grafy a CSV súbory vytvorené!")
print("=" * 80)
