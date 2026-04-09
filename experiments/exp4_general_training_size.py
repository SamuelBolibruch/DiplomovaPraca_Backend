# Experimentálny skript na analýzu vplyvu počtu tréningových vzoriek na výkon modelu.
# Experiment 4 – General – RandomForest: Vplyv veľkosti trénovacej množiny (TRAIN_SIZE) na výkon
#              pri fixed modeli RandomForest, combined feature sete a zdieľanom (general) texte (data/training).
# Metodika identická s experimentmi 1, 2 a 3:
#   - user-specific OOF threshold (StratifiedKFold, 5 foldov)
#   - minimalizácia abs(FAR - FRR)

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Konfigurácia
# ---------------------------------------------------------------------------

TRAINING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "training")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results", "exp4_general_training_size_analysis")
RANDOM_STATE = 42

TRAIN_SIZES = [3, 5, 8, 10, 12]
LEGIT_TEST_FIXED = 5  # minimálny počet legitímnych vzoriek rezervovaných pre test

EXCLUDE_COLS = {"UserId", "RoundId", "label"}

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Combined feature set (rovnaký ako v experimentoch 2 a 3)
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
    "acc_mag_mean", "acc_mag_std", "acc_energy", "acc_rms", "acc_jerk_std",
    "acc_mag_median", "acc_mag_iqr", "acc_axis_corr_xy", "acc_axis_corr_xz",
    "acc_axis_corr_yz", "acc_mag_p95", "acc_mag_p99", "acc_mag_mad",
    "acc_mag_trend_slope", "acc_mag_spectral_entropy", "acc_planarity",
    "acc_cv", "acc_n_samples", "acc_duration_sec", "acc_dt_mean", "acc_dt_std",
    "acc_fs_hz",
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

COMBINED_ALL_FEATURES = KEYSTROKE_FEATURES + SENSOR_FEATURES + CROSS_MODAL_FEATURES

# ---------------------------------------------------------------------------
# Pomocné funkcie
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def select_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        warnings.warn(f"  [WARN] Chýbajúce stĺpce (doplnené 0): {missing}")
    X = df[available].copy()
    for c in missing:
        X[c] = 0.0
    X = X[feature_cols]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"].values
    return X, y


def split_user_data(X, y, legit_train: int, legit_test: int):
    """
    Stratifikovaný split:
      - legit_train vzoriek triedy 1 → train
      - legit_test vzoriek triedy 1 → test (zvyšok po train)
      - trieda 0 rozdelená v rovnakom pomere
    Vracia (X_train, X_test, y_train, y_test) alebo None ak nie je dostatok vzoriek.
    """
    legit_indices = np.where(y == 1)[0]
    neg_indices   = np.where(y == 0)[0]

    n_legit = len(legit_indices)
    needed  = legit_train + legit_test

    if n_legit < needed:
        return None

    rng = np.random.default_rng(RANDOM_STATE)
    perm_legit  = rng.permutation(legit_indices)
    train_legit = perm_legit[:legit_train]
    test_legit  = perm_legit[legit_train:legit_train + legit_test]

    n_neg       = len(neg_indices)
    ratio       = legit_train / needed
    n_neg_train = int(round(n_neg * ratio))

    perm_neg  = rng.permutation(neg_indices)
    train_neg = perm_neg[:n_neg_train]
    test_neg  = perm_neg[n_neg_train:]

    train_idx = np.concatenate([train_legit, train_neg])
    test_idx  = np.concatenate([test_legit, test_neg])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_arr = X.values
    return (
        X_arr[train_idx], X_arr[test_idx],
        y[train_idx], y[test_idx],
    )


def _safe_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def compute_eer(probs, y_true) -> float:
    candidates = np.linspace(0.0, 1.0, 201)
    best_eer   = 1.0
    best_diff  = float("inf")
    for thr in candidates:
        preds = (probs >= thr).astype(int)
        tn, fp, fn, tp = _safe_confusion(y_true, preds)
        far  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer  = (far + frr) / 2.0
    return best_eer


def find_best_threshold_from_scores(oof_probs, y_train) -> float:
    candidates     = np.linspace(0.0, 1.0, 201)
    best_threshold = 0.5
    best_diff      = float("inf")
    for thr in candidates:
        preds = (oof_probs >= thr).astype(int)
        tn, fp, fn, tp = _safe_confusion(y_train, preds)
        far  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff      = diff
            best_threshold = thr
    return best_threshold


def compute_metrics(y_true, y_pred, probs) -> dict:
    tn, fp, fn, tp = _safe_confusion(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    eer = compute_eer(probs, y_true)
    return {"accuracy": acc, "far": far, "frr": frr, "eer": eer}


def get_rf_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_oof_probabilities(X_train, y_train, n_splits: int = 5, label: str = ""):
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min())
    effective_splits = min(n_splits, min_count)

    if effective_splits < 2:
        warnings.warn(
            f"[OOF] {label}: nedostatok vzoriek (min trieda má {min_count}) "
            f"pre StratifiedKFold. OOF nie je možné."
        )
        return None

    if effective_splits < n_splits:
        warnings.warn(
            f"[OOF] {label}: n_splits znížený z {n_splits} na {effective_splits} "
            f"(min trieda má {min_count} vzoriek)."
        )

    skf       = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y_train), dtype=float)

    for train_idx, val_idx in skf.split(X_train, y_train):
        fold_model = get_rf_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_model.fit(X_train[train_idx], y_train[train_idx])
        oof_probs[val_idx] = fold_model.predict_proba(X_train[val_idx])[:, 1]

    return oof_probs


def train_and_evaluate(X_train, X_test, y_train, y_test, label: str = ""):
    oof_probs = get_oof_probabilities(X_train, y_train, n_splits=5, label=label)
    if oof_probs is None:
        return None

    threshold = find_best_threshold_from_scores(oof_probs, y_train)

    model = get_rf_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    probs_test  = model.predict_proba(X_test)[:, 1]
    y_pred_test = (probs_test >= threshold).astype(int)

    metrics              = compute_metrics(y_test, y_pred_test, probs_test)
    metrics["threshold"] = threshold
    return metrics


def get_csv_files(directory: str) -> list:
    if not os.path.isdir(directory):
        warnings.warn(f"[WARN] Priečinok neexistuje: {directory}")
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.startswith("training_") and f.endswith(".csv")
    )


# ---------------------------------------------------------------------------
# Grafy
# ---------------------------------------------------------------------------

def plot_eer_vs_train_size(summary_df: pd.DataFrame, out_dir: str):
    """Čiarový graf EER v závislosti od train_size."""
    x            = summary_df["train_size"].tolist()
    avg_eer_vals = summary_df["avg_eer"].tolist()
    std_eer_vals = summary_df["std_eer"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x, avg_eer_vals, yerr=std_eer_vals,
        marker="o", linewidth=2, capsize=5,
        color="#4C72B0", label="EER",
    )
    for xi, yi in zip(x, avg_eer_vals):
        ax.text(xi, yi + 0.005, f"{yi:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Experiment 4 – EER vs počet tréningových vzoriek (RandomForest)")
    ax.set_ylabel("EER")
    ax.set_xlabel("Počet legitímnych tréningových vzoriek")
    ax.set_xticks(x)
    ax.set_ylim(0, min(1.05, max(avg_eer_vals) * 1.5 + 0.05))
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "eer_vs_train_size.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Graf EER uložený do: {out_path}")


def plot_far_frr_vs_train_size(summary_df: pd.DataFrame, out_dir: str):
    """Čiarový graf FAR a FRR v závislosti od train_size."""
    x        = summary_df["train_size"].tolist()
    avg_fars = summary_df["avg_far"].tolist()
    std_fars = summary_df["std_far"].tolist()
    avg_frrs = summary_df["avg_frr"].tolist()
    std_frrs = summary_df["std_frr"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x, avg_fars, yerr=std_fars,
        marker="o", linewidth=2, capsize=5,
        color="#DD8452", label="FAR",
    )
    ax.errorbar(
        x, avg_frrs, yerr=std_frrs,
        marker="s", linewidth=2, capsize=5,
        color="#55A868", label="FRR",
    )
    for xi, fi, ri in zip(x, avg_fars, avg_frrs):
        ax.text(xi, fi + 0.005, f"{fi:.3f}", ha="center", va="bottom", fontsize=8, color="#DD8452")
        ax.text(xi, ri - 0.018, f"{ri:.3f}", ha="center", va="bottom", fontsize=8, color="#55A868")

    ax.set_title("Experiment 4 – FAR a FRR vs počet tréningových vzoriek (RandomForest)")
    ax.set_ylabel("Hodnota")
    ax.set_xlabel("Počet legitímnych tréningových vzoriek")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "far_frr_vs_train_size.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Graf FAR/FRR uložený do: {out_path}")


# ---------------------------------------------------------------------------
# Hlavná logika
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Experiment 4 General – RandomForest: vplyv veľkosti trénovacej množiny (shared text)")
    print(f"Dáta: {TRAINING_DIR}")
    print(f"Feature set: combined (keystroke + sensor + cross-modal)")
    print(f"Počet čŕt: {len(COMBINED_ALL_FEATURES)}")
    print(f"  - Keystroke:   {len(KEYSTROKE_FEATURES)}")
    print(f"  - Sensor:      {len(SENSOR_FEATURES)}")
    print(f"  - Cross-modal: {len(CROSS_MODAL_FEATURES)}")
    print(f"Testované TRAIN_SIZES: {TRAIN_SIZES}")
    print("=" * 80)
    print()

    csv_files = get_csv_files(TRAINING_DIR)
    if not csv_files:
        print("Žiadne training súbory nenájdené. Ukončujem.")
        return

    print(f"Nájdených {len(csv_files)} training súborov.\n")

    per_user_records = []

    for train_size in TRAIN_SIZES:
        # Pre každého používateľa potrebujeme: train_size legit train + aspoň LEGIT_TEST_FIXED legit test
        needed = train_size + LEGIT_TEST_FIXED

        print(f"{'=' * 80}")
        print(f"TRAIN_SIZE = {train_size}  (test legit: zvyšok, min {LEGIT_TEST_FIXED})")
        print(f"{'=' * 80}")

        for csv_file in csv_files:
            uid      = csv_file.replace("training_", "").replace(".csv", "")
            csv_path = os.path.join(TRAINING_DIR, csv_file)

            df = load_dataset(csv_path)

            y_full  = df["label"].values
            n_legit = int((y_full == 1).sum())

            if n_legit < needed:
                warnings.warn(
                    f"[SKIP] User {uid}, train_size={train_size}: "
                    f"iba {n_legit} legitímnych vzoriek (potrebných {needed}). Preskočený."
                )
                continue

            X, y = select_features(df, COMBINED_ALL_FEATURES)

            # Počet test legit = zvyšok po train (ale aspoň LEGIT_TEST_FIXED)
            legit_test = n_legit - train_size

            split = split_user_data(X, y, legit_train=train_size, legit_test=legit_test)

            if split is None:
                warnings.warn(
                    f"[SKIP] User {uid}, train_size={train_size}: split zlyhal. Preskočený."
                )
                continue

            X_train, X_test, y_train, y_test = split

            print(
                f"  User: {uid:<30s}  "
                f"train: {len(y_train):3d} (legit={train_size})  "
                f"test: {len(y_test):3d} (legit={legit_test})"
            )

            metrics = train_and_evaluate(
                X_train, X_test, y_train, y_test,
                label=f"{uid} train_size={train_size}",
            )

            if metrics is None:
                warnings.warn(
                    f"[SKIP] User {uid}, train_size={train_size}: OOF zlyhalo. Preskočený."
                )
                continue

            print(
                f"    acc={metrics['accuracy']:.3f}  "
                f"FAR={metrics['far']:.3f}  "
                f"FRR={metrics['frr']:.3f}  "
                f"EER={metrics['eer']:.3f}  "
                f"thr={metrics['threshold']:.3f}"
            )

            per_user_records.append({
                "user_id":    uid,
                "train_size": train_size,
                "threshold":  metrics["threshold"],
                "accuracy":   metrics["accuracy"],
                "far":        metrics["far"],
                "frr":        metrics["frr"],
                "eer":        metrics["eer"],
            })

        print()

    # -----------------------------------------------------------------------
    # Uloženie per-user výsledkov
    # -----------------------------------------------------------------------
    if not per_user_records:
        print("Žiadne výsledky neboli vytvorené.")
        return

    per_user_df  = pd.DataFrame(per_user_records)
    per_user_csv = os.path.join(RESULTS_DIR, "per_user_results.csv")
    per_user_df.to_csv(per_user_csv, index=False)
    print(f"Per-user výsledky uložené do: {per_user_csv}\n")

    # -----------------------------------------------------------------------
    # Súhrn
    # -----------------------------------------------------------------------
    summary_records = []
    for train_size in TRAIN_SIZES:
        subset = per_user_df[per_user_df["train_size"] == train_size]
        if subset.empty:
            continue
        record = {"train_size": train_size}
        for metric in ["accuracy", "far", "frr", "eer"]:
            record[f"avg_{metric}"] = subset[metric].mean()
            record[f"std_{metric}"] = subset[metric].std()
        record["n_users"] = len(subset)
        summary_records.append(record)

    summary_df  = pd.DataFrame(summary_records)
    summary_csv = os.path.join(RESULTS_DIR, "summary_results.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Súhrnné výsledky uložené do: {summary_csv}\n")

    # -----------------------------------------------------------------------
    # Výpis tabuľky do konzoly
    # -----------------------------------------------------------------------
    print("=" * 95)
    print("SÚHRN – RandomForest – vplyv veľkosti trénovacej množiny – priemery ± std")
    print("=" * 95)
    header = (
        f"{'train_size':>10} {'N':>4} "
        f"{'Acc':>8} {'±':>6} "
        f"{'FAR':>8} {'±':>6} "
        f"{'FRR':>8} {'±':>6} "
        f"{'EER':>8} {'±':>6}"
    )
    print(header)
    print("-" * 95)
    for _, row in summary_df.iterrows():
        print(
            f"{int(row['train_size']):>10} {int(row['n_users']):>4}"
            f" {row['avg_accuracy']:>8.4f} {row['std_accuracy']:>6.4f}"
            f" {row['avg_far']:>8.4f} {row['std_far']:>6.4f}"
            f" {row['avg_frr']:>8.4f} {row['std_frr']:>6.4f}"
            f" {row['avg_eer']:>8.4f} {row['std_eer']:>6.4f}"
        )
    print("=" * 95)

    # -----------------------------------------------------------------------
    # Grafy
    # -----------------------------------------------------------------------
    plot_eer_vs_train_size(summary_df, RESULTS_DIR)
    plot_far_frr_vs_train_size(summary_df, RESULTS_DIR)

    print()
    print("=" * 80)
    print("Použitý feature set: combined")
    print("\nExperiment 4 General dokončený.")


if __name__ == "__main__":
    main()
