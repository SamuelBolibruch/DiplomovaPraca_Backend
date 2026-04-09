# Experimentálny skript na porovnanie skupín čŕt pre user-specific behaviorálnu biometrickú autentifikáciu.
# Experiment 2 – SVM: Porovnanie 3 skupín vstupných čŕt (keystroke_only / sensor_only / combined)
#              pri fixnom modeli SVM a rovnakom scenári ako v experimente 1
#              (10 train / 5 test legitímnych vzoriek, OOF threshold).

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------------------------------------------------------------------------
# Konfigurácia
# ---------------------------------------------------------------------------

TRAINING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "training")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "exp2_SVM_feature_group_comparison")
RANDOM_STATE = 42

LEGIT_TRAIN = 10
LEGIT_TEST = 5

EXCLUDE_COLS = {"UserId", "RoundId", "label"}

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Definícia skupín čŕt
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
# Pomocné funkcie (identické s experimentom 1)
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    return df


def select_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        warnings.warn(f"  [WARN] Chýbajúce stĺpce (doplnené 0): {missing}")
    X = df[available].copy()
    for c in missing:
        X[c] = 0.0
    X = X[feature_cols]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"].values
    return X, y


def split_user_data(X, y, legit_train: int = LEGIT_TRAIN, legit_test: int = LEGIT_TEST):
    legit_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    n_legit = len(legit_indices)
    needed = legit_train + legit_test

    if n_legit < needed:
        return None

    rng = np.random.default_rng(RANDOM_STATE)
    perm_legit = rng.permutation(legit_indices)
    train_legit = perm_legit[:legit_train]
    test_legit = perm_legit[legit_train:legit_train + legit_test]

    n_neg = len(neg_indices)
    ratio = legit_train / needed
    n_neg_train = int(round(n_neg * ratio))

    perm_neg = rng.permutation(neg_indices)
    train_neg = perm_neg[:n_neg_train]
    test_neg = perm_neg[n_neg_train:]

    train_idx = np.concatenate([train_legit, train_neg])
    test_idx = np.concatenate([test_legit, test_neg])

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


def compute_eer(probs, y_true):
    candidates = np.linspace(0.0, 1.0, 201)
    best_eer = 1.0
    best_diff = float("inf")
    for thr in candidates:
        preds = (probs >= thr).astype(int)
        tn, fp, fn, tp = _safe_confusion(y_true, preds)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer = (far + frr) / 2.0
    return best_eer


def find_best_threshold_from_scores(oof_probs, y_train):
    candidates = np.linspace(0.0, 1.0, 201)
    best_threshold = 0.5
    best_diff = float("inf")
    for thr in candidates:
        preds = (oof_probs >= thr).astype(int)
        tn, fp, fn, tp = _safe_confusion(y_train, preds)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_threshold = thr
    return best_threshold


def get_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def get_oof_probabilities(X_train, y_train, n_splits: int = 5, group_name: str = ""):
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min())
    effective_splits = min(n_splits, min_count)

    if effective_splits < 2:
        warnings.warn(
            f"[OOF] {group_name}: nedostatok vzoriek (min trieda má {min_count}) "
            f"pre StratifiedKFold. OOF nie je možné."
        )
        return None

    if effective_splits < n_splits:
        warnings.warn(
            f"[OOF] {group_name}: n_splits znížený z {n_splits} na {effective_splits} "
            f"(min trieda má {min_count} vzoriek)."
        )

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y_train), dtype=float)

    for train_idx, val_idx in skf.split(X_train, y_train):
        fold_model = get_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_model.fit(X_train[train_idx], y_train[train_idx])
        oof_probs[val_idx] = fold_model.predict_proba(X_train[val_idx])[:, 1]

    return oof_probs


def compute_metrics(y_true, y_pred, probs):
    tn, fp, fn, tp = _safe_confusion(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    eer = compute_eer(probs, y_true)
    return {"accuracy": acc, "far": far, "frr": frr, "eer": eer}


def train_and_evaluate(group_name: str, X_train, X_test, y_train, y_test):
    oof_probs = get_oof_probabilities(X_train, y_train, n_splits=5, group_name=group_name)
    if oof_probs is None:
        return None

    threshold = find_best_threshold_from_scores(oof_probs, y_train)

    model = get_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    probs_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (probs_test >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred_test, probs_test)
    metrics["threshold"] = threshold
    return metrics


# ---------------------------------------------------------------------------
# Hlavná logika
# ---------------------------------------------------------------------------

def main():
    group_names = list(FEATURE_GROUPS.keys())

    csv_files = sorted(
        f for f in os.listdir(TRAINING_DIR)
        if f.startswith("training_") and f.endswith(".csv")
    )

    print(f"Nájdených {len(csv_files)} training súborov.\n")
    for gname, gcols in FEATURE_GROUPS.items():
        print(f"  {gname}: {len(gcols)} čŕt")
    print()

    per_user_records = []

    for csv_file in csv_files:
        uid = csv_file.replace("training_", "").replace(".csv", "")
        csv_path = os.path.join(TRAINING_DIR, csv_file)

        df = load_dataset(csv_path)

        y_full = df["label"].values
        legit_indices = np.where(y_full == 1)[0]
        n_legit = len(legit_indices)
        needed = LEGIT_TRAIN + LEGIT_TEST

        if n_legit < needed:
            warnings.warn(
                f"[SKIP] User {uid}: iba {n_legit} legitímnych vzoriek "
                f"(potrebných {needed}). Používateľ preskočený."
            )
            continue

        print(f"User: {uid}")

        for group_name, feature_cols in FEATURE_GROUPS.items():
            X, y = select_features(df, feature_cols)

            split = split_user_data(X, y)
            if split is None:
                continue

            X_train, X_test, y_train, y_test = split

            metrics = train_and_evaluate(group_name, X_train, X_test, y_train, y_test)

            if metrics is None:
                warnings.warn(
                    f"[SKIP] User {uid}, skupina {group_name}: OOF zlyhalo, výsledky preskočené."
                )
                continue

            record = {
                "user_id": uid,
                "feature_group": group_name,
                "n_features": len(feature_cols),
                "threshold": metrics["threshold"],
                "accuracy": metrics["accuracy"],
                "far": metrics["far"],
                "frr": metrics["frr"],
                "eer": metrics["eer"],
            }
            per_user_records.append(record)
            print(
                f"  {group_name:<20s}  acc={metrics['accuracy']:.3f}  "
                f"FAR={metrics['far']:.3f}  FRR={metrics['frr']:.3f}  "
                f"EER={metrics['eer']:.3f}  thr={metrics['threshold']:.3f}"
            )

        print()

    if not per_user_records:
        print("Žiadne výsledky neboli vytvorené.")
        return

    per_user_df = pd.DataFrame(per_user_records)
    per_user_csv = os.path.join(RESULTS_DIR, "per_user_results.csv")
    per_user_df.to_csv(per_user_csv, index=False)
    print(f"Per-user výsledky uložené do: {per_user_csv}\n")

    summary_records = []
    for group_name in group_names:
        subset = per_user_df[per_user_df["feature_group"] == group_name]
        record = {
            "feature_group": group_name,
            "n_features": FEATURE_GROUPS[group_name].__len__(),
        }
        for metric in ["accuracy", "far", "frr", "eer"]:
            record[f"avg_{metric}"] = subset[metric].mean()
            record[f"std_{metric}"] = subset[metric].std()
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(RESULTS_DIR, "summary_results.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Súhrnné výsledky uložené do: {summary_csv}\n")

    print("=" * 90)
    print("SÚHRN – priemery a smerodajné odchýlky cez všetkých používateľov")
    print("=" * 90)
    header = (
        f"{'Skupina čŕt':<22} {'#čŕt':>6} {'Acc':>8} {'±':>6} "
        f"{'FAR':>8} {'±':>6} {'FRR':>8} {'±':>6} {'EER':>8} {'±':>6}"
    )
    print(header)
    print("-" * 90)
    for _, row in summary_df.iterrows():
        print(
            f"{row['feature_group']:<22} {int(row['n_features']):>6}"
            f" {row['avg_accuracy']:>8.4f} {row['std_accuracy']:>6.4f}"
            f" {row['avg_far']:>8.4f} {row['std_far']:>6.4f}"
            f" {row['avg_frr']:>8.4f} {row['std_frr']:>6.4f}"
            f" {row['avg_eer']:>8.4f} {row['std_eer']:>6.4f}"
        )
    print("=" * 90)

    metrics_to_plot = ["avg_accuracy", "avg_far", "avg_frr", "avg_eer"]
    metric_labels   = ["AAR (Accuracy)", "FAR", "FRR", "EER"]
    metric_colors   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    metric_std      = ["std_accuracy",  "std_far", "std_frr", "std_eer"]

    n_groups  = len(summary_df)
    n_metrics = len(metrics_to_plot)
    bar_width = 0.18
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (metric, label, color, std_col) in enumerate(
        zip(metrics_to_plot, metric_labels, metric_colors, metric_std)
    ):
        offsets = x + (i - (n_metrics - 1) / 2) * bar_width
        bars = ax.bar(
            offsets, summary_df[metric], bar_width,
            label=label, color=color,
            yerr=summary_df[std_col], capsize=3,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    ax.set_title("Experiment 2 SVM – porovnanie skupín čŕt: AAR, FAR, FRR, EER (SVM)")
    ax.set_ylabel("Hodnota")
    ax.set_xlabel("Skupina čŕt")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["feature_group"])
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    plt.tight_layout()
    combined_plot = os.path.join(RESULTS_DIR, "metrics_comparison.png")
    plt.savefig(combined_plot, dpi=150)
    plt.close()
    print(f"\nGraf metrík uložený do: {combined_plot}")


if __name__ == "__main__":
    main()
