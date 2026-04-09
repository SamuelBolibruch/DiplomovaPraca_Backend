# Experimentálny skript na porovnanie modelov pre user-specific behaviorálnu biometrickú autentifikáciu.
# Experiment 1: Porovnanie RandomForest, SVM a XGBoost na fixnom scenári (10 train / 5 test legitímnych vzoriek).
# Threshold sa určuje z out-of-fold predikcií na train množine (nie z in-sample predikcií).

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Konfigurácia
# ---------------------------------------------------------------------------

TRAINING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "training")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "exp1_model_comparison")
RANDOM_STATE = 42

# Počet legitímnych vzoriek pre train / test
LEGIT_TRAIN = 10
LEGIT_TEST = 5

EXCLUDE_COLS = {"UserId", "RoundId", "label"}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Pomocné funkcie
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str):
    """Načíta CSV súbor a vráti feature matrix X a vektor labelov y."""
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"].values
    return X, y


def split_user_data(X, y, legit_train: int = LEGIT_TRAIN, legit_test: int = LEGIT_TEST):
    """
    Stratifikovaný split:
      - legit_train vzoriek triedy 1 → train
      - legit_test vzoriek triedy 1 → test
      - trieda 0 sa rozdelí v rovnakom pomere (train/test)
    Vracia (X_train, X_test, y_train, y_test) alebo None, ak nie je dostatok legitímnych vzoriek.
    """
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

    # Negatives – rovnaký pomer ako legitímni
    n_neg = len(neg_indices)
    ratio = legit_train / needed  # podiel trainu z celku
    n_neg_train = int(round(n_neg * ratio))
    n_neg_test = n_neg - n_neg_train

    perm_neg = rng.permutation(neg_indices)
    train_neg = perm_neg[:n_neg_train]
    test_neg = perm_neg[n_neg_train:n_neg_train + n_neg_test]

    train_idx = np.concatenate([train_legit, train_neg])
    test_idx = np.concatenate([test_legit, test_neg])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_arr = X.values
    return (
        X_arr[train_idx], X_arr[test_idx],
        y[train_idx], y[test_idx],
    )


def find_best_threshold_from_scores(oof_probs, y_train):
    """
    Hľadá threshold z out-of-fold skóre tak, aby abs(FAR - FRR) bolo minimálne.
    Vracia najlepší threshold (float).
    """
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


def get_oof_probabilities(model_name: str, X_train, y_train, n_splits: int = 5):
    """
    Vypočíta out-of-fold pravdepodobnosti pre triedu 1 pomocou StratifiedKFold.
    Ak má niektorá trieda menej vzoriek ako n_splits, n_splits sa automaticky zníži.
    Vracia pole OOF skóre rovnakej dĺžky ako X_train, alebo None ak CV nie je možné.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min())

    # Bezpečne znížiť n_splits
    effective_splits = min(n_splits, min_count)
    if effective_splits < 2:
        warnings.warn(
            f"[OOF] Model {model_name}: nedostatok vzoriek (min trieda má {min_count}) "
            f"pre StratifiedKFold. OOF nie je možné."
        )
        return None

    if effective_splits < n_splits:
        warnings.warn(
            f"[OOF] Model {model_name}: n_splits znížený z {n_splits} na {effective_splits} "
            f"(min trieda má {min_count} vzoriek)."
        )

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y_train), dtype=float)

    for train_idx, val_idx in skf.split(X_train, y_train):
        fold_model = get_model(model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_model.fit(X_train[train_idx], y_train[train_idx])
        oof_probs[val_idx] = fold_model.predict_proba(X_train[val_idx])[:, 1]

    return oof_probs


def compute_eer(probs, y_true):
    """
    Vypočíta EER pomocou threshold sweep – bod, kde abs(FAR - FRR) je minimálne.
    Vracia EER ako float.
    """
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


def _safe_confusion(y_true, y_pred):
    """Vráti (tn, fp, fn, tp) s istotou, že všetky 4 hodnoty existujú."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def compute_metrics(y_true, y_pred, probs):
    """
    Vypočíta Accuracy, FAR, FRR a EER na testovacích dátach.
    Vracia dict s metrikami.
    """
    tn, fp, fn, tp = _safe_confusion(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    eer = compute_eer(probs, y_true)
    return {"accuracy": acc, "far": far, "frr": frr, "eer": eer}


def get_model(model_name: str):
    """
    Vráti čerstvú inštanciu modelu podľa názvu.
    SVM je zabalené do Pipeline so StandardScaler.
    """
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_name == "SVM":
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

    elif model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    else:
        raise ValueError(f"Neznámy model: {model_name}")


def train_and_evaluate_model(model_name: str, X_train, X_test, y_train, y_test):
    """
    Workflow pre jedného používateľa a jeden model:
      1. OOF predikcie na train sete → user-specific threshold
      2. Natrénuj čerstvý model na celom train sete
      3. Predikuj na test sete
      4. FAR/FRR/Accuracy z thresholdu; EER score-based sweep na test sete
    Vracia dict s metrikami, alebo None ak OOF nie je možné.
    """
    # --- Krok 1: OOF predikcie a threshold ---
    oof_probs = get_oof_probabilities(model_name, X_train, y_train, n_splits=5)
    if oof_probs is None:
        return None

    threshold = find_best_threshold_from_scores(oof_probs, y_train)

    # --- Krok 2: Natrénuj čerstvý model na celom train sete ---
    model = get_model(model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    # --- Krok 3 & 4: Predikuj na test sete a vyhodnoť ---
    probs_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (probs_test >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred_test, probs_test)
    metrics["threshold"] = threshold
    return metrics


# ---------------------------------------------------------------------------
# Hlavná logika
# ---------------------------------------------------------------------------

def main():
    model_names = ["RandomForest", "SVM", "XGBoost"]

    csv_files = sorted(
        f for f in os.listdir(TRAINING_DIR)
        if f.startswith("training_") and f.endswith(".csv")
    )

    print(f"Nájdených {len(csv_files)} training súborov.\n")

    per_user_records = []

    for csv_file in csv_files:
        uid = csv_file.replace("training_", "").replace(".csv", "")
        csv_path = os.path.join(TRAINING_DIR, csv_file)

        X, y = load_dataset(csv_path)

        split = split_user_data(X, y)
        if split is None:
            n_legit = int((y == 1).sum())
            warnings.warn(
                f"[SKIP] User {uid}: iba {n_legit} legitímnych vzoriek "
                f"(potrebných {LEGIT_TRAIN + LEGIT_TEST}). Používateľ preskočený."
            )
            continue

        X_train, X_test, y_train, y_test = split

        print(f"User: {uid}  |  train: {len(y_train)} vzoriek  |  test: {len(y_test)} vzoriek")

        for model_name in model_names:
            metrics = train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test)

            if metrics is None:
                warnings.warn(
                    f"[SKIP] User {uid}, model {model_name}: OOF zlyhalo, výsledky preskočené."
                )
                continue

            record = {
                "user_id": uid,
                "model": model_name,
                "threshold": metrics["threshold"],
                "accuracy": metrics["accuracy"],
                "far": metrics["far"],
                "frr": metrics["frr"],
                "eer": metrics["eer"],
            }
            per_user_records.append(record)
            print(
                f"  {model_name:14s}  acc={metrics['accuracy']:.3f}  "
                f"FAR={metrics['far']:.3f}  FRR={metrics['frr']:.3f}  "
                f"EER={metrics['eer']:.3f}  thr={metrics['threshold']:.3f}"
            )

        print()

    # -----------------------------------------------------------------------
    # Uloženie per-user výsledkov
    # -----------------------------------------------------------------------
    if not per_user_records:
        print("Žiadne výsledky neboli vytvorené.")
        return

    per_user_df = pd.DataFrame(per_user_records)
    per_user_csv = os.path.join(RESULTS_DIR, "per_user_results.csv")
    per_user_df.to_csv(per_user_csv, index=False)
    print(f"Per-user výsledky uložené do: {per_user_csv}\n")

    # -----------------------------------------------------------------------
    # Súhrn – priemer a std cez používateľov pre každý model
    # -----------------------------------------------------------------------
    summary_records = []
    for model_name in model_names:
        subset = per_user_df[per_user_df["model"] == model_name]
        record = {"model": model_name}
        for metric in ["accuracy", "far", "frr", "eer"]:
            record[f"avg_{metric}"] = subset[metric].mean()
            record[f"std_{metric}"] = subset[metric].std()
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(RESULTS_DIR, "summary_results.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Súhrnné výsledky uložené do: {summary_csv}\n")

    # -----------------------------------------------------------------------
    # Výpis tabuľky do konzoly
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("SÚHRN – priemery a smerodajné odchýlky cez všetkých používateľov")
    print("=" * 80)
    header = f"{'Model':<16} {'Acc':>8} {'±':>6} {'FAR':>8} {'±':>6} {'FRR':>8} {'±':>6} {'EER':>8} {'±':>6}"
    print(header)
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(
            f"{row['model']:<16}"
            f" {row['avg_accuracy']:>8.4f} {row['std_accuracy']:>6.4f}"
            f" {row['avg_far']:>8.4f} {row['std_far']:>6.4f}"
            f" {row['avg_frr']:>8.4f} {row['std_frr']:>6.4f}"
            f" {row['avg_eer']:>8.4f} {row['std_eer']:>6.4f}"
        )
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Graf – AAR, FAR, FRR, EER pre každý model v jednom grouped bar charte
    # -----------------------------------------------------------------------
    metrics_to_plot = ["avg_accuracy", "avg_far", "avg_frr", "avg_eer"]
    metric_labels   = ["AAR (Accuracy)", "FAR", "FRR", "EER"]
    metric_colors   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    metric_std      = ["std_accuracy",  "std_far", "std_frr", "std_eer"]

    n_models  = len(summary_df)
    n_metrics = len(metrics_to_plot)
    bar_width = 0.18
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(10, 6))

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

    ax.set_title("Experiment 1 – porovnanie modelov: AAR, FAR, FRR, EER")
    ax.set_ylabel("Hodnota")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"])
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
