import os
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


RANDOM_STATE = 42

N_ESTIMATORS = 200
MAX_DEPTH = None
CLASS_WEIGHT = "balanced"

CONFIGS = [
    {
        "training_dir": "data/training",
        "model_dir": "RandomForrest/models"
    },
    {
        "training_dir": "data/training_personal",
        "model_dir": "RandomForrest/models_personal"
    }
]


def get_rf_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _safe_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


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

    return float(best_threshold)


def get_oof_probabilities(X_train, y_train, n_splits=5, label=""):
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min())

    effective_splits = min(n_splits, min_count)
    if effective_splits < 2:
        warnings.warn(
            f"[OOF] {label}: nedostatok vzoriek (min trieda má {min_count}) "
            "pre StratifiedKFold. OOF nie je možné."
        )
        return None

    if effective_splits < n_splits:
        warnings.warn(
            f"[OOF] {label}: n_splits znížený z {n_splits} na {effective_splits} "
            f"(min trieda má {min_count} vzoriek)."
        )

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y_train), dtype=float)

    for train_idx, val_idx in skf.split(X_train, y_train):
        fold_model = get_rf_model()
        fold_model.fit(X_train[train_idx], y_train[train_idx])
        oof_probs[val_idx] = fold_model.predict_proba(X_train[val_idx])[:, 1]

    return oof_probs

# ===============================
# LOOP cez oba datasety
# ===============================

for config in CONFIGS:
    TRAINING_DIR = config["training_dir"]
    MODEL_DIR = config["model_dir"]

    os.makedirs(MODEL_DIR, exist_ok=True)

    target_uid_filter = os.environ.get("TARGET_UID")
    files = [
        f for f in os.listdir(TRAINING_DIR)
        if f.startswith("training_") and f.endswith(".csv")
        and (not target_uid_filter or f == f"training_{target_uid_filter}.csv")
    ]

    print(f"\n{'#'*60}")
    print(f"Spracovávam training dir: {TRAINING_DIR}")
    print(f"Nájdených {len(files)} training súborov")
    print(f"{'#'*60}\n")

    for file in files:
        CSV_PATH = os.path.join(TRAINING_DIR, file)
        uid = file.replace("training_", "").replace(".csv", "")

        print(f"{'='*50}")
        print(f"User: {uid}")
        print(f"Dataset: {TRAINING_DIR}")
        print(f"{'='*50}")

        df = pd.read_csv(CSV_PATH)

        feature_cols = [c for c in df.columns if c not in ["UserId", "RoundId", "label"]]

        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df["label"]

        print("Dataset shape:", df.shape)
        print("Label distribution:")
        print(y.value_counts())

        # ===============================
        # TRAIN FINAL MODEL ON FULL DATA
        # ===============================

        X_np = X.values
        y_np = y.values

        oof_probs = get_oof_probabilities(X_np, y_np, n_splits=5, label=f"{TRAINING_DIR}/{uid}")
        if oof_probs is None:
            print(f"⚠ Preskakujem usera {uid}: OOF threshold nie je možné spočítať.")
            continue

        threshold = find_best_threshold_from_scores(oof_probs, y_np)

        model = get_rf_model()
        model.fit(X_np, y_np)

        MODEL_PATH = os.path.join(MODEL_DIR, f"model_{uid}.pkl")

        joblib.dump(
            {
                "model": model,
                "features": feature_cols,
                "threshold": threshold,
                "user_id": uid
            },
            MODEL_PATH
        )

        print(f"✓ Model uložený: {MODEL_PATH} | threshold={threshold:.3f}\n")

print("🔥 Hotovo! Všetky modely natrénované.")