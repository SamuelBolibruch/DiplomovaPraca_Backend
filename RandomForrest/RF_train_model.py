import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


TRAINING_DIR = "data/training"
MODEL_DIR = "RandomForrest/models"

RANDOM_STATE = 42
N_SPLITS = 20

TEST_SIZE = 0.20
N_ESTIMATORS = 100
MAX_DEPTH = None
CLASS_WEIGHT = "balanced"
THRESHOLD = 0.2


os.makedirs(MODEL_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)

    return tn, fp, fn, tp, far, frr, acc


# ===============================
# LOOP cez všetky training súbory
# ===============================

files = [f for f in os.listdir(TRAINING_DIR) if f.startswith("training_") and f.endswith(".csv")]

print(f"Nájdených {len(files)} training súborov\n")

for file in files:
    CSV_PATH = os.path.join(TRAINING_DIR, file)

    # extrahuj UID zo súboru
    uid = file.replace("training_", "").replace(".csv", "")

    print(f"{'='*50}")
    print(f"User: {uid}")
    print(f"{'='*50}")

    df = pd.read_csv(CSV_PATH)

    feature_cols = [c for c in df.columns if c not in ["UserId", "RoundId", "label"]]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"]

    print("Dataset shape:", df.shape)
    print("Label distribution:")
    print(y.value_counts())

    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    results = []

    for split_idx, (train_idx, test_idx) in enumerate(sss.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE + split_idx,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        y_pred = (probs >= THRESHOLD).astype(int)

        tn, fp, fn, tp, far, frr, acc = compute_metrics(y_test, y_pred)

        results.append({
            "accuracy": acc,
            "far": far,
            "frr": frr
        })

    results_df = pd.DataFrame(results)

    print("\nAverage:")
    print(results_df.mean())

    print("\nStd:")
    print(results_df.std())

    # ===============================
    # FINAL MODEL
    # ===============================

    final_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    final_model.fit(X, y)

    MODEL_PATH = os.path.join(MODEL_DIR, f"model_{uid}.pkl")

    joblib.dump(
        {
            "model": final_model,
            "features": feature_cols,
            "threshold": THRESHOLD,
            "user_id": uid
        },
        MODEL_PATH
    )

    print(f"\n✓ Model uložený: {MODEL_PATH}\n")


print("\n🔥 Hotovo! Všetky modely natrénované.")