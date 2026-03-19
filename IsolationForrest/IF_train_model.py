import os
import pandas as pd
import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest


VECTORS_DIR = "data/vectors"
MODEL_DIR = "IsolationForrest/models"

RANDOM_STATE = 42
N_SPLITS = 20

# nastavenie modelu
N_ESTIMATORS = 300
CONTAMINATION = 0.08
MAX_SAMPLES = 12
MAX_FEATURES = 1.0

# počet impostor vzoriek od každého iného používateľa
IMPOSTOR_SAMPLES_PER_USER = 3

# feature selection
TOP_K_FEATURES = 30
MIN_FEATURE_STD = 1e-8

# threshold z genuine train score
TRAIN_SCORE_PERCENTILE = 10  # spodných 10% genuine train skóre bude reject

os.makedirs(MODEL_DIR, exist_ok=True)

files = [f for f in os.listdir(VECTORS_DIR) if f.startswith("vector_") and f.endswith(".csv")]

print(f"Nájdených {len(files)} vector súborov\n")

# načítaj všetky dáta dopredu
all_data = {}
for file in files:
    uid = file.replace("vector_", "").replace(".csv", "")
    df = pd.read_csv(os.path.join(VECTORS_DIR, file))
    all_data[uid] = df


def select_top_variable_features(X_train: pd.DataFrame, top_k: int, min_std: float):
    stds = X_train.std(axis=0, ddof=1).fillna(0.0)
    stds = stds[stds > min_std].sort_values(ascending=False)

    if len(stds) == 0:
        return X_train.columns.tolist()

    return stds.head(min(top_k, len(stds))).index.tolist()


for file in files:
    uid = file.replace("vector_", "").replace(".csv", "")
    df = all_data[uid]

    print(f"{'='*50}")
    print(f"User: {uid}")
    print(f"{'='*50}")

    feature_cols = [c for c in df.columns if c not in ["UserId", "RoundId"]]

    X_user = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(X_user) < 15:
        print(f"Preskakujem {uid} — má len {len(X_user)} vzoriek, treba aspoň 15.\n")
        continue

    splitter = ShuffleSplit(
        n_splits=N_SPLITS,
        test_size=3,
        random_state=RANDOM_STATE
    )

    results = []

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X_user), start=1):
        X_train_full = X_user.iloc[train_idx].copy()
        X_test_genuine_full = X_user.iloc[test_idx].copy()

        # impostor test dáta: náhodne 3 vzorky od každého iného usera
        impostor_parts = []
        for other_uid, other_df in all_data.items():
            if other_uid == uid:
                continue

            X_other = other_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

            n_take = min(IMPOSTOR_SAMPLES_PER_USER, len(X_other))
            X_other_sample = X_other.sample(
                n=n_take,
                random_state=RANDOM_STATE + split_idx
            )

            impostor_parts.append(X_other_sample)

        if len(impostor_parts) == 0:
            print(f"Preskakujem split {split_idx} — chýbajú impostor dáta.")
            continue

        X_impostor_full = pd.concat(impostor_parts, ignore_index=True)

        # -----------------------------
        # Feature selection iba z TRAIN genuine
        # -----------------------------
        selected_features = select_top_variable_features(
            X_train_full,
            top_k=TOP_K_FEATURES,
            min_std=MIN_FEATURE_STD
        )

        X_train = X_train_full[selected_features].copy()
        X_test_genuine = X_test_genuine_full[selected_features].copy()
        X_impostor = X_impostor_full[selected_features].copy()

        # -----------------------------
        # Robust scaling
        # -----------------------------
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_genuine_scaled = scaler.transform(X_test_genuine)
        X_impostor_scaled = scaler.transform(X_impostor)

        # -----------------------------
        # Isolation Forest
        # -----------------------------
        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            max_samples=MAX_SAMPLES,
            max_features=MAX_FEATURES,
            bootstrap=True,
            random_state=RANDOM_STATE + split_idx
        )

        model.fit(X_train_scaled)

        # -----------------------------
        # Threshold z genuine train score
        # vyššie score = viac genuine
        # -----------------------------
        train_scores = model.decision_function(X_train_scaled)
        threshold = np.percentile(train_scores, TRAIN_SCORE_PERCENTILE)

        genuine_scores = model.decision_function(X_test_genuine_scaled)
        impostor_scores = model.decision_function(X_impostor_scaled)

        pred_genuine = (genuine_scores >= threshold).astype(int)
        pred_impostor = (impostor_scores >= threshold).astype(int)

        tp = int(np.sum(pred_genuine == 1))
        fn = int(np.sum(pred_genuine == 0))

        fp = int(np.sum(pred_impostor == 1))
        tn = int(np.sum(pred_impostor == 0))

        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        results.append({
            "split": split_idx,
            "accuracy": acc,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "far": far,
            "frr": frr,
            "threshold": float(threshold),
            "n_features": len(selected_features),
            "n_train_pos": len(X_train),
            "n_test_pos": len(X_test_genuine),
            "n_test_neg": len(X_impostor),
        })

    results_df = pd.DataFrame(results)

    print("\nPer-split results:")
    print(results_df)

    print("\nAverage results:")
    print(results_df[["accuracy", "far", "frr"]].mean())

    print("\nStd results:")
    print(results_df[["accuracy", "far", "frr"]].std())
    print()