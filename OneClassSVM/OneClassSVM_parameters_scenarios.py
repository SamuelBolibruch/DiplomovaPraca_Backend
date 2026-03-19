import os
import pandas as pd
import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM


VECTORS_DIR = "data/vectors"

RANDOM_STATE = 42
N_SPLITS = 20

IMPOSTOR_SAMPLES_PER_USER = 3

# vybraný user
TARGET_UID = "05TW8Ljp5AOArnLclD8e8LIHgwg2"

# kandidáti na tuning
NU_VALUES = [0.02, 0.03, 0.05, 0.08]
GAMMA_VALUES = [0.001, 0.003, 0.005, 0.01, "scale"]
THRESHOLD_PERCENTILES = [1, 2, 5]

files = [f for f in os.listdir(VECTORS_DIR) if f.startswith("vector_") and f.endswith(".csv")]

print(f"Nájdených {len(files)} vector súborov\n")

# load all data
all_data = {}
for file in files:
    uid = file.replace("vector_", "").replace(".csv", "")
    df = pd.read_csv(os.path.join(VECTORS_DIR, file))
    all_data[uid] = df


def evaluate_user_with_params(X_user, all_data, uid, feature_cols, nu, gamma, threshold_percentile):
    splitter = ShuffleSplit(
        n_splits=N_SPLITS,
        test_size=3,
        random_state=RANDOM_STATE
    )

    results = []

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X_user), start=1):
        X_train = X_user.iloc[train_idx].copy()
        X_test_genuine = X_user.iloc[test_idx].copy()

        impostor_parts = []
        for other_uid, other_df in all_data.items():
            if other_uid == uid:
                continue

            X_other = other_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

            n_take = min(IMPOSTOR_SAMPLES_PER_USER, len(X_other))
            sample = X_other.sample(
                n=n_take,
                random_state=RANDOM_STATE + split_idx
            )
            impostor_parts.append(sample)

        if not impostor_parts:
            continue

        X_impostor = pd.concat(impostor_parts, ignore_index=True)

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_genuine_scaled = scaler.transform(X_test_genuine)
        X_impostor_scaled = scaler.transform(X_impostor)

        model = OneClassSVM(
            kernel="rbf",
            nu=nu,
            gamma=gamma
        )

        model.fit(X_train_scaled)

        # threshold podľa genuine training score
        train_scores = model.decision_function(X_train_scaled)
        threshold = np.percentile(train_scores, threshold_percentile)

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
            "far": far,
            "frr": frr
        })

    if len(results) == 0:
        return None

    results_df = pd.DataFrame(results)

    avg_acc = results_df["accuracy"].mean()
    avg_far = results_df["far"].mean()
    avg_frr = results_df["frr"].mean()

    # skóre: chceme nízke FAR + nízke FRR + aby boli čo najbližšie k sebe
    score = (avg_far + avg_frr) + abs(avg_far - avg_frr)

    return {
        "nu": nu,
        "gamma": gamma,
        "threshold_percentile": threshold_percentile,
        "accuracy": avg_acc,
        "far": avg_far,
        "frr": avg_frr,
        "score": score
    }


# -----------------------------
# iba jeden user
# -----------------------------
if TARGET_UID not in all_data:
    raise ValueError(f"User {TARGET_UID} neexistuje v {VECTORS_DIR}")

df = all_data[TARGET_UID]

print(f"{'='*50}")
print(f"User: {TARGET_UID}")
print(f"{'='*50}")

feature_cols = [c for c in df.columns if c not in ["UserId", "RoundId"]]
X_user = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

if len(X_user) < 15:
    raise ValueError(f"User {TARGET_UID} má len {len(X_user)} vzoriek, treba aspoň 15.")

param_results = []

for nu in NU_VALUES:
    for gamma in GAMMA_VALUES:
        for threshold_percentile in THRESHOLD_PERCENTILES:
            res = evaluate_user_with_params(
                X_user=X_user,
                all_data=all_data,
                uid=TARGET_UID,
                feature_cols=feature_cols,
                nu=nu,
                gamma=gamma,
                threshold_percentile=threshold_percentile
            )

            if res is not None:
                param_results.append(res)

if len(param_results) == 0:
    raise ValueError("Žiadne výsledky.")

param_df = pd.DataFrame(param_results)
param_df = param_df.sort_values(
    ["score", "far", "frr", "accuracy"],
    ascending=[True, True, True, False]
)

best = param_df.iloc[0]

print("Najlepší config:")
print(best[["nu", "gamma", "threshold_percentile", "accuracy", "far", "frr", "score"]])

print("\nTop 10 configov:")
print(param_df.head(10).to_string(index=False))

# uloženie
out_path = f"OneClassSVM_best_params_{TARGET_UID}.csv"
param_df.to_csv(out_path, index=False)
print(f"\n✓ Všetky výsledky uložené do: {out_path}")