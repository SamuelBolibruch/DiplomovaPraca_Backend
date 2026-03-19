import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM


VECTORS_DIR = "data/vectors"
MODEL_DIR = "OneClassSVM/models"

RANDOM_STATE = 42
N_SPLITS = 20

IMPOSTOR_SAMPLES_PER_USER = 3

# FIXNÉ globálne parametre pre všetkých userov
NU = 0.08
GAMMA = "scale"
THRESHOLD_PERCENTILE = 5

os.makedirs(MODEL_DIR, exist_ok=True)

files = [f for f in os.listdir(VECTORS_DIR) if f.startswith("vector_") and f.endswith(".csv")]

print(f"Nájdených {len(files)} vector súborov\n")

# load all data
all_data = {}
for file in files:
    uid = file.replace("vector_", "").replace(".csv", "")
    df = pd.read_csv(os.path.join(VECTORS_DIR, file))
    all_data[uid] = df


def evaluate_user_fixed_params(X_user, all_data, uid, feature_cols):
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
            nu=NU,
            gamma=GAMMA
        )

        model.fit(X_train_scaled)

        train_scores = model.decision_function(X_train_scaled)
        threshold = np.percentile(train_scores, THRESHOLD_PERCENTILE)

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
            "frr": frr,
            "threshold": float(threshold),
            "n_train_pos": len(X_train),
            "n_test_pos": len(X_test_genuine),
            "n_test_neg": len(X_impostor),
        })

    if len(results) == 0:
        return None

    return pd.DataFrame(results)


all_summaries = []

for file in files:
    uid = file.replace("vector_", "").replace(".csv", "")
    df = all_data[uid]

    print(f"{'='*50}")
    print(f"User: {uid}")
    print(f"{'='*50}")

    feature_cols = [c for c in df.columns if c not in ["UserId", "RoundId"]]
    X_user = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(X_user) < 15:
        print(f"Preskakujem {uid} — má len {len(X_user)} vzoriek\n")
        continue

    # -----------------------------
    # Evaluation with fixed params
    # -----------------------------
    results_df = evaluate_user_fixed_params(
        X_user=X_user,
        all_data=all_data,
        uid=uid,
        feature_cols=feature_cols
    )

    if results_df is None or len(results_df) == 0:
        print("Žiadne eval výsledky.\n")
        continue

    print("Per-split results:")
    print(results_df)

    avg_acc = results_df["accuracy"].mean()
    avg_far = results_df["far"].mean()
    avg_frr = results_df["frr"].mean()

    std_acc = results_df["accuracy"].std()
    std_far = results_df["far"].std()
    std_frr = results_df["frr"].std()

    print("\nAverage:")
    print(results_df[["accuracy", "far", "frr"]].mean())

    print("\nStd:")
    print(results_df[["accuracy", "far", "frr"]].std())

    # -----------------------------
    # Final training on FULL user data
    # -----------------------------
    final_scaler = RobustScaler()
    X_user_scaled = final_scaler.fit_transform(X_user)

    final_model = OneClassSVM(
        kernel="rbf",
        nu=NU,
        gamma=GAMMA
    )

    final_model.fit(X_user_scaled)

    final_train_scores = final_model.decision_function(X_user_scaled)
    final_threshold = float(np.percentile(final_train_scores, THRESHOLD_PERCENTILE))

    model_path = os.path.join(MODEL_DIR, f"oneclass_svm_{uid}.pkl")

    joblib.dump(
        {
            "model": final_model,
            "scaler": final_scaler,
            "features": feature_cols,
            "user_id": uid,
            "nu": NU,
            "gamma": GAMMA,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "threshold": final_threshold
        },
        model_path
    )

    print(f"\n✓ Model uložený do: {model_path}\n")

    all_summaries.append({
        "user_id": uid,
        "nu": NU,
        "gamma": GAMMA,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "avg_accuracy": avg_acc,
        "avg_far": avg_far,
        "avg_frr": avg_frr,
        "std_accuracy": std_acc,
        "std_far": std_far,
        "std_frr": std_frr,
        "model_path": model_path
    })

if len(all_summaries) > 0:
    summary_df = pd.DataFrame(all_summaries)

    print(f"{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(summary_df.to_string(index=False))

    summary_df.to_csv("OneClassSVM_summary.csv", index=False)
    print("\n✓ Summary uložené do: OneClassSVM_summary.csv")