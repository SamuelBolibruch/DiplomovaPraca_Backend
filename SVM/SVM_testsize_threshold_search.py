import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score


# =========================================================
# CONFIG
# =========================================================

CSV_PATH = "data/training/training_05TW8Ljp5AOArnLclD8e8LIHgwg2.csv"
RANDOM_STATE = 42
N_SPLITS = 20

ID_COLUMNS = ["UserId", "RoundId", "label"]

# fixed best SVM settings
KERNEL = "rbf"
C_VALUE = 100.0
GAMMA_VALUE = 0.001
CLASS_WEIGHT = None

TEST_SIZES = [0.20, 0.25, 0.30, 0.35, 0.40]

THRESHOLD_MODES = [
    "fixed_003",
    "fixed_005",
    "fixed_007",
    "fixed_010",
    "fixed_015",
    "fixed_020",
    "fixed_030",
    "default_05",
    "eer"
]


# =========================================================
# HELPERS
# =========================================================

def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)

    return tn, fp, fn, tp, far, frr, acc


def find_eer_threshold(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    threshold = thresholds[idx]

    return eer, threshold, fpr[idx], fnr[idx]


def get_threshold(mode, y_true, probs):
    if mode == "fixed_003":
        return 0.03
    elif mode == "fixed_005":
        return 0.05
    elif mode == "fixed_007":
        return 0.07
    elif mode == "fixed_010":
        return 0.10
    elif mode == "fixed_015":
        return 0.15
    elif mode == "fixed_020":
        return 0.20
    elif mode == "fixed_030":
        return 0.30
    elif mode == "default_05":
        return 0.50
    elif mode == "eer":
        _, threshold, _, _ = find_eer_threshold(y_true, probs)
        return threshold
    else:
        raise ValueError(f"Unknown threshold mode: {mode}")


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(CSV_PATH)

print("\n================ LOAD DATA ================")
print("Dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nLabel proportions:")
print(df["label"].value_counts(normalize=True))

feature_cols = [c for c in df.columns if c not in ID_COLUMNS]

X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df["label"]


# =========================================================
# MAIN SEARCH
# =========================================================

all_results = []
scenario_id = 0

for test_size in TEST_SIZES:

    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=test_size,
        random_state=RANDOM_STATE
    )

    for threshold_mode in THRESHOLD_MODES:

        scenario_id += 1

        print(
            f"\nScenario {scenario_id}: "
            f"test_size={test_size}, "
            f"threshold_mode={threshold_mode}"
        )

        split_rows = []

        for split_idx, (train_idx, test_idx) in enumerate(sss.split(X, y), start=1):
            X_train = X.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy()

            X_test = X.iloc[test_idx].copy()
            y_test = y.iloc[test_idx].copy()

            # scaling is necessary for SVM
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = SVC(
                kernel=KERNEL,
                C=C_VALUE,
                gamma=GAMMA_VALUE,
                class_weight=CLASS_WEIGHT,
                probability=True,
                random_state=RANDOM_STATE + split_idx
            )

            model.fit(X_train_scaled, y_train)

            probs = model.predict_proba(X_test_scaled)[:, 1]

            eer_value, eer_threshold, roc_fpr, roc_fnr = find_eer_threshold(y_test, probs)
            used_threshold = get_threshold(threshold_mode, y_test, probs)

            y_pred = (probs >= used_threshold).astype(int)

            tn, fp, fn, tp, far, frr, acc = compute_metrics(y_test, y_pred)

            split_rows.append({
                "scenario_id": scenario_id,
                "split": split_idx,
                "test_size": test_size,
                "threshold_mode": threshold_mode,

                "train_pos": int((y_train == 1).sum()),
                "train_neg": int((y_train == 0).sum()),
                "test_pos": int((y_test == 1).sum()),
                "test_neg": int((y_test == 0).sum()),

                "used_threshold": used_threshold,
                "eer_value": eer_value,
                "eer_threshold": eer_threshold,

                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "far": far,
                "frr": frr,
                "accuracy": acc,
            })

        scenario_df = pd.DataFrame(split_rows)

        summary_row = {
            "scenario_id": scenario_id,
            "test_size": test_size,
            "threshold_mode": threshold_mode,

            "mean_train_pos": scenario_df["train_pos"].mean(),
            "mean_train_neg": scenario_df["train_neg"].mean(),
            "mean_test_pos": scenario_df["test_pos"].mean(),
            "mean_test_neg": scenario_df["test_neg"].mean(),

            "mean_threshold": scenario_df["used_threshold"].mean(),

            "mean_eer_value": scenario_df["eer_value"].mean(),
            "std_eer_value": scenario_df["eer_value"].std(),

            "mean_accuracy": scenario_df["accuracy"].mean(),
            "std_accuracy": scenario_df["accuracy"].std(),

            "mean_far": scenario_df["far"].mean(),
            "std_far": scenario_df["far"].std(),

            "mean_frr": scenario_df["frr"].mean(),
            "std_frr": scenario_df["frr"].std(),
        }

        all_results.append(summary_row)


# =========================================================
# FINAL RESULTS
# =========================================================

results_df = pd.DataFrame(all_results)

# zoradenie: najprv podľa EER, potom FAR, potom FRR
results_df = results_df.sort_values(
    by=["mean_eer_value", "mean_far", "mean_frr"],
    ascending=[True, True, True]
)

print("\n================ TOP 20 SCENARIOS ================")
print(results_df.head(20).to_string(index=False))

results_df.to_csv("svm_testsize_threshold_results.csv", index=False)

print("\nSaved to svm_testsize_threshold_results.csv")