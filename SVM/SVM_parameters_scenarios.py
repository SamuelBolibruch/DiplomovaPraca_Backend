import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


# =========================================================
# CONFIG
# =========================================================

CSV_PATH = "data/training/training_05TW8Ljp5AOArnLclD8e8LIHgwg2.csv"

N_SPLITS = 20
TEST_SIZE = 0.25
RANDOM_STATE = 42

NEG_POS_RATIO = 4

# fixný threshold len kvôli porovnaniu scenárov
FIXED_THRESHOLD = 0.5


# =========================================================
# HELPERS
# =========================================================

def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)

    return tn, fp, fn, tp, far, frr, acc


def undersample(X, y):
    df = X.copy()
    df["label"] = y.values

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    max_neg = min(len(neg), len(pos) * NEG_POS_RATIO)

    neg_sample = neg.sample(n=max_neg, random_state=RANDOM_STATE)

    new_df = pd.concat([pos, neg_sample]).sample(frac=1, random_state=RANDOM_STATE)

    y_new = new_df["label"]
    X_new = new_df.drop(columns=["label"])

    return X_new, y_new


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(CSV_PATH)

print("\nDataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

X = df.drop(columns=["UserId", "RoundId", "label"])
y = df["label"]

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)


# =========================================================
# PARAMETER GRID
# =========================================================

kernels = ["linear", "rbf"]
c_values = [0.1, 1, 10, 100]
gamma_values = ["scale", "auto", 0.1, 0.01, 0.001]
weights = [None, "balanced"]
undersampling = [False, True]


# =========================================================
# SPLIT
# =========================================================

sss = StratifiedShuffleSplit(
    n_splits=N_SPLITS,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

results = []


# =========================================================
# LOOP
# =========================================================

scenario_id = 0

for kernel in kernels:
    for c_val in c_values:
        for weight in weights:
            for use_under in undersampling:

                # gamma má význam len pre non-linear kernely
                if kernel == "linear":
                    gamma_list = [None]
                else:
                    gamma_list = gamma_values

                for gamma_val in gamma_list:
                    scenario_id += 1
                    print("\nRunning scenario", scenario_id)

                    acc_list = []
                    far_list = []
                    frr_list = []

                    for split_idx, (train_idx, test_idx) in enumerate(sss.split(X, y), start=1):

                        X_train = X.iloc[train_idx].copy()
                        y_train = y.iloc[train_idx].copy()

                        X_test = X.iloc[test_idx].copy()
                        y_test = y.iloc[test_idx].copy()

                        if use_under:
                            X_train, y_train = undersample(X_train, y_train)

                        # scaling je pri SVM dôležitý
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        model = SVC(
                            kernel=kernel,
                            C=c_val,
                            gamma=gamma_val if gamma_val is not None else "scale",
                            class_weight=weight,
                            probability=True,
                            random_state=RANDOM_STATE + split_idx
                        )

                        model.fit(X_train_scaled, y_train)

                        probs = model.predict_proba(X_test_scaled)[:, 1]
                        preds = (probs >= FIXED_THRESHOLD).astype(int)

                        tn, fp, fn, tp, far, frr, acc = compute_metrics(y_test, preds)

                        acc_list.append(acc)
                        far_list.append(far)
                        frr_list.append(frr)

                    results.append({
                        "kernel": kernel,
                        "C": c_val,
                        "gamma": gamma_val,
                        "class_weight": weight,
                        "undersampling": use_under,
                        "accuracy_mean": np.mean(acc_list),
                        "FAR_mean": np.mean(far_list),
                        "FRR_mean": np.mean(frr_list),
                        "accuracy_std": np.std(acc_list),
                        "FAR_std": np.std(far_list),
                        "FRR_std": np.std(frr_list)
                    })


# =========================================================
# RESULTS
# =========================================================

results_df = pd.DataFrame(results)

# môžeš meniť podľa toho, čo chceš preferovať
results_df = results_df.sort_values(
    by=["accuracy_mean", "FAR_mean", "FRR_mean"],
    ascending=[False, True, True]
)

print("\n================ BEST SCENARIOS =================")
print(results_df.head(10))

results_df.to_csv("svm_scenarios_results.csv", index=False)

print("\nSaved to svm_scenarios_results.csv")