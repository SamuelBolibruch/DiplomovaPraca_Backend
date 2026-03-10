import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve


# =========================================================
# CONFIG
# =========================================================

CSV_PATH = "data/training/training_05TW8Ljp5AOArnLclD8e8LIHgwg2.csv"

N_SPLITS = 20
TEST_SIZE = 0.25
RANDOM_STATE = 42

NEG_POS_RATIO = 4


# =========================================================
# HELPERS
# =========================================================

def compute_far_frr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return far, frr


def find_eer(y_true, probs):

    fpr, tpr, thresholds = roc_curve(y_true, probs)

    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))

    eer = (fpr[idx] + fnr[idx]) / 2
    threshold = thresholds[idx]

    return eer, threshold


def undersample(X, y):

    df = X.copy()
    df["label"] = y

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    max_neg = min(len(neg), len(pos) * NEG_POS_RATIO)

    neg_sample = neg.sample(n=max_neg)

    new_df = pd.concat([pos, neg_sample]).sample(frac=1)

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

depths = [None, 10, 20, 30]
trees = [100, 300, 600]
weights = [None, "balanced", "balanced_subsample"]
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

for depth in depths:
    for n_tree in trees:
        for weight in weights:
            for use_under in undersampling:

                scenario_id += 1

                print("\nRunning scenario", scenario_id)

                eer_list = []
                far_list = []
                frr_list = []

                for train_idx, test_idx in sss.split(X, y):

                    X_train = X.iloc[train_idx]
                    y_train = y.iloc[train_idx]

                    X_test = X.iloc[test_idx]
                    y_test = y.iloc[test_idx]

                    if use_under:
                        X_train, y_train = undersample(X_train, y_train)

                    model = RandomForestClassifier(
                        n_estimators=n_tree,
                        max_depth=depth,
                        class_weight=weight,
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )

                    model.fit(X_train, y_train)

                    probs = model.predict_proba(X_test)[:,1]

                    eer, thr = find_eer(y_test, probs)

                    preds = (probs >= thr).astype(int)

                    far, frr = compute_far_frr(y_test, preds)

                    eer_list.append(eer)
                    far_list.append(far)
                    frr_list.append(frr)

                results.append({

                    "depth": depth,
                    "trees": n_tree,
                    "class_weight": weight,
                    "undersampling": use_under,

                    "EER_mean": np.mean(eer_list),
                    "FAR_mean": np.mean(far_list),
                    "FRR_mean": np.mean(frr_list),

                    "EER_std": np.std(eer_list)

                })


# =========================================================
# RESULTS
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values("EER_mean")

print("\n================ BEST SCENARIOS =================")

print(results_df.head(10))


results_df.to_csv("rf_scenarios_results.csv", index=False)

print("\nSaved to rf_scenarios_results.csv")