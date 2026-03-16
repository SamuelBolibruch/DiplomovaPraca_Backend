import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


CSV_PATH = "data/training/training_zwWo0D9MehY88RXRA0BYoGdJKCz2.csv"

RANDOM_STATE = 42
N_SPLITS = 20

# finálne vybrané nastavenia
TEST_SIZE = 0.20
N_ESTIMATORS = 100
MAX_DEPTH = None
CLASS_WEIGHT = "balanced"
THRESHOLD = 0.20


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)

    return tn, fp, fn, tp, far, frr, acc


# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(CSV_PATH)

feature_cols = [c for c in df.columns if c not in ["UserId", "RoundId", "label"]]

X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df["label"]

print("Dataset shape:", df.shape)
print("\nLabel distribution:")
print(y.value_counts())

sss = StratifiedShuffleSplit(
    n_splits=N_SPLITS,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

results = []

for split_idx, (train_idx, test_idx) in enumerate(sss.split(X, y), start=1):
    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()

    X_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()

    # finálny RF model s najlepšími nastaveniami
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE + split_idx,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    # fixed selected threshold
    y_pred = (probs >= THRESHOLD).astype(int)

    tn, fp, fn, tp, far, frr, acc = compute_metrics(y_test, y_pred)

    results.append({
        "split": split_idx,
        "threshold": THRESHOLD,
        "accuracy": acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "far": far,
        "frr": frr,
        "n_train_pos": int((y_train == 1).sum()),
        "n_train_neg": int((y_train == 0).sum()),
        "n_test_pos": int((y_test == 1).sum()),
        "n_test_neg": int((y_test == 0).sum())
    })

results_df = pd.DataFrame(results)

print("\nPer-split results:")
print(results_df)

print("\nAverage results:")
print(results_df[["accuracy", "far", "frr"]].mean())

print("\nStd results:")
print(results_df[["accuracy", "far", "frr"]].std())