import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42

N_ESTIMATORS = 100
MAX_DEPTH = None
CLASS_WEIGHT = "balanced"
THRESHOLD = 0.2

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

# ===============================
# LOOP cez oba datasety
# ===============================

for config in CONFIGS:
    TRAINING_DIR = config["training_dir"]
    MODEL_DIR = config["model_dir"]

    os.makedirs(MODEL_DIR, exist_ok=True)

    files = [
        f for f in os.listdir(TRAINING_DIR)
        if f.startswith("training_") and f.endswith(".csv")
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

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(X, y)

        MODEL_PATH = os.path.join(MODEL_DIR, f"model_{uid}.pkl")

        joblib.dump(
            {
                "model": model,
                "features": feature_cols,
                "threshold": THRESHOLD,
                "user_id": uid
            },
            MODEL_PATH
        )

        print(f"✓ Model uložený: {MODEL_PATH}\n")

print("🔥 Hotovo! Všetky modely natrénované.")