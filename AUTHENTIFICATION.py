import os
import argparse
import joblib
import pandas as pd
import numpy as np

MODEL_DIRS = {
    "general": "RandomForrest/models",
    "personal": "RandomForrest/models_personal",
}

AUTH_BASE_DIR = "data/authentification"
VECTOR_FILENAME = "vector_authentication.csv"


def run_authentication(uid: str, auth_type: str) -> dict:
    if auth_type not in MODEL_DIRS:
        raise ValueError(f"Neznámy auth_type='{auth_type}'. Použi 'general' alebo 'personal'.")

    model_dir = MODEL_DIRS[auth_type]

    model_path = os.path.join(model_dir, f"model_{uid}.pkl")
    vector_path = os.path.join(AUTH_BASE_DIR, uid, VECTOR_FILENAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model neexistuje: {model_path}")

    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Vektor neexistuje: {vector_path}")

    data = joblib.load(model_path)

    model = data["model"]
    feature_cols = data["features"]
    threshold = float(data["threshold"])
    user_id = data["user_id"]

    df = pd.read_csv(vector_path)

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Vo vektore chýbajú požadované features. "
            f"Počet chýbajúcich: {len(missing_cols)}\n"
            f"Prvých 20: {missing_cols[:20]}"
        )

    X_new = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    prob = float(model.predict_proba(X_new)[0][1])
    pred = int(prob >= threshold)

    result = {
        "user_id": user_id,
        "auth_type": auth_type,
        "model_path": model_path,
        "vector_path": vector_path,
        "threshold": threshold,
        "probability_genuine": prob,
        "accepted": bool(pred),
        "decision": "ACCEPT" if pred == 1 else "REJECT",
    }

    print(f"User model: {result['user_id']}")
    print(f"Auth type: {result['auth_type']}")
    print(f"Model path: {result['model_path']}")
    print(f"Vector path: {result['vector_path']}")
    print(f"Threshold: {result['threshold']}")
    print(f"Probability genuine: {result['probability_genuine']:.6f}")

    if result["accepted"]:
        print("✅ ACCEPT")
    else:
        print("❌ REJECT")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Authenticate one user using RF model and authentication vector."
    )
    parser.add_argument("--uid", required=True, help="User ID")
    parser.add_argument(
        "--auth-type",
        required=True,
        choices=["general", "personal"],
        help="Authentication type: general alebo personal",
    )

    args = parser.parse_args()

    run_authentication(uid=args.uid, auth_type=args.auth_type)


if __name__ == "__main__":
    main()