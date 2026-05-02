import os
import argparse
import pandas as pd
import firebase_admin
from firebase_admin import credentials, storage

import fix_data

# -----------------------------
# Config
# -----------------------------
SERVICE_ACCOUNT_KEY = "serviceAccountKey.json"
STORAGE_BUCKET = "dp-project-4970a.firebasestorage.app"

AUTH_OUTPUT_DIR = "data/authentification"

AUTH_CONFIGS = {
    "general": {
        "remote_type": "general",
        "keystrokes_file": "keystrokes_authentication.csv",
    },
    "personal": {
        "remote_type": "personal",
        "keystrokes_file": "keystrokes_authentication.csv",
    },
}

SENSOR_FILES = [
    "sensor_accelerometer.csv",
    "sensor_gyroscope.csv",
]

# -----------------------------
# Init Firebase
# -----------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
    firebase_admin.initialize_app(cred, {"storageBucket": STORAGE_BUCKET})

bucket = storage.bucket()


def download_latest_authentication_attempt(uid: str, auth_type: str) -> dict:
    if auth_type not in AUTH_CONFIGS:
        raise ValueError(f"Neznámy auth_type='{auth_type}'. Použi 'general' alebo 'personal'.")

    config = AUTH_CONFIGS[auth_type]
    keystrokes_file = config["keystrokes_file"]
    remote_type = config["remote_type"]

    prefix = f"csv_uploads/{uid}/authentication/{remote_type}/"
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        raise FileNotFoundError(f"Nenašli sa žiadne dáta v: {prefix}")

    timestamps = set()
    for blob in blobs:
        rel = blob.name.replace(prefix, "", 1)
        parts = rel.split("/")
        if len(parts) >= 2 and parts[0].strip():
            timestamps.add(parts[0])

    if not timestamps:
        raise FileNotFoundError(f"Nenašli sa žiadne timestamp foldery v: {prefix}")

    try:
        latest_timestamp = sorted(timestamps, key=lambda x: int(x))[-1]
    except ValueError:
        latest_timestamp = sorted(timestamps)[-1]

    out_dir = os.path.join(AUTH_OUTPUT_DIR, uid, auth_type)
    os.makedirs(out_dir, exist_ok=True)

    files_to_download = [keystrokes_file] + SENSOR_FILES
    downloaded = []
    missing = []

    # Vyčisti staré súbory, aby sa pri neúplnom downloade nepoužili zastarané dáta.
    for filename in files_to_download + ["vector_authentication.csv"]:
        stale_path = os.path.join(out_dir, filename)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    print(f"[{uid}] auth_type={auth_type}, latest_timestamp={latest_timestamp}")

    for filename in files_to_download:
        blob_path = f"{prefix}{latest_timestamp}/{filename}"
        blob = bucket.blob(blob_path)
        out_path = os.path.join(out_dir, filename)

        try:
            if not blob.exists():
                print(f"  ✗ {filename} — súbor v storage neexistuje")
                missing.append(filename)
                continue

            blob.download_to_filename(out_path)
            print(f"  ✓ {filename}")
            downloaded.append(out_path)

            if filename == keystrokes_file:
                # Fix sa vykoná až po úspešnom stiahnutí
                fix_data.fix_biometry_csv(out_path)

                # Nastav správne UserId
                df = pd.read_csv(out_path)
                if "UserId" in df.columns:
                    df["UserId"] = uid
                    df.to_csv(out_path, index=False)

        except Exception as e:
            print(f"  ✗ {filename} — chyba: {e}")
            missing.append(filename)

    if missing:
        missing_unique = sorted(set(missing))
        raise FileNotFoundError(
            f"Chýbajú povinné auth súbory pre uid={uid}, auth_type={auth_type}, "
            f"timestamp={latest_timestamp}: {missing_unique}"
        )

    return {
        "uid": uid,
        "auth_type": auth_type,
        "timestamp": latest_timestamp,
        "output_dir": out_dir,
        "downloaded_files": downloaded,
        "keystrokes_path": os.path.join(out_dir, keystrokes_file),
        "accelerometer_path": os.path.join(out_dir, "sensor_accelerometer.csv"),
        "gyroscope_path": os.path.join(out_dir, "sensor_gyroscope.csv"),
    }


def main():
    parser = argparse.ArgumentParser(description="Download latest authentication CSV files for one user.")
    parser.add_argument("--uid", required=True, help="User ID")
    parser.add_argument(
        "--auth-type",
        required=True,
        choices=["general", "personal"],
        help="Authentication type: general alebo personal",
    )

    args = parser.parse_args()

    result = download_latest_authentication_attempt(
        uid=args.uid,
        auth_type=args.auth_type,
    )

    print("\nHotovo.")
    print(f"UID: {result['uid']}")
    print(f"Typ: {result['auth_type']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Uložené do: {result['output_dir']}")


if __name__ == "__main__":
    main()