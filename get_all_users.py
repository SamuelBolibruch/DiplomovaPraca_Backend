import argparse
import json
import os
import pandas as pd
import firebase_admin
from firebase_admin import credentials, storage

import fix_data

parser = argparse.ArgumentParser()
parser.add_argument("--uid", type=str, default=None, help="Spracuj iba tohto používateľa")
args = parser.parse_args()

SERVICE_ACCOUNT_KEY = "serviceAccountKey.json"

TRAINING_CONFIGS = [
    {
        "training_type": "common_training",
        "output_dir": "data/raw_common",
        "keystrokes_file": "keystrokes_common.csv",
    },
    {
        "training_type": "personal_training",
        "output_dir": "data/raw_personal",
        "keystrokes_file": "keystrokes_personal.csv",
    },
]

SENSOR_FILES = [
    "sensor_accelerometer.csv",
    "sensor_gyroscope.csv",
]

with open(SERVICE_ACCOUNT_KEY) as f:
    _sa = json.load(f)

cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
firebase_admin.initialize_app(cred, {"storageBucket": f"{_sa['project_id']}.firebasestorage.app"})

bucket = storage.bucket()

print("Načítavam používateľov zo Storage...")
if args.uid:
    uids = [args.uid]
    print(f"Režim konkrétny používateľ: {args.uid}\n")
else:
    blobs = bucket.list_blobs(prefix="csv_uploads/", delimiter="/")
    list(blobs)  # treba iterovať aby sa naplnili prefixes
    uids = [p.replace("csv_uploads/", "").rstrip("/") for p in blobs.prefixes]
    print(f"Nájdených {len(uids)} používateľov.\n")

for config in TRAINING_CONFIGS:
    training_type = config["training_type"]
    output_dir = config["output_dir"]
    keystrokes_file = config["keystrokes_file"]
    files_to_download = [keystrokes_file] + SENSOR_FILES

    print(f"\n{'='*50}")
    print(f"Sťahujem: {training_type} -> {output_dir}")
    print(f"{'='*50}\n")

    for uid in uids:
        prefix = f"csv_uploads/{uid}/{training_type}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            print(f"[SKIP] {uid} — žiadne dáta v {prefix}")
            continue

        sessions = set()
        for blob in blobs:
            parts = blob.name.replace(prefix, "").split("/")
            if len(parts) >= 2:
                sessions.add(parts[0])

        if not sessions:
            print(f"[SKIP] {uid} — nenašli sa session foldre")
            continue

        latest_session = sorted(sessions)[-1]
        print(f"[{uid}] session: {latest_session}")

        out_dir = os.path.join(output_dir, uid)
        os.makedirs(out_dir, exist_ok=True)

        for filename in files_to_download:
            blob_path = f"csv_uploads/{uid}/{training_type}/{latest_session}/{filename}"
            blob = bucket.blob(blob_path)

            out_path = os.path.join(out_dir, filename)

            try:
                blob.download_to_filename(out_path)
                print(f"  ✓ {filename}")
                if filename == keystrokes_file:
                    fix_data.fix_biometry_csv(out_path)
                    df = pd.read_csv(out_path)
                    if "UserId" in df.columns:
                        df["UserId"] = uid
                        df.to_csv(out_path, index=False)
            except Exception as e:
                print(f"  ✗ {filename} — chyba: {e}")

        print()

print(f"\nHotovo!")