import os
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, storage

import fix_data

# -----------------------------
# Config
# -----------------------------
SERVICE_ACCOUNT_KEY = "serviceAccountKey.json"
STORAGE_BUCKET = "dp-project-4970a.firebasestorage.app"

TRAINING_CONFIGS = [
    {
        "training_type": "common_training",
        "output_dir": "data/common_training",
        "keystrokes_file": "keystrokes_common.csv",
    },
    {
        "training_type": "personal_training",
        "output_dir": "data/personal_training",
        "keystrokes_file": "keystrokes_personal.csv",
    },
]

SENSOR_FILES = [
    "sensor_accelerometer.csv",
    "sensor_gyroscope.csv",
]

# -----------------------------
# Init Firebase
# -----------------------------
cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
firebase_admin.initialize_app(cred, {"storageBucket": STORAGE_BUCKET})

bucket = storage.bucket()

# -----------------------------
# Získaj všetkých používateľov
# -----------------------------
print("Načítavam používateľov...")
users = []
page = auth.list_users()
while page:
    for user in page.users:
        users.append({"uid": user.uid, "email": user.email})
    page = page.get_next_page()

print(f"Nájdených {len(users)} používateľov.\n")

# -----------------------------
# Pre každý typ tréningu
# -----------------------------
for config in TRAINING_CONFIGS:
    training_type = config["training_type"]
    output_dir = config["output_dir"]
    keystrokes_file = config["keystrokes_file"]
    files_to_download = [keystrokes_file] + SENSOR_FILES

    print(f"\n{'='*50}")
    print(f"Sťahujem: {training_type} -> {output_dir}")
    print(f"{'='*50}\n")

    for user in users:
        uid = user["uid"]
        email = user["email"]

        prefix = f"csv_uploads/{uid}/{training_type}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            print(f"[SKIP] {email} — žiadne dáta v {prefix}")
            continue

        sessions = set()
        for blob in blobs:
            parts = blob.name.replace(prefix, "").split("/")
            if len(parts) >= 2:
                sessions.add(parts[0])

        if not sessions:
            print(f"[SKIP] {email} — nenašli sa session foldery")
            continue

        latest_session = sorted(sessions)[-1]
        print(f"[{email}] session: {latest_session}")

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