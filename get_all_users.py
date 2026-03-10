import os
import firebase_admin
from firebase_admin import credentials, auth, storage

# -----------------------------
# Config
# -----------------------------
SERVICE_ACCOUNT_KEY = "serviceAccountKey.json"
STORAGE_BUCKET = "dp-project-4970a.firebasestorage.app"
OUTPUT_DIR = "data/common_training"
TRAINING_TYPE = "common_training"  # zmeň na "personal_training" ak budeš sťahovať osobné

FILES_TO_DOWNLOAD = [
    "keystrokes_common.csv",
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
# Pre každého používateľa stiahni súbory
# -----------------------------
for user in users:
    uid = user["uid"]
    email = user["email"]

    # Nájdi všetky session foldery v csv_uploads/{uid}/common_training/
    prefix = f"csv_uploads/{uid}/{TRAINING_TYPE}/"
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        print(f"[SKIP] {email} — žiadne dáta v {prefix}")
        continue

    # Získaj unikátne session foldery (timestampy)
    sessions = set()
    for blob in blobs:
        # csv_uploads/{uid}/common_training/{session_id}/file.csv
        parts = blob.name.replace(prefix, "").split("/")
        if len(parts) >= 2:
            sessions.add(parts[0])

    if not sessions:
        print(f"[SKIP] {email} — nenašli sa session foldery")
        continue

    # Vyber najnovší session folder (najväčší timestamp)
    latest_session = sorted(sessions)[-1]
    print(f"[{email}] session: {latest_session}")

    # Vytvor output priečinok
    out_dir = os.path.join(OUTPUT_DIR, uid)
    os.makedirs(out_dir, exist_ok=True)

    # Stiahni každý súbor
    for filename in FILES_TO_DOWNLOAD:
        blob_path = f"csv_uploads/{uid}/{TRAINING_TYPE}/{latest_session}/{filename}"
        blob = bucket.blob(blob_path)

        out_path = os.path.join(out_dir, filename)

        try:
            blob.download_to_filename(out_path)
            print(f"  ✓ {filename}")
        except Exception:
            print(f"  ✗ {filename} — nenašiel sa")

    print()

print(f"Hotovo! Dáta uložené v: {OUTPUT_DIR}/")