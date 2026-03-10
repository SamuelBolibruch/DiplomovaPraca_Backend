import os
import pandas as pd

VECTORS_DIR = "data/vectors"
OUTPUT_DIR = "data/training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_uids = [
    f.replace("vector_", "").replace(".csv", "")
    for f in os.listdir(VECTORS_DIR)
    if f.endswith(".csv")
]

for target_uid in all_uids:
    dfs = []
    for uid in all_uids:
        df = pd.read_csv(os.path.join(VECTORS_DIR, f"vector_{uid}.csv"))
        df["label"] = 1 if uid == target_uid else 0
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, f"training_{target_uid}.csv")
    combined.to_csv(out_path, index=False)
    print(f"✓ {target_uid} → {len(combined[combined['label']==1])} genuine, {len(combined[combined['label']==0])} impostors")

print(f"\nHotovo! Tréningové súbory v: {OUTPUT_DIR}/")