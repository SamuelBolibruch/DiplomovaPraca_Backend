import os
import pandas as pd

VECTOR_SOURCES = [
    {"input_dir": "data/vectors",          "output_dir": "data/training"},
    {"input_dir": "data/vectors_personal", "output_dir": "data/training_personal"},
]

for source in VECTOR_SOURCES:

    VECTORS_DIR = source["input_dir"]
    OUTPUT_DIR = source["output_dir"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nProcessing: {VECTORS_DIR}")

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

        print(
            f"✓ {target_uid} → "
            f"{len(combined[combined['label']==1])} genuine, "
            f"{len(combined[combined['label']==0])} impostors"
        )

print("\nHotovo! Všetky tréningové súbory vytvorené.")