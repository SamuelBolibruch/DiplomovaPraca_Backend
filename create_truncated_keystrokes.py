"""
Skript vytvorí orezané verzie keystroke súborov.
Pre každého usera a každú hodnotu N (10, 20, 50, 75) vytvorí súbor
kde sú zachované len riadky do momentu keď bolo reálne napísaných N znakov
(= N INSERT akcií) v rámci každého RoundId.

Výstup:
  data/common_training/<uid>/keystrokes_10_common.csv
  data/common_training/<uid>/keystrokes_20_common.csv
  ...
  data/personal_training/<uid>/keystrokes_10_personal.csv
  ...
"""

import os
import pandas as pd

TRUNCATE_AT = [10, 20, 25, 50, 75]

CONFIGS = [
    {
        "data_dir": "data/common_training",
        "src_file": "keystrokes_common.csv",
        "out_template": "keystrokes_{n}_common.csv",
    },
    {
        "data_dir": "data/personal_training",
        "src_file": "keystrokes_personal.csv",
        "out_template": "keystrokes_{n}_personal.csv",
    },
]


def truncate_keystrokes(df: pd.DataFrame, n_chars: int) -> pd.DataFrame:
    """
    Pre každý RoundId zachová riadky až do momentu keď dĺžka InputContent prvýkrát
    dosiahne n_chars znakov.

    Dĺžka obsahu po akcii:
      INSERT -> ContentLengthBefore + 1
      DELETE -> ContentLengthBefore - 1

    Keďže INSERT pridáva vždy 1, prvýkrát sa dĺžka N dosiahne na INSERT riadku
    kde ContentLengthBefore == N - 1. Zachovajú sa všetky riadky vrátane tohto.
    """
    result_parts = []

    for round_id, group in df.groupby("RoundId", sort=True):
        group = group.reset_index(drop=True)
        action_l = group["ActionType"].astype(str).str.lower()
        content_before = pd.to_numeric(group["ContentLengthBefore"], errors="coerce").fillna(0).astype(int)

        # Dĺžka obsahu po každej akcii
        content_after = content_before.copy()
        content_after[action_l == "insert"] += 1
        content_after[action_l == "delete"] -= 1

        # Prvý riadok kde content_after >= n_chars
        reached = (content_after >= n_chars)

        if not reached.any():
            # Nikdy nedosiahlo N znakov — zachovať celý round
            result_parts.append(group)
        else:
            cutoff_idx = int(reached.idxmax())
            result_parts.append(group.iloc[: cutoff_idx + 1])

    if len(result_parts) == 0:
        return df.iloc[0:0].copy()

    return pd.concat(result_parts, ignore_index=True)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for config in CONFIGS:
        data_dir = os.path.join(base_dir, config["data_dir"])

        if not os.path.exists(data_dir):
            print(f"[SKIP] {data_dir} — neexistuje")
            continue

        user_dirs = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        print(f"\n{'='*55}")
        print(f"Spracovávam: {config['data_dir']}")
        print(f"{'='*55}")

        for uid in user_dirs:
            user_dir = os.path.join(data_dir, uid)
            src_path = os.path.join(user_dir, config["src_file"])

            if not os.path.exists(src_path):
                print(f"  [SKIP] {uid} — chýba {config['src_file']}")
                continue

            df = pd.read_csv(src_path)

            for n in TRUNCATE_AT:
                out_filename = config["out_template"].format(n=n)
                out_path = os.path.join(user_dir, out_filename)

                truncated = truncate_keystrokes(df, n)

                truncated.to_csv(out_path, index=False)

                total_rounds = df["RoundId"].nunique() if "RoundId" in df.columns else 0
                kept_rounds = truncated["RoundId"].nunique() if "RoundId" in truncated.columns and len(truncated) > 0 else 0
                print(f"  [{uid}] N={n:>2}: {len(truncated):>5} riadkov z {len(df):>5} | "
                      f"{kept_rounds}/{total_rounds} rounds → {out_filename}")


if __name__ == "__main__":
    main()
