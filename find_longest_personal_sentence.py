#!/usr/bin/env python3
"""List users sorted by length of their longest sentence.

The script scans all `keystrokes_personal.csv` files under `data/personal_training`.
For each user, it finds their longest `InputContent` value and then prints users
sorted by sentence length.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LongestRecord:
    user_id: str
    length: int
    sentence: str
    file_path: Path
    row_index: int


def collect_user_longest_sentences(root_dir: Path) -> list[LongestRecord]:
    results: list[LongestRecord] = []

    for csv_path in sorted(root_dir.glob("*/keystrokes_personal.csv")):
        user_id = csv_path.parent.name
        longest_for_user: LongestRecord | None = None

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames or "InputContent" not in reader.fieldnames:
                continue

            for idx, row in enumerate(reader, start=2):
                sentence = row.get("InputContent", "") or ""
                sentence_len = len(sentence)

                if longest_for_user is None or sentence_len > longest_for_user.length:
                    longest_for_user = LongestRecord(
                        user_id=user_id,
                        length=sentence_len,
                        sentence=sentence,
                        file_path=csv_path,
                        row_index=idx,
                    )

        if longest_for_user is not None:
            results.append(longest_for_user)

    return sorted(results, key=lambda r: r.length, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find each user's longest sentence in personal keystroke data and "
            "sort users by sentence length."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/personal_training"),
        help="Path containing user folders with keystrokes_personal.csv files.",
    )
    args = parser.parse_args()

    if not args.root.exists() or not args.root.is_dir():
        print(f"Error: directory not found: {args.root}")
        return 1

    ordered = collect_user_longest_sentences(args.root)

    if not ordered:
        print("No valid records found (missing files or InputContent column).")
        return 1

    print("Users sorted by longest sentence length (descending):")
    for i, item in enumerate(ordered, start=1):
        print(f"{i}. User ID: {item.user_id}")
        print(f"   Length: {item.length} characters")
        print(f"   Sentence: {item.sentence}")
        print(f"   Source file: {item.file_path}")
        print(f"   CSV row: {item.row_index}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
