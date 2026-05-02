#!/usr/bin/env python3
"""Load users from Firebase Firestore and print basic demographics stats.

Reads all documents from `users` collection and prints:
- total number of users
- age distribution in predefined ranges
- right-handed vs left-handed counts
- male vs female counts
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore


DEFAULT_SERVICE_ACCOUNT_KEY = Path("serviceAccountKey.json")

AGE_BUCKETS: list[tuple[int, int, str]] = [
    (0, 17, "0-17"),
    (18, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 200, "55+"),
]


def init_firestore(service_account_key: Path) -> firestore.Client:
    if not service_account_key.exists():
        raise FileNotFoundError(f"Service account key not found: {service_account_key}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(service_account_key))
        firebase_admin.initialize_app(cred)

    return firestore.client()


def normalize_age(value: Any) -> int | None:
    if value is None:
        return None

    try:
        age = int(float(value))
    except (TypeError, ValueError):
        return None

    if age < 0 or age > 120:
        return None

    return age


def age_bucket_label(age: int | None) -> str:
    if age is None:
        return "unknown"

    for lo, hi, label in AGE_BUCKETS:
        if lo <= age <= hi:
            return label

    return "unknown"


def normalize_hand(value: Any) -> str:
    if value is None:
        return "unknown"

    text = str(value).strip().lower()
    if text in {"right", "r", "pravak", "prava", "pravá"}:
        return "right"
    if text in {"left", "l", "lavak", "lava", "ľavá", "lavaa"}:
        return "left"
    return "unknown"


def normalize_gender(value: Any) -> str:
    if value is None:
        return "unknown"

    text = str(value).strip().lower()
    if text in {"male", "m", "muz", "muž"}:
        return "male"
    if text in {"female", "f", "zena", "žena"}:
        return "female"
    return "unknown"


def format_counter(counter: Counter[str], order: list[str]) -> list[str]:
    lines: list[str] = []
    for key in order:
        lines.append(f"  {key}: {counter.get(key, 0)}")

    extra_keys = [k for k in counter.keys() if k not in order]
    for key in sorted(extra_keys):
        lines.append(f"  {key}: {counter[key]}")

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print Firestore users stats: age ranges, dominant hand, gender."
    )
    parser.add_argument(
        "--service-account",
        type=Path,
        default=DEFAULT_SERVICE_ACCOUNT_KEY,
        help="Path to Firebase service account JSON file.",
    )
    args = parser.parse_args()

    try:
        db = init_firestore(args.service_account)
    except Exception as exc:
        print(f"Error initializing Firebase: {exc}")
        return 1

    docs = list(db.collection("users").stream())

    total_users = len(docs)
    age_counter: Counter[str] = Counter()
    hand_counter: Counter[str] = Counter()
    gender_counter: Counter[str] = Counter()

    for doc in docs:
        data = doc.to_dict() or {}

        age = normalize_age(data.get("age"))
        age_counter[age_bucket_label(age)] += 1

        hand = normalize_hand(data.get("dominantHand"))
        hand_counter[hand] += 1

        gender = normalize_gender(data.get("gender"))
        gender_counter[gender] += 1

    print("Statistiky pouzivatelov (Firestore collection: users)")
    print(f"Celkovy pocet pouzivatelov: {total_users}")
    print()

    print("Vekove rozmedzia:")
    age_order = [label for _, _, label in AGE_BUCKETS] + ["unknown"]
    for line in format_counter(age_counter, age_order):
        print(line)
    print()

    print("Dominantna ruka:")
    for line in format_counter(hand_counter, ["right", "left", "unknown"]):
        print(line)
    print()

    print("Pohlavie:")
    for line in format_counter(gender_counter, ["male", "female", "unknown"]):
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
