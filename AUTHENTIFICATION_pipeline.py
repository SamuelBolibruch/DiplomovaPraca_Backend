import subprocess
import argparse

parser = argparse.ArgumentParser(description="Run authentication pipeline")
parser.add_argument("--uid", required=True, help="User ID")
parser.add_argument(
    "--auth-type",
    required=True,
    choices=["general", "personal"],
    help="Authentication type"
)

args = parser.parse_args()

uid = args.uid
auth_type = args.auth_type

scripts = [
    ["python3", "load_auth_data.py", "--uid", uid, "--auth-type", auth_type],
    ["python3", "auth_vector_creation.py", "--uid", uid],
    ["python3", "AUTHENTIFICATION.py", "--uid", uid, "--auth-type", auth_type],
]

for command in scripts:
    print(f"\n🚀 Spúšťam: {' '.join(command)}\n")

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"❌ Chyba pri: {' '.join(command)}")
        break

print("\n✅ Pipeline dokončená.")