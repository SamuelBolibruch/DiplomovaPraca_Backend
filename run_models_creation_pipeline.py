import argparse
import subprocess
import sys  

parser = argparse.ArgumentParser()
parser.add_argument("--uid", type=str, default=None, help="Pridaj iba nového používateľa (UID)")
args = parser.parse_args()

if args.uid:
    print(f"\n🆕 Režim nový používateľ: {args.uid}")
    print("Predpoklad: dáta ostatných používateľov sú už stiahnuté a spracované.\n")
    scripts = [
        ["get_all_users.py", "--uid", args.uid],
        ["vectors_creation.py", "--uid", args.uid],
        ["prepare_training_data.py"],
        ["RandomForrest/RF_train_models_final.py"],
    ]
else:
    scripts = [
        ["get_all_users.py"],
        ["vectors_creation.py"],
        ["prepare_training_data.py"],
        ["RandomForrest/RF_train_models_final.py"],
    ]

for script_args in scripts:
    label = " ".join(script_args)
    print(f"\n🚀 Spúšťam {label}...\n")

    result = subprocess.run([sys.executable] + script_args)

    if result.returncode != 0:
        print(f"❌ Chyba v {label}, pipeline sa zastavuje.")
        break

print("\n✅ Pipeline dokončená.")
