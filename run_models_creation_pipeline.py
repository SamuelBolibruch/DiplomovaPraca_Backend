import subprocess

scripts = [
    "get_all_users.py",
    "vectors_creation.py",
    "prepare_training_data.py",
    "RandomForrest/RF_train_models_final.py"
]

for script in scripts:
    print(f"\n🚀 Spúšťam {script}...\n")

    result = subprocess.run(["python3", script])

    if result.returncode != 0:
        print(f"❌ Chyba v {script}, pipeline sa zastavuje.")
        break

print("\n✅ Pipeline dokončená.")