import pandas as pd
from scipy.stats import ttest_rel

INPUT_FILE = "per_user_results.csv"
OUTPUT_FILE = "ttest_text_scenarios_results.csv"

# Načítanie dát
df = pd.read_csv(INPUT_FILE)

# Kontrola očakávaných stĺpcov
required_columns = {"user_id", "text_type", "accuracy", "far", "frr", "eer"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Chýbajú stĺpce v CSV súbore: {missing_columns}")

# Rozdelenie podľa scenára
shared = df[df["text_type"] == "shared_text"].copy()
personal = df[df["text_type"] == "personal_text"].copy()

# Spojenie podľa používateľa, aby vznikli párové hodnoty
merged = shared.merge(
    personal,
    on="user_id",
    suffixes=("_shared", "_personal")
)

if merged.empty:
    raise ValueError("Nepodarilo sa vytvoriť párové dáta. Skontroluj user_id a text_type.")

metrics = ["accuracy", "far", "frr", "eer"]
results = []

for metric in metrics:
    shared_values = merged[f"{metric}_shared"]
    personal_values = merged[f"{metric}_personal"]

    t_stat, p_value = ttest_rel(shared_values, personal_values)

    results.append({
        "metric": metric,
        "shared_mean": shared_values.mean(),
        "personal_mean": personal_values.mean(),
        "difference_personal_minus_shared": personal_values.mean() - shared_values.mean(),
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_p_lt_0_05": p_value < 0.05
    })

# Uloženie výsledkov
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print(f"Výsledky boli uložené do: {OUTPUT_FILE}")
print(results_df)