import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "summary_all_datasets.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "exp4_training_size_eer.png")

TRAIN_SIZES = [3, 5, 8, 10]

df = pd.read_csv(CSV_PATH)
df = df[df["train_size"].isin(TRAIN_SIZES)]

general = df[df["Dataset"] == "general"].sort_values("train_size")
personal = df[df["Dataset"] == "personal"].sort_values("train_size")

fig, ax = plt.subplots(figsize=(6, 4))

ax.errorbar(
    general["train_size"],
    general["avg_eer"] * 100,
    yerr=general["std_eer"] * 100,
    marker="o",
    linestyle="-",
    color="black",
    capsize=4,
    label="Preddefinovaný text",
)
ax.errorbar(
    personal["train_size"],
    personal["avg_eer"] * 100,
    yerr=personal["std_eer"] * 100,
    marker="s",
    linestyle="--",
    color="dimgray",
    capsize=4,
    label="Vlastný text",
)

ax.set_xlabel("Počet tréningových vzoriek", fontsize=12)
ax.set_ylabel("EER (%)", fontsize=12)
ax.set_xticks(TRAIN_SIZES)
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(axis="y", linestyle=":", linewidth=0.7, color="gray", alpha=0.7)
ax.legend(fontsize=11)
ax.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=300)
print(f"Graf uložený: {OUTPUT_PATH}")
