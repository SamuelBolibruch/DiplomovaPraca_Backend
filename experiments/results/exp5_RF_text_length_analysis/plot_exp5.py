import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "summary_all_datasets.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "exp5_text_length_eer.png")

df = pd.read_csv(CSV_PATH, dtype={"text_length": str})

LENGTH_ORDER = ["10", "25", "50", "75"]
x_pos = list(range(len(LENGTH_ORDER)))
label_map = {v: i for i, v in enumerate(LENGTH_ORDER)}

general = df[df["Dataset"] == "general"].copy()
personal = df[df["Dataset"] == "personal"].copy()
general["x"] = general["text_length"].map(label_map)
personal["x"] = personal["text_length"].map(label_map)
general = general.sort_values("x")
personal = personal.sort_values("x")

fig, ax = plt.subplots(figsize=(6, 4))

ax.errorbar(
    general["x"],
    general["avg_eer"] * 100,
    yerr=general["std_eer"] * 100,
    marker="o",
    linestyle="-",
    color="black",
    capsize=4,
    label="Preddefinovaný text",
)
ax.errorbar(
    personal["x"],
    personal["avg_eer"] * 100,
    yerr=personal["std_eer"] * 100,
    marker="s",
    linestyle="--",
    color="dimgray",
    capsize=4,
    label="Vlastný text",
)

ax.set_xlabel("Dĺžka textu (počet znakov)", fontsize=12)
ax.set_ylabel("EER (%)", fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(LENGTH_ORDER)
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(axis="y", linestyle=":", linewidth=0.7, color="gray", alpha=0.7)
ax.legend(fontsize=11, loc="upper right")
ax.set_ylim(bottom=0, top=16)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=300)
print(f"Graf uložený: {OUTPUT_PATH}")
