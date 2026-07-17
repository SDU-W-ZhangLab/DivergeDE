"""Fit the complete 1000-gene simulation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import divergede

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "simulated" / "simulation_2"
OUTPUT = ROOT / "examples" / "output" / "simulation_2"
OUTPUT.mkdir(parents=True, exist_ok=True)

counts = pd.read_csv(DATA / "counts.csv", index_col=0)
cells = pd.read_csv(DATA / "cell_metadata.csv", index_col=0)

result = divergede.fit(
    counts,
    cells["pseudotime"],
    cells[["branch1_probability", "branch2_probability"]],
    branch_names=("Branch 1", "Branch 2"),
)
result.to_csv(OUTPUT / "summary.csv")
result.save(OUTPUT / "fit.joblib")

for index, figure in enumerate(divergede.plot_genes(result, top_n=12), start=1):
    figure.savefig(OUTPUT / f"top_genes_page_{index}.pdf", bbox_inches="tight")
    plt.close(figure)

figure = divergede.plot_bic_vs_posttau_fc(result)
figure.savefig(OUTPUT / "bic_vs_posttau_fc.pdf", bbox_inches="tight")
plt.close(figure)
