"""Fit and plot the 60-DE-gene simulation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import divergede

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "simulated" / "simulation_1"
OUTPUT = ROOT / "examples" / "output" / "simulation_1"
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

top_gene = result.summary.loc[result.summary["converged"]].nlargest(1, "delta_bic").iloc[0]["gene"]
figure = divergede.plot_gene(result, top_gene)
figure.savefig(OUTPUT / "top_gene.pdf", bbox_inches="tight")
plt.close(figure)

pages = divergede.plot_genes(result, top_n=12, order_by="tau")
for index, figure in enumerate(pages, start=1):
    figure.savefig(OUTPUT / f"top_genes_page_{index}.pdf", bbox_inches="tight")
    plt.close(figure)

figure = divergede.plot_bic_vs_terminal_fc(result)
figure.savefig(OUTPUT / "bic_vs_terminal_fc.pdf", bbox_inches="tight")
plt.close(figure)

