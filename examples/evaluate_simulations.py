"""Report tau accuracy and detection benchmarks from saved example fits."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_merged(name: str) -> pd.DataFrame:
    summary = pd.read_csv(ROOT / "examples" / "output" / name / "summary.csv")
    truth = pd.read_csv(ROOT / "data" / "simulated" / name / "gene_truth.csv")
    return summary.merge(truth, left_on="gene", right_on="gene_id", validate="one_to_one")


def finite_converged(table: pd.DataFrame) -> pd.DataFrame:
    mask = (
        table["converged"].astype(bool)
        & np.isfinite(table["delta_bic"])
        & np.isfinite(table["tau"])
        & np.isfinite(table["terminal_log2fc"])
    )
    return table.loc[mask].copy()


def rank_auc(scores: pd.Series, labels: pd.Series) -> float:
    """Compute ROC AUC from average ranks, including score ties."""
    labels = labels.astype(bool)
    n_positive = int(labels.sum())
    n_negative = int((~labels).sum())
    ranks = scores.rank(method="average")
    rank_sum = float(ranks.loc[labels].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


simulation_1 = finite_converged(read_merged("simulation_1"))
tau_error_1 = simulation_1["tau"] - simulation_1["true_tau"]
print("Simulation 1")
print(f"  converged: {len(simulation_1)}/60")
print(f"  tau MAE: {tau_error_1.abs().mean():.4f}")
print(f"  tau correlation: {simulation_1['tau'].corr(simulation_1['true_tau']):.4f}")

simulation_2 = finite_converged(read_merged("simulation_2"))
labels = simulation_2["is_de"].astype(bool)
threshold = float(simulation_2["delta_bic"].quantile(0.75))
evidence = (simulation_2["delta_bic"] >= threshold) & (
    simulation_2["terminal_log2fc"].abs() >= 1.0
)
tau_table = simulation_2.loc[labels]
tau_error_2 = tau_table["tau"] - tau_table["true_tau"]
false_positives = int((evidence & ~labels).sum())
print("Simulation 2")
print(f"  converged: {len(simulation_2)}/1000")
print(f"  delta-BIC ROC AUC: {rank_auc(simulation_2['delta_bic'], labels):.4f}")
print(f"  evidence genes: {int(evidence.sum())}")
print(f"  null false positives: {false_positives}/{int((~labels).sum())}")
print(f"  DE-gene tau MAE: {tau_error_2.abs().mean():.4f}")
print(f"  DE-gene tau correlation: {tau_table['tau'].corr(tau_table['true_tau']):.4f}")
