"""Result containers and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(slots=True)
class GeneFitResult:
    beta: np.ndarray
    basis_spec: dict[str, Any]
    tau: float
    delta1: float
    delta2: float
    r_null: float
    r_alternative: float


@dataclass(slots=True)
class DivergeDEResult:
    """Complete result returned by :func:`divergede.fit`."""

    summary: pd.DataFrame
    fits: dict[str, GeneFitResult]
    counts: np.ndarray | sparse.spmatrix
    gene_names: list[str]
    cell_ids: np.ndarray
    pseudotime: np.ndarray
    branch_probabilities: np.ndarray
    size_factors: np.ndarray
    fit_mask: np.ndarray
    common_terminal: float
    branch_names: tuple[str, str]
    size_factor_mode: str
    settings: dict[str, Any] = field(default_factory=dict)
    package_version: str = "0.2.0"

    @property
    def diagnostics(self) -> pd.DataFrame:
        """Return structured per-gene fit outcomes without changing ``summary``.

        Results written by DivergeDE 0.1 did not store structured statuses.  In
        that case they are conservatively reconstructed from the legacy
        convergence flag and message.
        """
        messages = self.settings.get("messages", {})
        statuses = self.settings.get("fit_statuses", {})
        attempts = self.settings.get("refit_attempts", {})
        records = []
        indexed = self.summary.set_index("gene")
        for gene in self.gene_names:
            message = str(messages.get(gene, ""))
            status = statuses.get(gene)
            if status is None:
                status = _legacy_status(bool(indexed.loc[gene, "converged"]), message)
            records.append(
                {
                    "gene": gene,
                    "fit_status": str(status),
                    "message": message,
                    "refit_attempts": int(attempts.get(gene, 0)),
                }
            )
        return pd.DataFrame.from_records(
            records, columns=["gene", "fit_status", "message", "refit_attempts"]
        )

    def to_csv(self, path: str | Path) -> None:
        """Write the per-gene summary without a DataFrame index."""
        self.summary.to_csv(path, index=False)

    def save(self, path: str | Path) -> None:
        """Save the complete result with joblib."""
        path = Path(path)
        if path.suffix.lower() != ".joblib":
            raise ValueError("DivergeDE result files must use the .joblib extension.")
        joblib.dump(self, path, compress=3)

    def gene_counts(self, gene: str) -> np.ndarray:
        """Return one fitted gene's counts in original cell order."""
        if gene not in self.gene_names:
            raise KeyError(f"Gene {gene!r} is not present in this result.")
        index = self.gene_names.index(gene)
        if sparse.issparse(self.counts):
            return np.asarray(self.counts[:, index].toarray()).reshape(-1)
        return np.asarray(self.counts[:, index]).reshape(-1)


@dataclass(slots=True)
class ThreeBranchStageResult:
    """One two-branch numerical stage inside a three-branch fit."""

    summary: pd.DataFrame
    fits: dict[str, GeneFitResult]
    fit_mask: np.ndarray
    branch_probabilities: np.ndarray
    branch_names: tuple[str, str]
    common_terminal: float
    tau_bounds: tuple[float, float]
    messages: dict[str, str] = field(default_factory=dict)
    fit_statuses: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ThreeBranchDivergeDEResult:
    """Complete result returned by :func:`divergede.fit_three_branch`."""

    summary: pd.DataFrame
    stage1: ThreeBranchStageResult
    stage2: ThreeBranchStageResult
    counts: np.ndarray | sparse.spmatrix
    gene_names: list[str]
    cell_ids: np.ndarray
    pseudotime: np.ndarray
    branch_probabilities: np.ndarray
    size_factors: np.ndarray
    topology: tuple[str, tuple[str, str]]
    endpoints: dict[str, float]
    cell_types: np.ndarray | None
    endpoint_labels: np.ndarray
    size_factor_mode: str
    settings: dict[str, Any] = field(default_factory=dict)
    package_version: str = "0.2.0"

    @property
    def branch_names(self) -> tuple[str, str, str]:
        first, descendants = self.topology
        return first, descendants[0], descendants[1]

    @property
    def diagnostics(self) -> pd.DataFrame:
        attempts = self.settings.get("refit_attempts", {})
        records = []
        for gene in self.gene_names:
            for stage_name, stage in (("stage1", self.stage1), ("stage2", self.stage2)):
                records.append(
                    {
                        "gene": gene,
                        "stage": stage_name,
                        "fit_status": stage.fit_statuses.get(gene, "error"),
                        "message": stage.messages.get(gene, ""),
                        "refit_attempts": int(attempts.get(f"{stage_name}:{gene}", 0)),
                    }
                )
        return pd.DataFrame.from_records(
            records,
            columns=["gene", "stage", "fit_status", "message", "refit_attempts"],
        )

    def rank_genes(self, stage: str = "stage1") -> pd.DataFrame:
        """Rank converged genes within one stage by descending conditional ΔBIC."""
        if stage not in {"stage1", "stage2"}:
            raise ValueError("stage must be 'stage1' or 'stage2'.")
        status_column = f"{stage}_fit_status"
        score_column = f"{stage}_delta_bic"
        ranked = self.summary.loc[self.summary[status_column] == "converged"].copy()
        ranked = ranked.loc[np.isfinite(ranked[score_column])]
        return ranked.sort_values(score_column, ascending=False, kind="stable").reset_index(drop=True)

    def to_csv(self, path: str | Path) -> None:
        self.summary.to_csv(path, index=False)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() != ".joblib":
            raise ValueError("DivergeDE result files must use the .joblib extension.")
        joblib.dump(self, path, compress=3)

    def gene_counts(self, gene: str) -> np.ndarray:
        if gene not in self.gene_names:
            raise KeyError(f"Gene {gene!r} is not present in this result.")
        index = self.gene_names.index(gene)
        if sparse.issparse(self.counts):
            return np.asarray(self.counts[:, index].toarray()).reshape(-1)
        return np.asarray(self.counts[:, index]).reshape(-1)


def _legacy_status(converged: bool, message: str) -> str:
    if converged:
        return "converged"
    lowered = message.lower()
    if "all counts are zero" in lowered:
        return "not_fitted"
    if "maximum iterations" in lowered or (
        "iteration" in lowered and ("limit" in lowered or "reached" in lowered)
    ):
        return "max_iter"
    if any(
        token in lowered
        for token in (
            "alternative starts failed",
            "log-likelihood",
            "line search",
            "abnormal_termination",
            "nan",
            "infinite",
        )
    ):
        return "numerical_failure"
    return "error"


def load_result(path: str | Path) -> DivergeDEResult | ThreeBranchDivergeDEResult:
    """Load a complete DivergeDE joblib result."""
    result = joblib.load(path)
    if not isinstance(result, (DivergeDEResult, ThreeBranchDivergeDEResult)):
        raise TypeError("The file does not contain a DivergeDE result object.")
    return result
