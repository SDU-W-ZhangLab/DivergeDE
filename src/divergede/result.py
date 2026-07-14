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
    package_version: str = "0.1.0"

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


def load_result(path: str | Path) -> DivergeDEResult:
    """Load a complete DivergeDE joblib result."""
    result = joblib.load(path)
    if not isinstance(result, DivergeDEResult):
        raise TypeError("The file does not contain a DivergeDEResult object.")
    return result

