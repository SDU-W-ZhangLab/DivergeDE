"""Input validation and alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(slots=True)
class PreparedData:
    counts: np.ndarray | sparse.spmatrix
    gene_names: list[str]
    cell_ids: np.ndarray
    pseudotime: np.ndarray
    probabilities: np.ndarray
    size_factors: np.ndarray
    size_factor_mode: str
    branch_names: tuple[str, str]
    fit_mask: np.ndarray
    common_terminal: float


def _aligned_vector(values, index: pd.Index | None, name: str) -> np.ndarray:
    if isinstance(values, pd.Series) and index is not None:
        if values.index.has_duplicates:
            raise ValueError(f"{name} has duplicate cell identifiers.")
        values = values.reindex(index)
        if values.isna().any():
            raise ValueError(f"{name} is missing values for one or more count-matrix cells.")
    array = np.asarray(values, dtype=float).reshape(-1)
    return array


def _validate_counts(counts) -> tuple[np.ndarray | sparse.spmatrix, list[str], pd.Index | None]:
    if isinstance(counts, pd.DataFrame):
        if counts.index.has_duplicates:
            raise ValueError("counts has duplicate cell identifiers.")
        if counts.columns.has_duplicates:
            raise ValueError("counts has duplicate gene names.")
        matrix = counts.to_numpy(dtype=float)
        names = [str(column) for column in counts.columns]
        index = counts.index
    elif sparse.issparse(counts):
        matrix = counts.tocsr().astype(float)
        names = [f"gene_{index}" for index in range(matrix.shape[1])]
        index = None
    else:
        matrix = np.asarray(counts, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("counts must be a two-dimensional cell-by-gene matrix.")
        names = [f"gene_{index}" for index in range(matrix.shape[1])]
        index = None
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("counts must contain at least one cell and one gene.")
    values = matrix.data if sparse.issparse(matrix) else matrix
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("counts must contain finite, non-negative values.")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("counts must contain integer values; transformed expression is not supported.")
    return matrix, names, index


def _select_genes(
    matrix: np.ndarray | sparse.spmatrix,
    names: list[str],
    genes: Sequence[str | int] | None,
) -> tuple[np.ndarray | sparse.spmatrix, list[str]]:
    if genes is None:
        return matrix, names
    requested = list(genes)
    if not requested:
        raise ValueError("genes cannot be an empty sequence.")
    if len(set(requested)) != len(requested):
        raise ValueError("genes contains duplicates.")
    name_to_index = {name: index for index, name in enumerate(names)}
    indices: list[int] = []
    selected_names: list[str] = []
    for gene in requested:
        if isinstance(gene, (int, np.integer)):
            index = int(gene)
            if index < 0 or index >= len(names):
                raise KeyError(f"Gene index {index} is out of range.")
        else:
            key = str(gene)
            if key not in name_to_index:
                raise KeyError(f"Gene {key!r} is not present in counts.")
            index = name_to_index[key]
        indices.append(index)
        selected_names.append(names[index])
    return matrix[:, indices], selected_names


def prepare_data(
    counts,
    pseudotime,
    branch_probabilities,
    genes,
    branch_names,
    size_factors,
) -> PreparedData:
    matrix, names, cell_index = _validate_counts(counts)
    n_cells = matrix.shape[0]
    t = _aligned_vector(pseudotime, cell_index, "pseudotime")
    if t.size != n_cells:
        raise ValueError("pseudotime length must equal the number of cells.")
    if not np.isfinite(t).all() or np.any(t < 0.0) or np.any(t > 1.0):
        raise ValueError("pseudotime must contain finite values in [0, 1]; DivergeDE does not normalize it.")

    probability_columns: tuple[str, str] | None = None
    if isinstance(branch_probabilities, pd.DataFrame):
        if branch_probabilities.index.has_duplicates:
            raise ValueError("branch_probabilities has duplicate cell identifiers.")
        if branch_probabilities.columns.has_duplicates:
            raise ValueError("branch_probabilities has duplicate branch names.")
        if branch_probabilities.shape[1] != 2:
            raise ValueError("branch_probabilities must have exactly two columns.")
        if cell_index is not None:
            branch_probabilities = branch_probabilities.reindex(cell_index)
            if branch_probabilities.isna().any().any():
                raise ValueError("branch_probabilities is missing values for one or more cells.")
        probability_columns = tuple(str(value) for value in branch_probabilities.columns)
        probabilities = branch_probabilities.to_numpy(dtype=float)
    else:
        probabilities = np.asarray(branch_probabilities, dtype=float)
    if probabilities.shape != (n_cells, 2):
        raise ValueError("branch_probabilities must have shape (n_cells, 2).")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("branch_probabilities must contain finite, non-negative values.")
    totals = probabilities.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Each branch-probability row must have a positive sum.")
    probabilities = probabilities / totals[:, None]

    if branch_names is None:
        resolved_branch_names = probability_columns or ("Branch 1", "Branch 2")
    else:
        if len(branch_names) != 2:
            raise ValueError("branch_names must contain exactly two names.")
        resolved_branch_names = (str(branch_names[0]), str(branch_names[1]))
    if resolved_branch_names[0] == resolved_branch_names[1]:
        raise ValueError("branch_names must be distinct.")

    if size_factors is None:
        factors = np.ones(n_cells, dtype=float)
        factor_mode = "none"
    elif isinstance(size_factors, str):
        if size_factors != "library_size":
            raise ValueError("size_factors must be None, 'library_size', or a positive vector.")
        totals = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(float)
        if np.any(totals <= 0) or not np.isfinite(totals).all():
            raise ValueError("Library-size factors require every cell to have a positive total count.")
        median = float(np.median(totals))
        factors = totals / median
        factor_mode = "library_size"
    else:
        factors = _aligned_vector(size_factors, cell_index, "size_factors")
        if factors.size != n_cells or not np.isfinite(factors).all() or np.any(factors <= 0):
            raise ValueError("size_factors must be finite, positive, and match the number of cells.")
        factor_mode = "provided"

    branch1 = probabilities[:, 0] > probabilities[:, 1]
    branch2 = probabilities[:, 1] > probabilities[:, 0]
    if not np.any(branch1) or not np.any(branch2):
        raise ValueError("Both branches need at least one probability-dominant cell to define their endpoints.")
    endpoint1 = float(np.max(t[branch1]))
    endpoint2 = float(np.max(t[branch2]))
    common_terminal = min(endpoint1, endpoint2)
    fit_mask = t <= common_terminal + 1e-12
    if not np.any(fit_mask & branch1) or not np.any(fit_mask & branch2):
        raise ValueError("The common pseudotime range must retain cells supporting both branches.")

    selected_matrix, selected_names = _select_genes(matrix, names, genes)
    if cell_index is None:
        cell_ids = np.asarray([f"cell_{index}" for index in range(n_cells)], dtype=object)
    else:
        cell_ids = cell_index.astype(str).to_numpy()
    return PreparedData(
        counts=selected_matrix,
        gene_names=selected_names,
        cell_ids=cell_ids,
        pseudotime=t,
        probabilities=probabilities,
        size_factors=factors,
        size_factor_mode=factor_mode,
        branch_names=resolved_branch_names,
        fit_mask=fit_mask,
        common_terminal=common_terminal,
    )
