"""Explicit two-stage fitting for a three-terminal-branch topology."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import sparse
from tqdm.auto import tqdm

from ._model import make_basis
from ._validation import _aligned_vector, _select_genes, _validate_counts
from .fitting import SUMMARY_COLUMNS, _fit_one_gene, _validate_options
from .result import (
    GeneFitResult,
    ThreeBranchDivergeDEResult,
    ThreeBranchStageResult,
)

ZERO_PROBABILITY_TOLERANCE = 1e-12
THREE_BRANCH_SUMMARY_COLUMNS = [
    "gene",
    "stage1_fit_status",
    "stage1_delta_bic",
    "stage1_tau",
    "stage1_terminal_log2fc",
    "stage1_loglik_null",
    "stage1_loglik_alternative",
    "stage1_r_null",
    "stage1_r_alternative",
    "stage1_n_iter",
    "stage2_fit_status",
    "stage2_delta_bic",
    "stage2_tau",
    "stage2_terminal_log2fc",
    "stage2_loglik_null",
    "stage2_loglik_alternative",
    "stage2_r_null",
    "stage2_r_alternative",
    "stage2_n_iter",
    "tau_gap",
    "tau_order",
]


def _resolve_topology(topology) -> tuple[str, tuple[str, str]]:
    if isinstance(topology, (str, bytes)):
        raise ValueError("topology must have the form ('first', ('second', 'third')).")
    try:
        first, descendants = topology
        if isinstance(descendants, (str, bytes)):
            raise ValueError
        second, third = descendants
    except (TypeError, ValueError) as error:
        raise ValueError(
            "topology must have the form ('first', ('second', 'third'))."
        ) from error
    resolved = (str(first), (str(second), str(third)))
    if len(set((resolved[0], *resolved[1]))) != 3:
        raise ValueError("topology must name three distinct terminal branches.")
    return resolved


def _prepare_three_branch_data(
    counts,
    pseudotime,
    branch_probabilities,
    genes,
    branch_names,
    size_factors,
    topology,
    cell_types,
):
    matrix, names, cell_index = _validate_counts(counts)
    n_cells = matrix.shape[0]
    t = _aligned_vector(pseudotime, cell_index, "pseudotime")
    if t.size != n_cells:
        raise ValueError("pseudotime length must equal the number of cells.")
    if not np.isfinite(t).all() or np.any(t < 0.0) or np.any(t > 1.0):
        raise ValueError(
            "pseudotime must contain finite values in [0, 1]; DivergeDE does not normalize it."
        )

    probability_columns = None
    if isinstance(branch_probabilities, pd.DataFrame):
        if branch_probabilities.index.has_duplicates:
            raise ValueError("branch_probabilities has duplicate cell identifiers.")
        if branch_probabilities.columns.has_duplicates:
            raise ValueError("branch_probabilities has duplicate branch names.")
        if branch_probabilities.shape[1] != 3:
            raise ValueError("branch_probabilities must have exactly three columns.")
        if cell_index is not None:
            branch_probabilities = branch_probabilities.reindex(cell_index)
            if branch_probabilities.isna().any().any():
                raise ValueError(
                    "branch_probabilities is missing values for one or more count-matrix cells."
                )
        probability_columns = tuple(str(value) for value in branch_probabilities.columns)
        probabilities = branch_probabilities.to_numpy(dtype=float)
    else:
        probabilities = np.asarray(branch_probabilities, dtype=float)
    if probabilities.shape != (n_cells, 3):
        raise ValueError("branch_probabilities must have shape (n_cells, 3).")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("branch_probabilities must contain finite, non-negative values.")
    totals = probabilities.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("Each branch-probability row must have a positive sum.")
    probabilities = probabilities / totals[:, None]

    resolved_topology = _resolve_topology(topology)
    topology_names = (resolved_topology[0], *resolved_topology[1])
    if branch_names is None:
        if probability_columns is None:
            raise ValueError(
                "branch_names is required when branch_probabilities is not a DataFrame."
            )
        resolved_names = probability_columns
    else:
        if len(branch_names) != 3:
            raise ValueError("branch_names must contain exactly three names.")
        resolved_names = tuple(str(value) for value in branch_names)
    if len(set(resolved_names)) != 3:
        raise ValueError("branch_names must be distinct.")
    missing = [name for name in topology_names if name not in resolved_names]
    if missing:
        raise ValueError(f"topology branch(es) absent from branch probabilities: {', '.join(missing)}")
    order = [resolved_names.index(name) for name in topology_names]
    probabilities = probabilities[:, order]

    if cell_types is None:
        maxima = probabilities.max(axis=1)
        tied = probabilities == maxima[:, None]
        unique = tied.sum(axis=1) == 1
        argmax = probabilities.argmax(axis=1)
        labels = np.full(n_cells, None, dtype=object)
        labels[unique] = np.asarray(topology_names, dtype=object)[argmax[unique]]
        first_tied = tied[:, 0] & ~unique
        stored_types = None
    else:
        if isinstance(cell_types, pd.Series) and cell_index is not None:
            if cell_types.index.has_duplicates:
                raise ValueError("cell_types has duplicate cell identifiers.")
            aligned_types = cell_types.reindex(cell_index)
            if aligned_types.isna().any():
                raise ValueError("cell_types is missing values for one or more count-matrix cells.")
            labels = aligned_types.astype(str).to_numpy(dtype=object)
        else:
            raw_types = np.asarray(cell_types, dtype=object).reshape(-1)
            if raw_types.size != n_cells:
                raise ValueError("cell_types length must equal the number of cells.")
            if pd.isna(raw_types).any():
                raise ValueError("cell_types cannot contain missing values.")
            labels = raw_types.astype(str)
        first_tied = np.zeros(n_cells, dtype=bool)
        stored_types = labels.copy()

    endpoints: dict[str, float] = {}
    for name in topology_names:
        mask = labels == name
        if not np.any(mask):
            source = "cell_types" if cell_types is not None else "probability-dominant labels"
            raise ValueError(f"Terminal branch {name!r} has no matching cells in {source}.")
        endpoints[name] = float(np.max(t[mask]))

    first, second, third = topology_names
    stage1_mask = t <= endpoints[first] + ZERO_PROBABILITY_TOLERANCE
    remainder = probabilities[:, 1] + probabilities[:, 2]
    stage2_mask = (
        (t <= min(endpoints[second], endpoints[third]) + ZERO_PROBABILITY_TOLERANCE)
        & (labels != first)
        & ~first_tied
        & (remainder > ZERO_PROBABILITY_TOLERANCE)
    )
    stage1_probabilities = np.column_stack((probabilities[:, 0], remainder))
    stage2_probabilities = np.zeros((n_cells, 2), dtype=float)
    stage2_probabilities[stage2_mask] = probabilities[stage2_mask, 1:3] / remainder[
        stage2_mask, None
    ]

    selected_matrix, selected_names = _select_genes(matrix, names, genes)
    if size_factors is None:
        factors = np.ones(n_cells, dtype=float)
        factor_mode = "none"
    elif isinstance(size_factors, str):
        if size_factors != "library_size":
            raise ValueError("size_factors must be None, 'library_size', or a positive vector.")
        libraries = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(float)
        if np.any(libraries <= 0) or not np.isfinite(libraries).all():
            raise ValueError("Library-size factors require every cell to have a positive total count.")
        factors = libraries / float(np.median(libraries))
        factor_mode = "library_size"
    else:
        factors = _aligned_vector(size_factors, cell_index, "size_factors")
        if factors.size != n_cells or not np.isfinite(factors).all() or np.any(factors <= 0):
            raise ValueError("size_factors must be finite, positive, and match the number of cells.")
        factor_mode = "provided"

    if cell_index is None:
        cell_ids = np.asarray([f"cell_{index}" for index in range(n_cells)], dtype=object)
    else:
        cell_ids = cell_index.astype(str).to_numpy()
    excluded_zero = int(
        np.sum(
            (t <= min(endpoints[second], endpoints[third]) + ZERO_PROBABILITY_TOLERANCE)
            & (labels != first)
            & ~first_tied
            & (remainder <= ZERO_PROBABILITY_TOLERANCE)
        )
    )
    return {
        "counts": selected_matrix,
        "gene_names": selected_names,
        "cell_ids": cell_ids,
        "pseudotime": t,
        "probabilities": probabilities,
        "size_factors": factors,
        "size_factor_mode": factor_mode,
        "topology": resolved_topology,
        "labels": labels,
        "cell_types": stored_types,
        "endpoints": endpoints,
        "stage1_mask": stage1_mask,
        "stage2_mask": stage2_mask,
        "stage1_probabilities": stage1_probabilities,
        "stage2_probabilities": stage2_probabilities,
        "excluded_zero": excluded_zero,
    }


def _run_stage(
    *,
    stage_name: str,
    counts,
    gene_names: list[str],
    pseudotime: np.ndarray,
    probabilities: np.ndarray,
    size_factors: np.ndarray,
    fit_mask: np.ndarray,
    branch_names: tuple[str, str],
    common_terminal: float,
    spline_df: int,
    kappa: float,
    tau_quantiles: tuple[float, float],
    tau_grid_size: int,
    n_starts: int,
    max_iter: int,
    likelihood_tolerance: float,
    parameter_tolerance: float,
    n_jobs: int,
    verbose: int,
) -> ThreeBranchStageResult:
    t_fit = pseudotime[fit_mask]
    minimum_cells = max(10, int(spline_df) + 4)
    if t_fit.size < minimum_cells:
        raise ValueError(
            f"{stage_name} retains {t_fit.size} cells; at least {minimum_cells} are required."
        )
    tau_bounds = tuple(float(value) for value in np.quantile(t_fit, tau_quantiles))
    if not tau_bounds[1] > tau_bounds[0]:
        raise ValueError(f"{stage_name} tau q05-q95 bounds are not distinct.")
    X, basis_spec = make_basis(t_fit, int(spline_df))
    probability_fit = probabilities[fit_mask]
    if np.any(probability_fit.sum(axis=1) <= 0):
        raise ValueError(f"{stage_name} contains a zero-sum probability row after filtering.")
    if np.any(probability_fit.sum(axis=0) <= ZERO_PROBABILITY_TOLERANCE):
        raise ValueError(f"{stage_name} lacks soft-probability support for one fitted branch.")
    log_size_factor_fit = np.log(size_factors[fit_mask])

    def call(index: int, gene: str):
        return _fit_one_gene(
            index,
            gene,
            counts,
            fit_mask,
            t_fit,
            probability_fit,
            log_size_factor_fit,
            X,
            basis_spec,
            common_terminal,
            float(kappa),
            tau_quantiles,
            int(tau_grid_size),
            int(n_starts),
            int(max_iter),
            float(likelihood_tolerance),
            float(parameter_tolerance),
        )

    if int(n_jobs) == 1:
        results = [
            call(index, gene)
            for index, gene in tqdm(
                enumerate(gene_names),
                total=len(gene_names),
                desc=f"Fitting {stage_name}",
                disable=int(verbose) == 0,
            )
        ]
    else:
        tasks = (delayed(call)(index, gene) for index, gene in enumerate(gene_names))
        with parallel_config(backend="loky", inner_max_num_threads=1):
            generated = Parallel(
                n_jobs=int(n_jobs),
                return_as="generator_unordered",
                batch_size="auto",
                max_nbytes="10M",
            )(tasks)
            results = list(
                tqdm(
                    generated,
                    total=len(gene_names),
                    desc=f"Fitting {stage_name}",
                    disable=int(verbose) == 0,
                )
            )
    results.sort(key=lambda value: value[0])
    records = [value[1] for value in results]
    statuses = {gene_names[index]: value[4] for index, value in enumerate(results)}
    messages = {gene_names[index]: value[3] for index, value in enumerate(results)}
    for index, record in enumerate(records):
        if statuses[gene_names[index]] != "converged":
            for column in SUMMARY_COLUMNS[2:]:
                record[column] = np.nan
    summary = pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)
    summary["converged"] = summary["converged"].astype(bool)
    fits = {
        gene_names[index]: value[2]
        for index, value in enumerate(results)
        if value[2] is not None
    }
    return ThreeBranchStageResult(
        summary=summary,
        fits=fits,
        fit_mask=fit_mask.copy(),
        branch_probabilities=probabilities.copy(),
        branch_names=branch_names,
        common_terminal=float(common_terminal),
        tau_bounds=tau_bounds,
        messages=messages,
        fit_statuses=statuses,
    )


def _assemble_summary(
    gene_names: list[str],
    stage1: ThreeBranchStageResult,
    stage2: ThreeBranchStageResult,
    tau_order_tolerance: float,
) -> pd.DataFrame:
    first = stage1.summary.set_index("gene")
    second = stage2.summary.set_index("gene")
    records = []
    metric_names = SUMMARY_COLUMNS[2:]
    for gene in gene_names:
        record: dict[str, object] = {
            "gene": gene,
            "stage1_fit_status": stage1.fit_statuses[gene],
            "stage2_fit_status": stage2.fit_statuses[gene],
        }
        for prefix, table in (("stage1", first), ("stage2", second)):
            for metric in metric_names:
                record[f"{prefix}_{metric}"] = table.loc[gene, metric]
        if (
            stage1.fit_statuses[gene] == "converged"
            and stage2.fit_statuses[gene] == "converged"
        ):
            gap = float(record["stage2_tau"] - record["stage1_tau"])
            if gap >= tau_order_tolerance:
                order = "canonical"
            elif gap <= -tau_order_tolerance:
                order = "inverse"
            else:
                order = "synchronous"
            record["tau_gap"] = gap
            record["tau_order"] = order
        else:
            record["tau_gap"] = np.nan
            record["tau_order"] = None
        records.append(record)
    return pd.DataFrame.from_records(records, columns=THREE_BRANCH_SUMMARY_COLUMNS)


def fit_three_branch(
    counts,
    pseudotime,
    branch_probabilities,
    *,
    topology,
    cell_types=None,
    genes: Sequence[str | int] | None = None,
    branch_names: Sequence[str] | None = None,
    size_factors=None,
    spline_df: int = 5,
    kappa: float = 12.0,
    tau_quantiles: tuple[float, float] = (0.05, 0.95),
    tau_grid_size: int = 9,
    n_starts: int = 3,
    max_iter: int = 100,
    likelihood_tolerance: float = 1e-6,
    parameter_tolerance: float = 1e-4,
    tau_order_tolerance: float = 0.1,
    n_jobs: int = 4,
    verbose: int = 1,
) -> ThreeBranchDivergeDEResult:
    """Fit an explicitly supplied three-branch topology in two numerical stages."""
    _validate_options(
        spline_df,
        kappa,
        tau_quantiles,
        tau_grid_size,
        n_starts,
        max_iter,
        likelihood_tolerance,
        parameter_tolerance,
        n_jobs,
        verbose,
    )
    if not np.isfinite(tau_order_tolerance) or tau_order_tolerance <= 0:
        raise ValueError("tau_order_tolerance must be finite and positive.")
    prepared = _prepare_three_branch_data(
        counts,
        pseudotime,
        branch_probabilities,
        genes,
        branch_names,
        size_factors,
        topology,
        cell_types,
    )
    first, (second, third) = prepared["topology"]
    stage1 = _run_stage(
        stage_name="stage1",
        counts=prepared["counts"],
        gene_names=prepared["gene_names"],
        pseudotime=prepared["pseudotime"],
        probabilities=prepared["stage1_probabilities"],
        size_factors=prepared["size_factors"],
        fit_mask=prepared["stage1_mask"],
        branch_names=(first, f"{second}+{third}"),
        common_terminal=prepared["endpoints"][first],
        spline_df=spline_df,
        kappa=kappa,
        tau_quantiles=tuple(map(float, tau_quantiles)),
        tau_grid_size=tau_grid_size,
        n_starts=n_starts,
        max_iter=max_iter,
        likelihood_tolerance=likelihood_tolerance,
        parameter_tolerance=parameter_tolerance,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    stage2_terminal = min(prepared["endpoints"][second], prepared["endpoints"][third])
    stage2 = _run_stage(
        stage_name="stage2",
        counts=prepared["counts"],
        gene_names=prepared["gene_names"],
        pseudotime=prepared["pseudotime"],
        probabilities=prepared["stage2_probabilities"],
        size_factors=prepared["size_factors"],
        fit_mask=prepared["stage2_mask"],
        branch_names=(second, third),
        common_terminal=stage2_terminal,
        spline_df=spline_df,
        kappa=kappa,
        tau_quantiles=tuple(map(float, tau_quantiles)),
        tau_grid_size=tau_grid_size,
        n_starts=n_starts,
        max_iter=max_iter,
        likelihood_tolerance=likelihood_tolerance,
        parameter_tolerance=parameter_tolerance,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    summary = _assemble_summary(
        prepared["gene_names"], stage1, stage2, float(tau_order_tolerance)
    )
    failed = int(
        (summary["stage1_fit_status"] != "converged").sum()
        + (summary["stage2_fit_status"] != "converged").sum()
    )
    if verbose:
        total = 2 * len(prepared["gene_names"])
        print(f"DivergeDE three-branch finished: {total - failed}/{total} stage fits converged.")
        if prepared["excluded_zero"]:
            print(
                f"Stage 2 excluded {prepared['excluded_zero']} cell(s) with "
                f"P({second})+P({third}) <= {ZERO_PROBABILITY_TOLERANCE:g}."
            )
    elif failed:
        warnings.warn(
            f"{failed} stage fit(s) did not converge; inspect result.diagnostics.",
            RuntimeWarning,
            stacklevel=2,
        )
    settings = {
        "spline_df": int(spline_df),
        "kappa": float(kappa),
        "tau_quantiles": tuple(map(float, tau_quantiles)),
        "tau_grid_size": int(tau_grid_size),
        "n_starts": int(n_starts),
        "max_iter": int(max_iter),
        "likelihood_tolerance": float(likelihood_tolerance),
        "parameter_tolerance": float(parameter_tolerance),
        "tau_order_tolerance": float(tau_order_tolerance),
        "n_jobs": int(n_jobs),
        "verbose": int(verbose),
        "stage2_zero_probability_threshold": ZERO_PROBABILITY_TOLERANCE,
        "stage2_excluded_zero_probability": int(prepared["excluded_zero"]),
        "refit_attempts": {},
        "score_type": "stage_specific_conditional_delta_bic",
    }
    return ThreeBranchDivergeDEResult(
        summary=summary,
        stage1=stage1,
        stage2=stage2,
        counts=prepared["counts"],
        gene_names=prepared["gene_names"],
        cell_ids=prepared["cell_ids"],
        pseudotime=prepared["pseudotime"],
        branch_probabilities=prepared["probabilities"],
        size_factors=prepared["size_factors"],
        topology=prepared["topology"],
        endpoints=prepared["endpoints"],
        cell_types=prepared["cell_types"],
        endpoint_labels=prepared["labels"],
        size_factor_mode=prepared["size_factor_mode"],
        settings=settings,
    )


def _refit_three_branch(
    result: ThreeBranchDivergeDEResult, max_iter: int, verbose: int
) -> ThreeBranchDivergeDEResult:
    updated = deepcopy(result)
    settings = result.settings
    attempts = dict(settings.get("refit_attempts", {}))
    total_selected = 0
    total_converged = 0
    for stage_name in ("stage1", "stage2"):
        source_stage = getattr(result, stage_name)
        target_stage = getattr(updated, stage_name)
        selected = [
            gene
            for gene in result.gene_names
            if source_stage.fit_statuses.get(gene) == "max_iter"
        ]
        total_selected += len(selected)
        if not selected:
            continue
        fit_mask = source_stage.fit_mask
        t_fit = result.pseudotime[fit_mask]
        X, basis_spec = make_basis(t_fit, int(settings.get("spline_df", 5)))
        log_size = np.log(result.size_factors[fit_mask])
        gene_to_index = {gene: index for index, gene in enumerate(result.gene_names)}
        for gene in selected:
            index = gene_to_index[gene]
            outcome = _fit_one_gene(
                index,
                gene,
                result.counts,
                fit_mask,
                t_fit,
                source_stage.branch_probabilities[fit_mask],
                log_size,
                X,
                basis_spec,
                source_stage.common_terminal,
                float(settings.get("kappa", 12.0)),
                tuple(settings.get("tau_quantiles", (0.05, 0.95))),
                int(settings.get("tau_grid_size", 9)),
                int(settings.get("n_starts", 3)),
                int(max_iter),
                float(settings.get("likelihood_tolerance", 1e-6)),
                float(settings.get("parameter_tolerance", 1e-4)),
                warm_start=source_stage.fits.get(gene),
            )
            _, record, details, message, status = outcome
            if status != "converged":
                for column in SUMMARY_COLUMNS[2:]:
                    record[column] = np.nan
            row_index = target_stage.summary.index[target_stage.summary["gene"] == gene][0]
            for column in SUMMARY_COLUMNS:
                target_stage.summary.at[row_index, column] = record[column]
            if details is not None:
                target_stage.fits[gene] = details
            target_stage.messages[gene] = message
            target_stage.fit_statuses[gene] = status
            attempts[f"{stage_name}:{gene}"] = int(
                attempts.get(f"{stage_name}:{gene}", 0)
            ) + 1
            total_converged += status == "converged"
    updated.summary = _assemble_summary(
        updated.gene_names,
        updated.stage1,
        updated.stage2,
        float(settings.get("tau_order_tolerance", 0.1)),
    )
    updated.settings = dict(updated.settings)
    updated.settings.update({"max_iter": int(max_iter), "refit_attempts": attempts})
    if verbose:
        if total_selected:
            print(
                f"DivergeDE three-branch refit: {total_converged}/{total_selected} "
                "retried stage fits converged."
            )
        else:
            print("DivergeDE three-branch refit: no max_iter stage fits to retry.")
    return updated
