"""Public fitting interface."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import sparse
from tqdm.auto import tqdm

from ._model import (
    AlternativeFit,
    fit_alternative_start,
    fit_null,
    gate,
    make_basis,
    observed_loglik,
    quick_start,
)
from ._validation import PreparedData, prepare_data
from .result import DivergeDEResult, GeneFitResult, ThreeBranchDivergeDEResult

SUMMARY_COLUMNS = [
    "gene",
    "converged",
    "delta_bic",
    "tau",
    "terminal_log2fc",
    "loglik_null",
    "loglik_alternative",
    "r_null",
    "r_alternative",
    "n_iter",
]


def _gene_vector(matrix: np.ndarray | sparse.spmatrix, index: int) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix[:, index].toarray(), dtype=float).reshape(-1)
    return np.asarray(matrix[:, index], dtype=float).reshape(-1)


def _empty_record(gene: str) -> dict[str, object]:
    return {
        "gene": gene,
        "converged": False,
        "delta_bic": np.nan,
        "tau": np.nan,
        "terminal_log2fc": np.nan,
        "loglik_null": np.nan,
        "loglik_alternative": np.nan,
        "r_null": np.nan,
        "r_alternative": np.nan,
        "n_iter": np.nan,
    }


def _fit_one_gene(
    index: int,
    gene: str,
    counts: np.ndarray | sparse.spmatrix,
    fit_mask: np.ndarray,
    t_fit: np.ndarray,
    probabilities_fit: np.ndarray,
    log_size_factor_fit: np.ndarray,
    X: np.ndarray,
    basis_spec: dict[str, object],
    common_terminal: float,
    kappa: float,
    tau_quantiles: tuple[float, float],
    tau_grid_size: int,
    n_starts: int,
    max_iter: int,
    likelihood_tolerance: float,
    parameter_tolerance: float,
    warm_start: GeneFitResult | None = None,
) -> tuple[int, dict[str, object], GeneFitResult | None, str, str]:
    record = _empty_record(gene)
    try:
        y_all = _gene_vector(counts, index)
        y = y_all[fit_mask]
        if not np.any(y > 0):
            return index, record, None, "all counts are zero", "not_fitted"

        null = fit_null(
            y,
            X,
            log_size_factor_fit,
            max_iter,
            likelihood_tolerance,
            initial_beta=None if warm_start is None else warm_start.beta,
            initial_r=None if warm_start is None else warm_start.r_null,
        )
        mu0 = np.exp(np.clip(log_size_factor_fit + X @ null.beta, -30.0, 30.0))
        tau_lower = float(np.quantile(t_fit, tau_quantiles[0]))
        tau_upper = float(np.quantile(t_fit, tau_quantiles[1]))
        if not tau_upper > tau_lower:
            return index, record, None, "tau bounds are not distinct", "numerical_failure"
        tau_bounds = (tau_lower, tau_upper)
        grid = np.linspace(tau_lower, tau_upper, tau_grid_size)
        starts = [
            quick_start(y, mu0, t_fit, probabilities_fit, float(tau), null.r, kappa)
            for tau in grid
        ]
        warm_candidate = None
        if warm_start is not None:
            warm_tau = float(np.clip(warm_start.tau, *tau_bounds))
            warm_score = observed_loglik(
                y,
                mu0,
                t_fit,
                probabilities_fit,
                warm_tau,
                warm_start.delta1,
                warm_start.delta2,
                warm_start.r_alternative,
                kappa,
            )
            warm_candidate = (
                warm_score,
                warm_tau,
                float(warm_start.delta1),
                float(warm_start.delta2),
                float(warm_start.r_alternative),
            )
        starts.sort(key=lambda value: value[0], reverse=True)
        starts = starts[: min(n_starts, len(starts))]
        if warm_candidate is not None:
            starts.insert(0, warm_candidate)
        alternatives: list[AlternativeFit] = [
            fit_alternative_start(
                y,
                mu0,
                t_fit,
                probabilities_fit,
                start,
                kappa,
                tau_bounds,
                max_iter,
                likelihood_tolerance,
                parameter_tolerance,
            )
            for start in starts
        ]
        finite = [value for value in alternatives if np.isfinite(value.loglik)]
        if not finite:
            return index, record, None, "all alternative starts failed", "numerical_failure"
        converged = [value for value in finite if value.converged]
        alternative = max(converged or finite, key=lambda value: value.loglik)
        overall_converged = bool(null.converged and alternative.converged)
        if alternative.loglik < null.loglik - 1e-7 * (1.0 + abs(null.loglik)):
            overall_converged = False
            message = "alternative log-likelihood is below the null log-likelihood"
        elif not null.converged:
            message = f"H0: {null.message}"
        elif not alternative.converged:
            message = f"H1: {alternative.message}"
        else:
            message = "converged"

        if overall_converged:
            fit_status = "converged"
        elif _is_max_iter_message(message):
            fit_status = "max_iter"
        else:
            fit_status = "numerical_failure"

        n_fit = int(y.size)
        delta_bic = 2.0 * (alternative.loglik - null.loglik) - 3.0 * np.log(n_fit)
        terminal_activation = float(gate(np.array([common_terminal - alternative.tau]), kappa)[0])
        terminal_log2fc = (
            (alternative.delta1 - alternative.delta2) * terminal_activation / np.log(2.0)
        )
        record.update(
            {
                "converged": overall_converged,
                "delta_bic": float(delta_bic),
                "tau": float(alternative.tau),
                "terminal_log2fc": float(terminal_log2fc),
                "loglik_null": float(null.loglik),
                "loglik_alternative": float(alternative.loglik),
                "r_null": float(null.r),
                "r_alternative": float(alternative.r),
                "n_iter": int(alternative.n_iter),
            }
        )
        details = GeneFitResult(
            beta=np.asarray(null.beta),
            basis_spec=basis_spec,
            tau=float(alternative.tau),
            delta1=float(alternative.delta1),
            delta2=float(alternative.delta2),
            r_null=float(null.r),
            r_alternative=float(alternative.r),
        )
        return index, record, details, message, fit_status
    except Exception as error:  # Keep one bad gene from stopping a genome-wide fit.
        return index, record, None, f"{type(error).__name__}: {error}", "error"


def _is_max_iter_message(message: str) -> bool:
    lowered = str(message).lower()
    return "maximum iterations" in lowered or (
        "iteration" in lowered and ("limit" in lowered or "reached" in lowered)
    )


def _validate_options(
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
) -> None:
    if int(spline_df) < 4:
        raise ValueError("spline_df must be at least 4 for a cubic B-spline.")
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive.")
    if len(tau_quantiles) != 2 or not 0 <= tau_quantiles[0] < tau_quantiles[1] <= 1:
        raise ValueError("tau_quantiles must satisfy 0 <= low < high <= 1.")
    if int(tau_grid_size) < 2:
        raise ValueError("tau_grid_size must be at least 2.")
    if int(n_starts) < 1 or int(n_starts) > int(tau_grid_size):
        raise ValueError("n_starts must be between 1 and tau_grid_size.")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be positive.")
    if likelihood_tolerance <= 0 or parameter_tolerance <= 0:
        raise ValueError("Convergence tolerances must be positive.")
    if int(n_jobs) == 0:
        raise ValueError("n_jobs cannot be zero.")
    if int(verbose) not in (0, 1, 2):
        raise ValueError("verbose must be 0, 1, or 2.")


def fit(
    counts,
    pseudotime,
    branch_probabilities,
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
    n_jobs: int = 4,
    verbose: int = 1,
) -> DivergeDEResult:
    """Fit DivergeDE to all genes or a selected subset.

    The three positional arguments are the only required inputs. Counts are
    modeled on their original integer scale with an unpenalized NB2 model.
    """
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
    prepared: PreparedData = prepare_data(
        counts,
        pseudotime,
        branch_probabilities,
        genes,
        branch_names,
        size_factors,
    )
    fit_mask = prepared.fit_mask
    t_fit = prepared.pseudotime[fit_mask]
    minimum_cells = max(10, int(spline_df) + 4)
    if t_fit.size < minimum_cells:
        raise ValueError(
            f"The common pseudotime range retains {t_fit.size} cells; at least {minimum_cells} are required."
        )
    X, basis_spec = make_basis(t_fit, int(spline_df))
    probabilities_fit = prepared.probabilities[fit_mask]
    log_size_factor_fit = np.log(prepared.size_factors[fit_mask])
    tasks = (
        delayed(_fit_one_gene)(
            index,
            gene,
            prepared.counts,
            fit_mask,
            t_fit,
            probabilities_fit,
            log_size_factor_fit,
            X,
            basis_spec,
            prepared.common_terminal,
            float(kappa),
            (float(tau_quantiles[0]), float(tau_quantiles[1])),
            int(tau_grid_size),
            int(n_starts),
            int(max_iter),
            float(likelihood_tolerance),
            float(parameter_tolerance),
        )
        for index, gene in enumerate(prepared.gene_names)
    )
    total = len(prepared.gene_names)
    if int(n_jobs) == 1:
        results = [
            _fit_one_gene(
                index,
                gene,
                prepared.counts,
                fit_mask,
                t_fit,
                probabilities_fit,
                log_size_factor_fit,
                X,
                basis_spec,
                prepared.common_terminal,
                float(kappa),
                (float(tau_quantiles[0]), float(tau_quantiles[1])),
                int(tau_grid_size),
                int(n_starts),
                int(max_iter),
                float(likelihood_tolerance),
                float(parameter_tolerance),
            )
            for index, gene in tqdm(
                enumerate(prepared.gene_names),
                total=total,
                desc="Fitting genes",
                disable=int(verbose) == 0,
            )
        ]
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            generated = Parallel(
                n_jobs=int(n_jobs),
                return_as="generator_unordered",
                batch_size="auto",
                max_nbytes="10M",
            )(tasks)
            results = list(
                tqdm(generated, total=total, desc="Fitting genes", disable=int(verbose) == 0)
            )
    results.sort(key=lambda value: value[0])
    records = [value[1] for value in results]
    summary = pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)
    summary["converged"] = summary["converged"].astype(bool)
    fits = {
        prepared.gene_names[index]: value[2]
        for index, value in enumerate(results)
        if value[2] is not None
    }
    messages = {prepared.gene_names[index]: value[3] for index, value in enumerate(results)}
    fit_statuses = {prepared.gene_names[index]: value[4] for index, value in enumerate(results)}
    n_failed = int((~summary["converged"]).sum())
    if int(verbose) >= 1:
        print(f"DivergeDE finished: {total - n_failed}/{total} genes converged; {n_failed} did not.")
    if int(verbose) >= 2 and n_failed:
        for gene in summary.loc[~summary["converged"], "gene"]:
            print(f"  {gene}: {messages[str(gene)]}")
    elif n_failed:
        warnings.warn(
            f"{n_failed} gene(s) did not converge and will be excluded from ranking and plots.",
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
        "n_jobs": int(n_jobs),
        "verbose": int(verbose),
        "score_type": "conditional_delta_bic",
        "messages": messages,
        "fit_statuses": fit_statuses,
        "refit_attempts": {gene: 0 for gene in prepared.gene_names},
    }
    return DivergeDEResult(
        summary=summary,
        fits=fits,
        counts=prepared.counts,
        gene_names=prepared.gene_names,
        cell_ids=prepared.cell_ids,
        pseudotime=prepared.pseudotime,
        branch_probabilities=prepared.probabilities,
        size_factors=prepared.size_factors,
        fit_mask=prepared.fit_mask,
        common_terminal=prepared.common_terminal,
        branch_names=prepared.branch_names,
        size_factor_mode=prepared.size_factor_mode,
        settings=settings,
    )


def refit_failed(
    result: DivergeDEResult | ThreeBranchDivergeDEResult,
    reason: str = "max_iter",
    max_iter: int = 300,
    verbose: int = 1,
) -> DivergeDEResult | ThreeBranchDivergeDEResult:
    """Retry failed fits from their best finite parameters.

    Only ``max_iter`` outcomes are retried automatically: other statuses need
    changed data or numerical investigation rather than simply more iterations.
    A new result is returned; the input result is never modified.
    """
    if reason != "max_iter":
        raise ValueError("reason must be 'max_iter'; other failures are not safe automatic retries.")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be positive.")
    if int(verbose) not in (0, 1, 2):
        raise ValueError("verbose must be 0, 1, or 2.")
    if isinstance(result, ThreeBranchDivergeDEResult):
        from .three_branch import _refit_three_branch

        return _refit_three_branch(result, int(max_iter), int(verbose))
    if not isinstance(result, DivergeDEResult):
        raise TypeError("result must be a DivergeDE fitting result.")

    updated = deepcopy(result)
    diagnostics = result.diagnostics.set_index("gene")
    selected = [
        gene for gene in result.gene_names if diagnostics.loc[gene, "fit_status"] == reason
    ]
    if not selected:
        if verbose:
            print("DivergeDE refit: no max_iter genes to retry.")
        return updated

    fit_mask = result.fit_mask
    t_fit = result.pseudotime[fit_mask]
    X, basis_spec = make_basis(t_fit, int(result.settings.get("spline_df", 5)))
    probabilities_fit = result.branch_probabilities[fit_mask]
    log_size_factor_fit = np.log(result.size_factors[fit_mask])
    settings = result.settings
    status_map = dict(settings.get("fit_statuses", {}))
    message_map = dict(settings.get("messages", {}))
    attempts = dict(settings.get("refit_attempts", {}))
    gene_to_index = {gene: index for index, gene in enumerate(result.gene_names)}
    for gene in selected:
        index = gene_to_index[gene]
        outcome = _fit_one_gene(
            index,
            gene,
            result.counts,
            fit_mask,
            t_fit,
            probabilities_fit,
            log_size_factor_fit,
            X,
            basis_spec,
            result.common_terminal,
            float(settings.get("kappa", 12.0)),
            tuple(settings.get("tau_quantiles", (0.05, 0.95))),
            int(settings.get("tau_grid_size", 9)),
            int(settings.get("n_starts", 3)),
            int(max_iter),
            float(settings.get("likelihood_tolerance", 1e-6)),
            float(settings.get("parameter_tolerance", 1e-4)),
            warm_start=result.fits.get(gene),
        )
        _, record, details, message, status = outcome
        row_index = updated.summary.index[updated.summary["gene"] == gene][0]
        for column in SUMMARY_COLUMNS:
            updated.summary.at[row_index, column] = record[column]
        if details is not None:
            updated.fits[gene] = details
        message_map[gene] = message
        status_map[gene] = status
        attempts[gene] = int(attempts.get(gene, 0)) + 1
    updated.settings = dict(updated.settings)
    updated.settings.update(
        {
            "max_iter": int(max_iter),
            "messages": message_map,
            "fit_statuses": status_map,
            "refit_attempts": attempts,
        }
    )
    if verbose:
        converged = sum(status_map.get(gene) == "converged" for gene in selected)
        print(f"DivergeDE refit: {converged}/{len(selected)} retried fits converged.")
    return updated
