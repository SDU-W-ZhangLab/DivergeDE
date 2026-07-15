"""Descriptive composite and audit plots for three-branch fits."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import make_smoothing_spline

from ._model import branch_means, evaluate_basis
from .result import ThreeBranchDivergeDEResult, ThreeBranchStageResult

THREE_BRANCH_COLORS = ("#D55E00", "#0072B2", "#009E73")
TAIL_MIN_CELLS = 20
TAIL_TARGET_CELLS_PER_BIN = 10
TAIL_MAX_BINS = 10
TAIL_ENDPOINT_ANCHOR_WEIGHT_MULTIPLIER = 2.0


def _stage_curves(
    stage: ThreeBranchStageResult, gene: str, grid: np.ndarray, kappa: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit = stage.fits[gene]
    basis = evaluate_basis(grid, fit.basis_spec)
    baseline = np.exp(np.clip(basis @ fit.beta, -30.0, 30.0))
    _, first, second = branch_means(
        baseline, grid, fit.tau, fit.delta1, fit.delta2, kappa
    )
    return baseline, first, second


def _resolve_one_gene(
    result: ThreeBranchDivergeDEResult,
    gene: str | None,
    genes: Sequence[str] | None,
) -> str:
    if gene is not None and genes is not None:
        raise ValueError("Pass gene or genes, not both.")
    if genes is not None:
        requested = [str(value) for value in genes]
        if len(requested) != 1:
            raise ValueError("get_three_branch_composite_curves accepts exactly one gene.")
        gene = requested[0]
    if gene is None:
        raise TypeError("A gene must be supplied.")
    resolved = str(gene)
    if resolved not in result.gene_names:
        raise KeyError(f"Gene {resolved!r} is not present in this result.")
    row = result.summary.set_index("gene").loc[resolved]
    for stage_name in ("stage1", "stage2"):
        if row[f"{stage_name}_fit_status"] != "converged":
            raise ValueError(
                f"Gene {resolved!r} did not converge in {stage_name}; "
                "inspect result.diagnostics or use plot_three_branch_stages."
            )
    return resolved


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _tail_curve(
    result: ThreeBranchDivergeDEResult,
    gene: str,
    branch: str,
    core_time: np.ndarray,
    core_mean: np.ndarray,
    n_points: int,
    min_cells: int = TAIL_MIN_CELLS,
    target_cells_per_bin: int = TAIL_TARGET_CELLS_PER_BIN,
    max_bins: int = TAIL_MAX_BINS,
) -> tuple[pd.DataFrame | None, dict[str, object]]:
    endpoint = float(result.endpoints[branch])
    core_endpoint = float(result.stage2.common_terminal)
    metadata: dict[str, object] = {
        "branch": branch,
        "used": False,
        "n_cells": 0,
        "n_bins": 0,
        "reason": "branch does not extend beyond the Stage 2 common endpoint",
    }
    if endpoint <= core_endpoint + 1e-12:
        return None, metadata
    t = result.pseudotime
    mask = (
        (result.endpoint_labels == branch)
        & (t > core_endpoint + 1e-12)
        & (t <= endpoint + 1e-12)
    )
    indices = np.flatnonzero(mask)
    metadata["n_cells"] = int(indices.size)
    if indices.size < min_cells:
        metadata["reason"] = f"fewer than {min_cells} branch-labelled tail cells"
        return None, metadata
    counts = result.gene_counts(gene) / result.size_factors
    indices = indices[np.argsort(t[indices], kind="stable")]
    n_bins = min(max_bins, max(2, int(np.ceil(indices.size / target_cells_per_bin))))
    groups = [group for group in np.array_split(indices, n_bins) if group.size]
    bin_t = np.asarray([np.mean(t[group]) for group in groups], dtype=float)
    bin_y = np.asarray([np.log1p(np.mean(counts[group])) for group in groups], dtype=float)
    bin_weights = np.sqrt(np.asarray([group.size for group in groups], dtype=float))
    keep = np.concatenate(([True], np.diff(bin_t) > 1e-12))
    bin_t, bin_y, bin_weights = bin_t[keep], bin_y[keep], bin_weights[keep]
    if bin_t.size < 2:
        metadata["reason"] = "tail bins do not contain distinct pseudotimes"
        return None, metadata

    # The last equal-frequency bin contains the branch endpoint.  Treat its
    # local mean as the endpoint estimate so the displayed tail never needs
    # unconstrained extrapolation beyond the last reliable bin.
    bin_t[-1] = endpoint

    core_log = np.log1p(np.clip(core_mean, 0.0, np.inf))
    endpoint_log = float(np.interp(core_endpoint, core_time, core_log))
    if core_time.size >= 2:
        core_slope = float(
            (core_log[-1] - core_log[-2]) / max(core_time[-1] - core_time[-2], 1e-12)
        )
    else:
        core_slope = 0.0

    fit_t = np.concatenate(([core_endpoint], bin_t))
    fit_y = np.concatenate(([endpoint_log], bin_y))
    fit_weights = np.concatenate(
        (
            [TAIL_ENDPOINT_ANCHOR_WEIGHT_MULTIPLIER * np.max(bin_weights)],
            bin_weights,
        )
    )
    smoothing_method = "gcv_cubic_smoothing_spline"
    smoother = None
    coefficients = None
    if fit_t.size >= 5:
        try:
            smoother = make_smoothing_spline(fit_t, fit_y, w=fit_weights)
            probe = np.asarray(smoother(fit_t), dtype=float)
            if not np.isfinite(probe).all():
                raise ValueError("non-finite smoothing-spline output")
        except Exception:
            smoother = None
    if smoother is None:
        degree = min(2, fit_t.size - 1)
        coefficients = np.polyfit(fit_t, fit_y, degree, w=fit_weights)
        smoothing_method = "weighted_polynomial_fallback"

    tail_grid = np.linspace(core_endpoint, endpoint, max(12, int(n_points / 3)))
    if smoother is not None:
        raw_values = np.asarray(smoother(tail_grid), dtype=float)
        raw_endpoint = float(smoother(core_endpoint))
        raw_slope = float(smoother.derivative()(core_endpoint))
        raw_bin_values = np.asarray(smoother(bin_t), dtype=float)
    else:
        raw_values = np.polyval(coefficients, tail_grid)
        raw_endpoint = float(np.polyval(coefficients, core_endpoint))
        raw_slope = float(np.polyval(np.polyder(coefficients), core_endpoint))
        raw_bin_values = np.polyval(coefficients, bin_t)

    # Preserve the screenshot-approved whole-tail reconstruction.  An affine
    # correction leaves the GCV spline curvature unchanged while matching both
    # the Stage 2 endpoint value and slope over the complete observed tail.
    value_correction = endpoint_log - raw_endpoint
    slope_correction = core_slope - raw_slope
    log_values = (
        raw_values
        + value_correction
        + slope_correction * (tail_grid - core_endpoint)
    )
    smoothed_bin_y = (
        raw_bin_values
        + value_correction
        + slope_correction * (bin_t - core_endpoint)
    )
    tail_mean = np.maximum(np.expm1(log_values), 0.0)
    metadata.update(
        {
            "used": True,
            "n_bins": int(bin_t.size),
            "reason": "observed-assisted smoothing-spline tail",
            "smoothing_method": smoothing_method,
            "endpoint_matching": "global_affine_value_and_slope",
            "endpoint_anchor_weight_multiplier": TAIL_ENDPOINT_ANCHOR_WEIGHT_MULTIPLIER,
            "bin_pseudotime": bin_t.tolist(),
            "bin_log1p_mean": bin_y.tolist(),
            "smoothed_bin_log1p_mean": np.asarray(smoothed_bin_y).tolist(),
        }
    )
    return (
        pd.DataFrame(
            {
                "gene": gene,
                "branch": branch,
                "pseudotime": tail_grid,
                "mean": tail_mean,
                "segment_type": "observed_tail",
            }
        ),
        metadata,
    )


def get_three_branch_composite_curves(
    result: ThreeBranchDivergeDEResult,
    gene: str | None = None,
    *,
    genes: Sequence[str] | None = None,
    n_points: int = 240,
) -> pd.DataFrame:
    """Return one gene's smooth descriptive three-branch composite curves.

    ``genes=[name]`` is accepted as a convenience alias for ``gene=name``.
    The returned DataFrame is on the size-factor-one count scale; metadata
    describing alignment, tau order, and optional observed-assisted tails is
    stored in ``curves.attrs``.
    """
    if not isinstance(result, ThreeBranchDivergeDEResult):
        raise TypeError("result must be returned by fit_three_branch().")
    if int(n_points) < 30:
        raise ValueError("n_points must be at least 30.")
    gene = _resolve_one_gene(result, gene, genes)
    first, second, third = result.branch_names
    row = result.summary.set_index("gene").loc[gene]
    tau1, tau2 = float(row["stage1_tau"]), float(row["stage2_tau"])
    order = str(row["tau_order"])
    kappa = float(result.settings.get("kappa", 12.0))

    common_mask = (
        result.stage1.fit_mask
        & result.stage2.fit_mask
        & (result.pseudotime <= min(tau1, tau2) + 1e-12)
    )
    common_t = result.pseudotime[common_mask]
    if common_t.size < 10 or np.unique(common_t).size < 5:
        raise ValueError(
            "Composite alignment needs at least 10 common pre-divergence cells "
            "at 5 distinct pseudotimes; use plot_three_branch_stages() instead."
        )
    _, _, stage1_common = _stage_curves(result.stage1, gene, common_t, kappa)
    stage2_baseline_common, _, _ = _stage_curves(result.stage2, gene, common_t, kappa)
    log1p_shift = float(
        np.median(np.log1p(stage1_common) - np.log1p(stage2_baseline_common))
    )

    t_min = float(
        min(
            np.min(result.pseudotime[result.stage1.fit_mask]),
            np.min(result.pseudotime[result.stage2.fit_mask]),
        )
    )
    core_terminal = float(result.stage2.common_terminal)
    core_grid = np.linspace(t_min, core_terminal, int(n_points))
    _, _, stage1_combined = _stage_curves(result.stage1, gene, core_grid, kappa)
    first_grid = np.linspace(t_min, float(result.endpoints[first]), int(n_points))
    _, stage1_first, _ = _stage_curves(result.stage1, gene, first_grid, kappa)
    stage2_baseline, stage2_second, stage2_third = _stage_curves(
        result.stage2, gene, core_grid, kappa
    )
    aligned_stage2_baseline = np.maximum(
        np.expm1(np.log1p(stage2_baseline) + log1p_shift), 0.0
    )
    aligned_second = np.maximum(
        np.expm1(np.log1p(stage2_second) + log1p_shift), 0.0
    )
    aligned_third = np.maximum(
        np.expm1(np.log1p(stage2_third) + log1p_shift), 0.0
    )

    if order == "inverse":
        center_log = np.log1p(stage1_combined)
        stage2_center_log = 0.5 * (
            np.log1p(aligned_second) + np.log1p(aligned_third)
        )
        second_mean = np.maximum(
            np.expm1(center_log + np.log1p(aligned_second) - stage2_center_log), 0.0
        )
        third_mean = np.maximum(
            np.expm1(center_log + np.log1p(aligned_third) - stage2_center_log), 0.0
        )
        handoff = None
    else:
        low, high = sorted((tau1, tau2))
        if high > low + 1e-12:
            weight = _smoothstep((core_grid - low) / (high - low))
        else:
            weight = (core_grid >= low).astype(float)
        combined_log = np.log1p(stage1_combined)
        second_mean = np.maximum(
            np.expm1((1.0 - weight) * combined_log + weight * np.log1p(aligned_second)),
            0.0,
        )
        third_mean = np.maximum(
            np.expm1((1.0 - weight) * combined_log + weight * np.log1p(aligned_third)),
            0.0,
        )
        handoff = (float(low), float(high))

    frames = []
    frames.append(
        pd.DataFrame(
            {
                "gene": gene,
                "branch": first,
                "pseudotime": first_grid,
                "mean": stage1_first,
                "segment_type": "model",
            }
        )
    )
    core_by_branch = {second: second_mean, third: third_mean}
    tail_metadata = {}
    for branch, mean in core_by_branch.items():
        frames.append(
            pd.DataFrame(
                {
                    "gene": gene,
                    "branch": branch,
                    "pseudotime": core_grid,
                    "mean": mean,
                    "segment_type": "model",
                }
            )
        )
        tail, metadata = _tail_curve(
            result, gene, branch, core_grid, mean, int(n_points)
        )
        tail_metadata[branch] = metadata
        if tail is not None:
            frames.append(tail.iloc[1:].copy())
    curves = pd.concat(frames, ignore_index=True)
    curves.attrs.update(
        {
            "gene": gene,
            "tau1": tau1,
            "tau2": tau2,
            "tau_gap": float(row["tau_gap"]),
            "tau_order": order,
            "synchronous_display_tau": float((tau1 + tau2) / 2.0),
            "handoff_interval": handoff,
            "stage2_log1p_alignment_shift": log1p_shift,
            "alignment_n_cells": int(common_t.size),
            "tail": tail_metadata,
            "stage2_aligned_baseline_terminal": float(aligned_stage2_baseline[-1]),
        }
    )
    return curves


def _display_values(
    result: ThreeBranchDivergeDEResult, gene: str, y_scale: str
) -> tuple[np.ndarray, str]:
    counts = result.gene_counts(gene)
    if result.size_factor_mode == "none":
        adjusted = counts
        label = "Count"
    else:
        adjusted = counts / result.size_factors
        label = "Size-factor adjusted count"
    if y_scale == "log1p":
        return np.log1p(adjusted), f"log(1 + {label.lower()})"
    if y_scale == "linear":
        return adjusted, label
    raise ValueError("y_scale must be 'log1p' or 'linear'.")


def plot_three_branch_composite(
    result: ThreeBranchDivergeDEResult,
    gene: str,
    n_points: int = 240,
    y_scale: str = "log1p",
    show_cells: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Plot the smooth descriptive composite for one three-branch gene."""
    curves = get_three_branch_composite_curves(result, gene, n_points=n_points)
    if ax is None:
        figure, ax = plt.subplots(figsize=(8.0, 5.0))
    else:
        figure = ax.figure
    values, ylabel = _display_values(result, gene, y_scale)
    if show_cells:
        color_map = dict(zip(result.branch_names, THREE_BRANCH_COLORS, strict=True))
        colors = [color_map.get(label, "#B3B3B3") for label in result.endpoint_labels]
        ax.scatter(
            result.pseudotime,
            values,
            c=colors,
            s=12,
            alpha=0.28,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
    for branch, color in zip(result.branch_names, THREE_BRANCH_COLORS, strict=True):
        branch_curve = curves.loc[curves["branch"] == branch]
        plotted = np.log1p(branch_curve["mean"]) if y_scale == "log1p" else branch_curve["mean"]
        ax.plot(
            branch_curve["pseudotime"],
            plotted,
            color=color,
            linestyle="-",
            linewidth=2.3,
            label=branch,
            zorder=3,
        )
    attrs = curves.attrs
    if attrs["tau_order"] == "synchronous":
        low, high = sorted((attrs["tau1"], attrs["tau2"]))
        ax.axvspan(low, high, color="#777777", alpha=0.14, label="fitted tau interval")
        ax.axvline(
            attrs["synchronous_display_tau"],
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            label="display midpoint",
        )
    else:
        ax.axvline(attrs["tau1"], color=THREE_BRANCH_COLORS[0], linestyle="--", linewidth=1.1, label="tau1")
        ax.axvline(attrs["tau2"], color=THREE_BRANCH_COLORS[1], linestyle="--", linewidth=1.1, label="tau2")
    note = (
        " (inverse expression order; topology unchanged)"
        if attrs["tau_order"] == "inverse"
        else ""
    )
    ax.set_title(f"{gene} three-branch composite | {attrs['tau_order']}{note}")
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, ncols=2)
    figure.tight_layout()
    return figure


def plot_three_branch_stages(
    result: ThreeBranchDivergeDEResult,
    gene: str,
    y_scale: str = "log1p",
    show_cells: bool = True,
) -> Figure:
    """Plot the two original fitted models without composite reconstruction."""
    if gene not in result.gene_names:
        raise KeyError(f"Gene {gene!r} is not present in this result.")
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), sharey=True)
    observed, ylabel = _display_values(result, gene, y_scale)
    kappa = float(result.settings.get("kappa", 12.0))
    for axis, stage_name, stage in zip(
        axes, ("Stage 1", "Stage 2"), (result.stage1, result.stage2), strict=True
    ):
        curve_colors = (
            (THREE_BRANCH_COLORS[0], "#666666")
            if stage_name == "Stage 1"
            else (THREE_BRANCH_COLORS[1], THREE_BRANCH_COLORS[2])
        )
        status = stage.fit_statuses.get(gene, "error")
        if show_cells:
            probability = stage.branch_probabilities[:, 0]
            colors = plt.get_cmap("coolwarm")(np.clip(probability, 0.0, 1.0))
            axis.scatter(
                result.pseudotime[stage.fit_mask],
                observed[stage.fit_mask],
                c=colors[stage.fit_mask],
                s=12,
                alpha=0.5,
                linewidths=0,
                rasterized=True,
            )
        if gene in stage.fits:
            grid = np.linspace(
                float(np.min(result.pseudotime[stage.fit_mask])),
                stage.common_terminal,
                240,
            )
            baseline, first, second = _stage_curves(stage, gene, grid, kappa)
            transform = np.log1p if y_scale == "log1p" else (lambda value: value)
            axis.plot(grid, transform(baseline), color="#666666", linestyle="--", linewidth=1.3, label="H0 baseline")
            axis.plot(grid, transform(first), color=curve_colors[0], linestyle="-", linewidth=2.1, label=stage.branch_names[0])
            axis.plot(grid, transform(second), color=curve_colors[1], linestyle="-", linewidth=2.1, label=stage.branch_names[1])
            axis.axvline(stage.fits[gene].tau, color="#222222", linestyle="--", linewidth=1.1, label="tau")
        else:
            axis.text(0.5, 0.5, "No finite fitted parameters", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(f"{stage_name}: {stage.branch_names[0]} vs {stage.branch_names[1]}\nstatus={status}")
        axis.set_xlabel("Pseudotime")
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(ylabel)
    figure.suptitle(f"{gene}: original two-stage model audit", y=1.02)
    figure.tight_layout()
    return figure
