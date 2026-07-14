"""Publication-oriented plotting helpers."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable

from ._model import branch_means, evaluate_basis
from .result import DivergeDEResult

BRANCH1_COLOR = "#D55E00"
BRANCH2_COLOR = "#0072B2"
NEUTRAL_COLOR = "#B3B3B3"
PROBABILITY_CMAP = LinearSegmentedColormap.from_list(
    "divergede_probability", [BRANCH2_COLOR, "#F7F7F7", BRANCH1_COLOR]
)


def _valid_summary(result: DivergeDEResult):
    summary = result.summary
    finite = np.isfinite(summary["delta_bic"]) & np.isfinite(summary["tau"]) & np.isfinite(
        summary["terminal_log2fc"]
    )
    return summary.loc[summary["converged"].astype(bool) & finite].copy()


def _check_gene(result: DivergeDEResult, gene: str) -> None:
    if gene not in result.gene_names:
        raise KeyError(f"Gene {gene!r} is not present in this result.")
    row = result.summary.set_index("gene").loc[gene]
    if not bool(row["converged"]):
        raise ValueError(f"Gene {gene!r} did not converge; inspect result.summary before plotting.")
    if gene not in result.fits:
        raise ValueError(f"Stored fit parameters are unavailable for gene {gene!r}.")


def _scale(values: np.ndarray, y_scale: str) -> np.ndarray:
    if y_scale == "log1p":
        return np.log1p(np.clip(values, 0.0, np.inf))
    if y_scale == "linear":
        return values
    raise ValueError("y_scale must be 'log1p' or 'linear'.")


def _draw_gene(
    result: DivergeDEResult,
    gene: str,
    ax: Axes,
    y_scale: str,
    show_cells: bool,
    show_baseline: bool,
    show_tau: bool,
    show_excluded: bool,
    add_colorbar: bool,
) -> None:
    _check_gene(result, gene)
    row = result.summary.set_index("gene").loc[gene]
    fit = result.fits[gene]
    t = result.pseudotime
    counts = result.gene_counts(gene)
    if result.size_factor_mode == "none":
        display_counts = counts
        y_label = "log(1 + count)" if y_scale == "log1p" else "Count"
    else:
        display_counts = counts / result.size_factors
        y_label = "log(1 + adjusted count)" if y_scale == "log1p" else "Size-factor adjusted count"
    probabilities1 = result.branch_probabilities[:, 0]
    colors = PROBABILITY_CMAP(np.clip(probabilities1, 0.0, 1.0))

    if show_cells:
        included = result.fit_mask
        ax.scatter(
            t[included],
            _scale(display_counts[included], y_scale),
            c=colors[included],
            s=13,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        if show_excluded and np.any(~included):
            ax.scatter(
                t[~included],
                _scale(display_counts[~included], y_scale),
                c=colors[~included],
                s=13,
                alpha=0.3,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )

    t_min = float(np.min(t[result.fit_mask]))
    grid = np.linspace(t_min, result.common_terminal, 240)
    X_grid = evaluate_basis(grid, fit.basis_spec)
    baseline = np.exp(np.clip(X_grid @ fit.beta, -30.0, 30.0))
    _, mean1, mean2 = branch_means(
        baseline,
        grid,
        fit.tau,
        fit.delta1,
        fit.delta2,
        float(result.settings["kappa"]),
    )
    if show_baseline:
        ax.plot(grid, _scale(baseline, y_scale), color="#666666", linestyle="--", linewidth=1.5, label="H0 baseline")
    ax.plot(grid, _scale(mean1, y_scale), color=BRANCH1_COLOR, linewidth=2.2, label=result.branch_names[0])
    ax.plot(grid, _scale(mean2, y_scale), color=BRANCH2_COLOR, linewidth=2.2, label=result.branch_names[1])
    if show_tau:
        ax.axvline(fit.tau, color="#222222", linestyle="--", linewidth=1.2, label="tau")
    ax.axvline(result.common_terminal, color="#777777", linestyle=":", linewidth=1.2, label="common terminal")
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel(y_label)
    ax.set_title(
        f"{gene} | ΔBIC={row['delta_bic']:.2f} | τ={row['tau']:.3f} | terminal log2FC={row['terminal_log2fc']:.2f}",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    if add_colorbar and show_cells:
        scalar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=PROBABILITY_CMAP)
        colorbar = ax.figure.colorbar(scalar, ax=ax, pad=0.02)
        colorbar.set_label(f"P({result.branch_names[0]})")


def plot_gene(
    result: DivergeDEResult,
    gene: str,
    y_scale: str = "log1p",
    show_cells: bool = True,
    show_baseline: bool = True,
    show_tau: bool = True,
    show_excluded: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Plot observed expression and fitted curves for one converged gene."""
    if ax is None:
        figure, ax = plt.subplots(figsize=(7.0, 4.7))
    else:
        figure = ax.figure
    _draw_gene(
        result,
        str(gene),
        ax,
        y_scale,
        show_cells,
        show_baseline,
        show_tau,
        show_excluded,
        add_colorbar=True,
    )
    figure.tight_layout()
    return figure


def plot_genes(
    result: DivergeDEResult,
    genes: Sequence[str] | None = None,
    top_n: int = 12,
    order_by: str = "delta_bic",
    ncols: int = 3,
    max_per_page: int = 12,
    y_scale: str = "log1p",
    show_cells: bool = True,
    show_baseline: bool = True,
    show_tau: bool = True,
    show_excluded: bool = True,
) -> list[Figure]:
    """Plot converged genes in automatically paginated panels."""
    if order_by not in {"delta_bic", "tau"}:
        raise ValueError("order_by must be 'delta_bic' or 'tau'.")
    if int(top_n) < 1 or int(ncols) < 1 or int(max_per_page) < 1:
        raise ValueError("top_n, ncols, and max_per_page must be positive.")
    valid = _valid_summary(result)
    if genes is None:
        selected = valid.sort_values("delta_bic", ascending=False).head(int(top_n))
    else:
        requested = [str(gene) for gene in genes]
        missing = [gene for gene in requested if gene not in result.gene_names]
        if missing:
            raise KeyError(f"Unknown gene(s): {', '.join(missing)}")
        valid_names = set(valid["gene"])
        skipped = [gene for gene in requested if gene not in valid_names]
        if skipped:
            warnings.warn(
                f"Skipped non-converged gene(s): {', '.join(skipped)}",
                RuntimeWarning,
                stacklevel=2,
            )
        selected = valid.set_index("gene").reindex([gene for gene in requested if gene in valid_names]).reset_index()
    if selected.empty:
        raise ValueError("No converged genes are available to plot.")
    selected = selected.sort_values(order_by, ascending=order_by == "tau")
    selected_genes = selected["gene"].tolist()
    figures: list[Figure] = []
    for start in range(0, len(selected_genes), int(max_per_page)):
        page = selected_genes[start : start + int(max_per_page)]
        nrows = math.ceil(len(page) / int(ncols))
        figure, axes = plt.subplots(
            nrows,
            int(ncols),
            figsize=(5.0 * int(ncols), 3.7 * nrows),
            squeeze=False,
        )
        active_axes: list[Axes] = []
        for axis, gene in zip(axes.flat, page, strict=False):
            _draw_gene(
                result,
                gene,
                axis,
                y_scale,
                show_cells,
                show_baseline,
                show_tau,
                show_excluded,
                add_colorbar=False,
            )
            active_axes.append(axis)
        for axis in list(axes.flat)[len(page) :]:
            axis.set_visible(False)
        if show_cells:
            scalar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=PROBABILITY_CMAP)
            colorbar = figure.colorbar(scalar, ax=active_axes, pad=0.015, fraction=0.02)
            colorbar.set_label(f"P({result.branch_names[0]})")
        figure.subplots_adjust(left=0.07, right=0.92, bottom=0.08, top=0.94, wspace=0.32, hspace=0.38)
        figures.append(figure)
    return figures


def plot_bic_vs_terminal_fc(
    result: DivergeDEResult,
    genes: Sequence[str] | None = None,
    label_top: int = 0,
    bic_quantile: float = 0.75,
    log2fc_threshold: float = 1.0,
    ax: Axes | None = None,
) -> Figure:
    """Plot conditional delta BIC against terminal log2 fold change."""
    if not 0.0 <= float(bic_quantile) <= 1.0:
        raise ValueError("bic_quantile must be in [0, 1].")
    if float(log2fc_threshold) < 0 or int(label_top) < 0:
        raise ValueError("log2fc_threshold and label_top must be non-negative.")
    valid = _valid_summary(result)
    excluded = len(result.summary) - len(valid)
    if excluded:
        warnings.warn(
            f"Excluded {excluded} non-converged or non-finite gene(s) from the BIC plot.",
            RuntimeWarning,
            stacklevel=2,
        )
    if valid.empty:
        raise ValueError("No converged genes are available to plot.")
    bic_threshold = float(valid["delta_bic"].quantile(float(bic_quantile)))
    if genes is None:
        display = valid.copy()
    else:
        requested = [str(gene) for gene in genes]
        unknown = [gene for gene in requested if gene not in result.gene_names]
        if unknown:
            raise KeyError(f"Unknown gene(s): {', '.join(unknown)}")
        valid_names = set(valid["gene"])
        skipped = [gene for gene in requested if gene not in valid_names]
        if skipped:
            warnings.warn(
                f"Skipped non-converged gene(s): {', '.join(skipped)}",
                RuntimeWarning,
                stacklevel=2,
            )
        display = valid.set_index("gene").reindex([gene for gene in requested if gene in valid_names]).reset_index()
    if display.empty:
        raise ValueError("No converged requested genes are available to plot.")
    x = display["terminal_log2fc"].to_numpy(dtype=float)
    y = display["delta_bic"].to_numpy(dtype=float)
    evidence = y >= bic_threshold
    branch1 = evidence & (x >= float(log2fc_threshold))
    branch2 = evidence & (x <= -float(log2fc_threshold))
    colors = np.full(len(display), NEUTRAL_COLOR, dtype=object)
    colors[branch1] = BRANCH1_COLOR
    colors[branch2] = BRANCH2_COLOR
    if ax is None:
        figure, ax = plt.subplots(figsize=(6.5, 5.0))
    else:
        figure = ax.figure
    ax.scatter(x, y, c=colors, s=28, alpha=0.8, linewidths=0)
    ax.axhline(bic_threshold, color="#555555", linestyle="--", linewidth=1.2, label=f"ΔBIC q{int(round(100*bic_quantile))}")
    ax.axvline(float(log2fc_threshold), color="#777777", linestyle="--", linewidth=1.0)
    ax.axvline(-float(log2fc_threshold), color="#777777", linestyle="--", linewidth=1.0)
    if int(label_top):
        label_mask = branch1 | branch2
        labelled = display.loc[label_mask].nlargest(int(label_top), "delta_bic")
        for row in labelled.itertuples(index=False):
            ax.annotate(
                str(row.gene),
                (float(row.terminal_log2fc), float(row.delta_bic)),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel(f"Terminal log2FC ({result.branch_names[0]} / {result.branch_names[1]})")
    ax.set_ylabel("Conditional ΔBIC")
    ax.set_title("DivergeDE evidence and terminal effect")
    ax.legend(frameon=False)
    figure.tight_layout()
    return figure

