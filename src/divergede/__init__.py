"""Public API for DivergeDE."""

from .fitting import fit, refit_failed
from .plotting import plot_bic_vs_terminal_fc, plot_gene, plot_genes
from .result import DivergeDEResult, ThreeBranchDivergeDEResult, load_result
from .three_branch import fit_three_branch
from .three_branch_plotting import (
    get_three_branch_composite_curves,
    plot_three_branch_composite,
    plot_three_branch_stages,
)

__all__ = [
    "DivergeDEResult",
    "ThreeBranchDivergeDEResult",
    "fit",
    "fit_three_branch",
    "get_three_branch_composite_curves",
    "load_result",
    "plot_bic_vs_terminal_fc",
    "plot_gene",
    "plot_genes",
    "plot_three_branch_composite",
    "plot_three_branch_stages",
    "refit_failed",
]

__version__ = "0.2.0"
