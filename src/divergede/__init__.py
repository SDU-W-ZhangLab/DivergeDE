"""Public API for DivergeDE."""

from .fitting import fit
from .plotting import plot_bic_vs_terminal_fc, plot_gene, plot_genes
from .result import DivergeDEResult, load_result

__all__ = [
    "DivergeDEResult",
    "fit",
    "load_result",
    "plot_bic_vs_terminal_fc",
    "plot_gene",
    "plot_genes",
]

__version__ = "0.1.0"

