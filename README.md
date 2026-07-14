# DivergeDE

DivergeDE fits an unpenalized, gene-wise negative-binomial model to detect
branch-divergent expression along a two-branch pseudotime trajectory. It uses
external soft branch probabilities, estimates a gene-specific divergence time
(`tau`), and ranks genes with a conditional delta BIC score.

## Installation

```bash
pip install .
```

DivergeDE requires Python 3.10 or newer.

## Quick start

```python
import pandas as pd
import divergede

counts = pd.read_csv("data/simulated/simulation_1/counts.csv", index_col=0)
cells = pd.read_csv(
    "data/simulated/simulation_1/cell_metadata.csv",
    index_col=0,
)

result = divergede.fit(
    counts,
    cells["pseudotime"],
    cells[["branch1_probability", "branch2_probability"]],
    branch_names=("Branch 1", "Branch 2"),
)

result.summary.to_csv("divergede_results.csv", index=False)
fig = divergede.plot_gene(result, gene="gene_001")
```

Only `counts`, `pseudotime`, and `branch_probabilities` are required. The
default fit is equivalent to:

```python
result = divergede.fit(
    counts,
    pseudotime,
    branch_probabilities,
    genes=None,
    branch_names=None,
    size_factors=None,
    spline_df=5,
    kappa=12.0,
    tau_quantiles=(0.05, 0.95),
    tau_grid_size=9,
    n_starts=3,
    max_iter=100,
    likelihood_tolerance=1e-6,
    parameter_tolerance=1e-4,
    n_jobs=4,
    verbose=1,
)
```

`genes=None` fits all genes. A list such as `genes=["GATA1", "SPI1"]`
fits only the requested genes. `size_factors=None` disables size-factor
offsets. Use `size_factors="library_size"` for total-count factors or pass a
positive vector of user-supplied factors.

Input pseudotime must already lie in `[0, 1]`; DivergeDE never rescales it.
Counts must be original, non-negative integers. Pandas DataFrames, NumPy
arrays, and SciPy sparse matrices are supported.

## Model summary

The null model uses a cubic B-spline baseline with a gene-specific NB2
parameter `r`:

```text
log(mu0_i) = log(size_factor_i) + X(t_i) beta
Var(Y_i)   = mu_i + mu_i^2 / r
```

The alternative keeps the fitted H0 baseline fixed and estimates two branch
effects, a new gene-specific `r`, and `tau`. Branch effects are zero through
`tau` and turn on smoothly afterward. The model contains no spline roughness
penalty, tau prior, expression-support penalty, or cross-gene shrinkage of
`r`.

The reported score is

```text
conditional_delta_BIC = 2 * (loglik_H1 - loglik_H0) - 3 * log(n_fit)
```

where `n_fit` is the number of cells retained through the shorter branch's
terminal pseudotime. Because the H0 baseline is fixed in H1 and `tau` is not
identified under H0, this score is intended for within-dataset gene ranking.
It is not a p-value or a calibrated false-discovery measure.

## Results and plots

```python
result.summary
result.to_csv("summary.csv")
result.save("fit.joblib")
result = divergede.load_result("fit.joblib")
```

The summary contains one row per gene with convergence status, conditional
delta BIC, tau, terminal log2 fold change, likelihoods, H0/H1 values of `r`,
and the H1 iteration count. Non-converged genes remain in the table but are
excluded from ranking and default plots.

```python
fig = divergede.plot_gene(result, "gene_001")

pages = divergede.plot_genes(
    result,
    top_n=12,
    order_by="tau",       # or "delta_bic"
    ncols=3,
    max_per_page=12,
)

fig = divergede.plot_bic_vs_terminal_fc(
    result,
    bic_quantile=0.75,
    log2fc_threshold=1.0,
    label_top=0,
)
```

In the BIC-effect plot, genes above the delta-BIC q75 threshold and with
`abs(terminal_log2fc) >= 1` are orange when Branch 1 is higher and blue when
Branch 2 is higher. Other genes are gray. This coloring is descriptive and is
not an inferential significance call.

## Parallel execution

Genes are fitted in separate `joblib`/`loky` processes. The default is
`n_jobs=4`. Use `n_jobs=1` for serial execution, `n_jobs=-1` for all available
CPU cores, or `n_jobs=-2` to leave one core unused. Nested BLAS threads are
limited inside workers.

## Example data

- `simulation_1`: 500 cells and 60 true DE genes. It is intended for curve
  visualization and tau evaluation, not AUC or false-positive estimation.
- `simulation_2`: 500 cells and 1000 genes, including 300 true DE and 700
  non-DE genes. It supports detection-ranking benchmarks.

See `data/README.md` and the scripts in `examples/`.
After running both fitting examples, use `python examples/evaluate_simulations.py`
to reproduce the ROC AUC, tau-error, and null false-positive summaries.

## Citation

DivergeDE is developed by Ling Sun and Naiqian Zhang at the School of
Mathematics and Statistics, Shandong University at Weihai. Citation metadata
are provided in `CITATION.cff`.

## License

MIT License.
