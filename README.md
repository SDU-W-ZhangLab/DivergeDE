# DivergeDE

[![Tests](https://github.com/SDU-W-ZhangLab/DivergeDE/actions/workflows/tests.yml/badge.svg)](https://github.com/SDU-W-ZhangLab/DivergeDE/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.1.0-4C72B0.svg)](https://github.com/SDU-W-ZhangLab/DivergeDE)
[![License](https://img.shields.io/badge/license-MIT-2E8B57.svg)](LICENSE)

**DivergeDE detects branch-divergent genes and estimates when their expression
trajectories begin to separate.** It fits an unpenalized, gene-wise
negative-binomial mixture model to raw single-cell counts along a two-branch
pseudotime trajectory. External soft branch probabilities are retained instead
of forcing every cell into a hard lineage.

For each gene, DivergeDE reports:

- a conditional ΔBIC score for within-dataset evidence ranking;
- an estimated divergence onset time, `tau`;
- a signed terminal log2 fold change between Branch 1 and Branch 2;
- fitted branch-specific expression curves and convergence diagnostics.

<p align="center">
  <img src="docs/assets/divergede_overview.png" width="100%" alt="Overview of the DivergeDE model and downstream applications">
</p>

**Method overview.** DivergeDE distinguishes non-divergent trajectories from
genes whose branch-specific expression curves separate after an estimated
onset. It uses a negative-binomial mixture with soft lineage probabilities and
supports downstream ranking, onset-ordered visualization, and dynamic
expression-pattern interpretation. The q75 ΔBIC rule shown in the overview is
a descriptive evidence rule, not a calibrated significance threshold.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Input requirements](#input-requirements)
- [Understanding the results](#understanding-the-results)
- [Visualization](#visualization)
- [Method](#method)
- [Parameters and defaults](#parameters-and-defaults)
- [Parallel execution](#parallel-execution)
- [Reproducible examples](#reproducible-examples)
- [Scope and limitations](#scope-and-limitations)
- [Citation](#citation)

## Installation

DivergeDE requires Python 3.10 or newer. The current research release is
installed directly from GitHub:

```bash
python -m pip install "git+https://github.com/SDU-W-ZhangLab/DivergeDE.git"
```

To obtain the example data, tests, and reproducibility scripts, clone the
repository and install it in editable mode:

```bash
git clone https://github.com/SDU-W-ZhangLab/DivergeDE.git
cd DivergeDE
python -m pip install -e ".[test]"
```

Verify the installation:

```bash
python -c "import divergede; print(divergede.__version__)"
```

The expected version is `0.1.0`. Core dependencies (`numpy`, `scipy`, `pandas`,
`matplotlib`, `joblib`, `threadpoolctl`, and `tqdm`) are installed
automatically.

## Quick start

The three required inputs are a cell-by-gene count matrix, a pseudotime vector,
and a two-column matrix of soft branch probabilities.

```python
import pandas as pd
import divergede

counts = pd.read_csv(
    "data/simulated/simulation_1/counts.csv",
    index_col=0,
)
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

print(result.summary.head())
result.to_csv("divergede_summary.csv")
result.save("divergede_fit.joblib")

figure = divergede.plot_gene(result, gene="gene_001")
figure.savefig("gene_001_fit.pdf", bbox_inches="tight")
```

All genes are fitted by default. To fit a subset:

```python
result = divergede.fit(
    counts,
    pseudotime,
    branch_probabilities,
    genes=["GATA1", "SPI1"],
)
```

## Input requirements

| Input | Required format | Notes |
|---|---|---|
| `counts` | Cells × genes | Original non-negative integer counts; pandas, NumPy, and SciPy sparse matrices are supported. |
| `pseudotime` | Length = number of cells | Must already be finite and scaled to `[0, 1]`; DivergeDE does not normalize pseudotime. |
| `branch_probabilities` | Cells × 2 | Non-negative soft probabilities or weights. Each row is normalized internally to sum to one. |
| `genes` | Optional sequence | `None` fits all genes; names require a labeled pandas count matrix, while integer column indices also work. |
| `branch_names` | Optional pair of strings | Used in plots and axis labels. DataFrame probability-column names are used when available. |
| `size_factors` | Optional | `None` disables offsets; use `"library_size"` or supply a finite positive cell-level vector. |

When pandas objects are supplied, DivergeDE aligns pseudotime, branch
probabilities, and size factors to the count-matrix cell index and rejects
missing or duplicated identifiers.

### Common fitted pseudotime range

Soft probabilities are used throughout model fitting. A temporary
`argmax(p1, p2)` assignment is used only to identify the observed terminal
pseudotime of each branch. DivergeDE then fits cells through the shorter branch
endpoint:

```text
T_common = min(T_branch1, T_branch2)
```

This avoids extrapolating one branch beyond the pseudotime range supported by
the other branch. Cells after `T_common` remain stored in the result and can be
shown in plots, but they do not contribute to model fitting.

## Understanding the results

`result.summary` contains exactly one row per fitted gene:

| Column | Meaning |
|---|---|
| `gene` | Gene identifier. |
| `converged` | Whether both the null and alternative fits passed convergence and boundary checks. |
| `delta_bic` | Conditional ΔBIC ranking score; larger values indicate stronger relative evidence for branch divergence. |
| `tau` | Estimated divergence onset in the input pseudotime scale. |
| `terminal_log2fc` | Signed Branch 1 / Branch 2 log2 fold change at `T_common`. |
| `loglik_null` | Negative-binomial log-likelihood under the shared trajectory. |
| `loglik_alternative` | Observed-data mixture log-likelihood under branch divergence. |
| `r_null` | Gene-specific NB2 size parameter under the null model. |
| `r_alternative` | Gene-specific NB2 size parameter under the alternative model. |
| `n_iter` | Number of alternative-model iterations. |

Positive `terminal_log2fc` means higher fitted expression on Branch 1;
negative values mean higher fitted expression on Branch 2.

Non-converged genes remain in `result.summary` and in exported CSV files. They
are excluded from default ranking, ΔBIC quantile calculation, and plotting.

Complete results can be restored without refitting:

```python
result.save("divergede_fit.joblib")
restored = divergede.load_result("divergede_fit.joblib")
```

Only load `.joblib` files from trusted sources.

## Visualization

### One gene

```python
figure = divergede.plot_gene(
    result,
    gene="GATA1",
    y_scale="log1p",
    show_cells=True,
    show_baseline=True,
    show_tau=True,
    show_excluded=True,
)
```

Cells are colored continuously by the Branch 1 probability. Branch 1 curves
are orange-red, Branch 2 curves are blue, and the shared null trajectory is a
gray dashed line. The plotted `log1p` scale changes only the display; fitting
always uses original integer counts.

### Multiple genes

```python
pages = divergede.plot_genes(
    result,
    top_n=12,
    order_by="delta_bic",  # or "tau"
    ncols=3,
    max_per_page=12,
)

for page_number, figure in enumerate(pages, start=1):
    figure.savefig(f"divergede_page_{page_number}.pdf", bbox_inches="tight")
```

With `genes=None`, `top_n` first selects the highest-ΔBIC converged genes.
`order_by="tau"` keeps that selected set but displays it from earlier to later
onset.

### Evidence versus terminal effect

```python
figure = divergede.plot_bic_vs_terminal_fc(
    result,
    genes=None,
    label_top=0,
    bic_quantile=0.75,
    log2fc_threshold=1.0,
)
```

By default, genes satisfying both

```text
delta_bic >= q75(delta_bic)
abs(terminal_log2fc) >= 1
```

are colored orange-red for a positive Branch 1 effect and blue for a negative
Branch 2 effect. Other genes are gray. This is an exploratory evidence rule;
it is not a p-value cutoff and does not control the false-discovery rate.

## Method

### Null model

For gene `g` and cell `i`, the null model uses a cubic B-spline trajectory and
a gene-specific NB2 size parameter `r_g`:

```text
Y_gi ~ NB2(mu_g0(t_i), r_g0)
log(mu_g0(t_i)) = log(size_factor_i) + X(t_i) beta_g
Var(Y_gi) = mu_gi + mu_gi^2 / r_g
```

The default spline has five degrees of freedom. No roughness penalty is added.

### Conditional alternative

The fitted null baseline is held fixed in the alternative model. Two
branch-specific effects, a new gene-specific `r`, and the onset `tau` are then
estimated. Branch effects are exactly zero through `tau` and activate smoothly
afterward:

```text
g(t, tau) = 2 * max(sigmoid(kappa * (t - tau)) - 0.5, 0)
mu_gk(t) = mu_g0(t) * exp(delta_gk * g(t, tau))
```

The external branch probabilities enter through time-dependent mixture
weights. Before onset, weights are close to `0.5 / 0.5`; after onset, they
transition toward the supplied probabilities. The model is optimized with
multiple tau starts and monotonic observed-log-likelihood backtracking.

DivergeDE v0.1.0 uses one unpenalized main model per gene. It has no tau prior,
expression-support penalty, spline roughness penalty, cross-gene shrinkage of
`r`, or automatic `kappa` selection.

### Conditional ΔBIC

The reported ranking score is

```text
conditional_delta_BIC =
    2 * (loglik_alternative - loglik_null) - 3 * log(n_fit)
```

where `n_fit` is the number of cells through `T_common`. The penalty accounts
for the two branch effects and `tau` beyond the null model.

Because the H0 baseline is fixed rather than jointly re-optimized in H1, and
because `tau` is not identified under H0, this is not a standard nested-model
BIC. It is designed for ranking genes within the same dataset. It must not be
interpreted as a p-value or as a calibrated cross-dataset evidence scale.

## Parameters and defaults

Only the first three arguments are required:

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

| Parameter | Default | Role |
|---|---:|---|
| `spline_df` | `5` | Degrees of freedom of the shared cubic B-spline baseline. |
| `kappa` | `12.0` | Fixed steepness of post-onset branch activation. |
| `tau_quantiles` | `(0.05, 0.95)` | Allowed onset range within the common fitted pseudotime interval. |
| `tau_grid_size` | `9` | Number of initial tau grid points. |
| `n_starts` | `3` | Best grid starts retained for full optimization. |
| `max_iter` | `100` | Maximum iterations for each fit stage. |
| `likelihood_tolerance` | `1e-6` | Relative observed-log-likelihood convergence tolerance. |
| `parameter_tolerance` | `1e-4` | Relative parameter convergence tolerance. |
| `n_jobs` | `4` | Number of gene-level worker processes. |
| `verbose` | `1` | `0` is quiet, `1` reports progress, and `2` also lists failure reasons. |

Numerical safety bounds are `delta1, delta2 ∈ [-5, 5]` and
`r ∈ [1e-4, 1e6]`. A solution reaching one of these bounds is retained in the
summary but marked as non-converged.

## Parallel execution

Genes are independent and are fitted with `joblib`'s `loky` process backend,
which provides true multi-core execution rather than Python threads. Nested
BLAS threads are limited inside each worker.

```python
# Default: four worker processes
result = divergede.fit(counts, pseudotime, probabilities, n_jobs=4)

# Use all available logical CPU cores
result = divergede.fit(counts, pseudotime, probabilities, n_jobs=-1)

# Serial execution for debugging or small examples
result = divergede.fit(counts, pseudotime, probabilities, n_jobs=1)
```

Memory use increases with the number of processes. For large dense matrices,
start with `n_jobs=2` or `n_jobs=4` and increase only when memory allows.

## Reproducible examples

The repository includes two simulated datasets under `data/simulated/`. They
are intentionally excluded from the Python wheel and are available after
cloning the repository.

### Simulation 1: onset and curve accuracy

- 500 cells;
- 60 true branch-divergent genes only;
- intended for fitted-curve visualization and tau evaluation;
- not suitable by itself for detection AUC or false-positive estimation.

```bash
python examples/simulation_1.py
```

### Simulation 2: detection benchmark

- 500 cells;
- 1000 genes: 300 true DE and 700 non-DE;
- no-added-noise `noise_0/rep_001` realization;
- suitable for ΔBIC ranking and null false-positive evaluation.

```bash
python examples/simulation_2.py
```

After both fits finish, reproduce the benchmark summary with:

```bash
python examples/evaluate_simulations.py
```

Under the v0.1.0 defaults, the bundled examples produce:

| Benchmark | Result |
|---|---:|
| Simulation 1 convergence | 60 / 60 genes |
| Simulation 1 tau MAE | 0.0410 |
| Simulation 1 tau correlation | 0.9823 |
| Simulation 2 convergence | 997 / 1000 genes |
| Simulation 2 ΔBIC ROC AUC | 0.9964 |
| Simulation 2 q75 + effect evidence set | 250 genes |
| Simulation 2 null false positives in that evidence set | 0 / 700 |
| Simulation 2 DE-gene tau MAE | 0.0557 |
| Simulation 2 DE-gene tau correlation | 0.9322 |

These values validate this bundled simulation realization; they are not a
guarantee for other datasets.

## Scope and limitations

- DivergeDE currently supports exactly two branches.
- It is a downstream method: pseudotime and branch probabilities must be
  estimated before running DivergeDE.
- Pseudotime uncertainty from the upstream trajectory method is not propagated.
- Counts must be raw non-negative integers; normalized or log-transformed
  expression is rejected.
- Size-factor normalization is disabled by default and must be requested
  explicitly when appropriate for the experimental design.
- The conditional ΔBIC supports within-dataset ranking, not formal hypothesis
  testing, p-values, or false-discovery-rate control.
- Very sparse genes, weak branch support, or boundary solutions may fail to
  converge. Always inspect the `converged` column.
- Independent gene-wise fitting does not model gene-gene covariance.

## Repository layout

```text
DivergeDE/
├── src/divergede/        # package implementation
├── tests/                # unit and integration tests
├── examples/             # fitting and evaluation scripts
├── data/simulated/       # reproducible simulation datasets
├── docs/assets/          # README and manuscript figures
├── CITATION.cff          # software citation metadata
└── pyproject.toml        # build metadata and dependencies
```

Run the test suite with:

```bash
python -m pytest
```

## Citation

DivergeDE is developed by **Ling Sun** and **Naiqian Zhang** at the School of
Mathematics and Statistics, Shandong University at Weihai. A manuscript
citation will be added after publication. Until then, cite the software using
the metadata in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{sun_divergede_2026,
  author  = {Sun, Ling and Zhang, Naiqian},
  title   = {DivergeDE: Conditional negative-binomial detection of
             branch-divergent expression along pseudotime},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/SDU-W-ZhangLab/DivergeDE}
}
```

Correspondence: [nqzhang@email.sdu.edu.cn](mailto:nqzhang@email.sdu.edu.cn)

## Issues and license

Please report reproducible bugs or usage questions through
[GitHub Issues](https://github.com/SDU-W-ZhangLab/DivergeDE/issues). Include the
DivergeDE version, Python version, operating system, input shapes, and the full
error message.

DivergeDE is distributed under the [MIT License](LICENSE).
