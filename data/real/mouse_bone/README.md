# Mouse bone marrow data

## Source

The expression count matrix and cell-type annotations were derived from the
prefiltered Paul et al. (2015) mouse bone marrow dataset used in the tradeSeq
case study. The dataset contains 2,660 cells and 3,004 genes.

## Files

- `expression.csv`: raw non-negative integer counts.
- `cell_type.csv`: cell-type annotations.
- `pseudotime.csv`: Palantir pseudotime.
- `branch_probabilities.csv`: Palantir probabilities for the Erythrocyte and
  Myeloid terminal fates.

## Processing

For Palantir, counts were normalized per cell and log-transformed, and 1,500
highly variable genes were used for PCA and diffusion mapping.

For DivergeDE, analysis was restricted to 2,455 cells within the common
terminal range. Genes detected in at least 50 cells with at least 50 total
counts were retained, yielding 2,664 genes. A pseudocount of 1 was then added
once to every retained count (`y' = y + 1`), with size factors disabled. This
was an explicit modified-count analysis: τ, likelihood and Delta BIC were
obtained from the shifted-count model, while original-count-scale mean curves
were calculated as `max(μ' − 1, 0)`.

## References

- Paul F. et al. *Cell* 163, 1663–1677 (2015).
  https://doi.org/10.1016/j.cell.2015.11.013
- GEO accession: GSE72857.
- Van den Berge K. et al. *Nature Communications* 11, 1201 (2020).
  https://doi.org/10.1038/s41467-020-14766-3
- Prefiltered tradeSeq dataset:
  https://doi.org/10.5281/zenodo.3514927 (CC BY 4.0).
