# Data

The repository contains two compact simulated datasets used by the examples.
They are not installed in the Python wheel.

Each dataset contains:

- `counts.csv`: cells by genes, with original integer counts.
- `cell_metadata.csv`: pseudotime, size factor, and two soft branch probabilities.
- `gene_truth.csv`: simulated gene-level parameters and DE truth.

Future real datasets belong under `data/real/<dataset_name>/`. Large or
restricted datasets should be distributed through a public archive; this
repository should then contain only download and preprocessing instructions.

