from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import divergede


def _simulated_counts():
    rng = np.random.default_rng(11)
    branch_time = np.linspace(0.0, 1.0, 35)
    t = np.concatenate([branch_time, branch_time])
    probabilities = np.column_stack(
        [np.concatenate([np.full(35, 0.9), np.full(35, 0.1)]), np.concatenate([np.full(35, 0.1), np.full(35, 0.9)])]
    )
    activation = np.maximum(t - 0.4, 0.0)
    base = np.exp(1.5 + 0.6 * t)
    mean_de = base * np.exp(np.where(np.arange(70) < 35, 2.0 * activation, -1.5 * activation))
    mean_null = np.exp(1.2 + 0.4 * t)
    r = 20.0
    p_de = r / (r + mean_de)
    p_null = r / (r + mean_null)
    counts = pd.DataFrame(
        {
            "de_gene": rng.negative_binomial(r, p_de),
            "null_gene": rng.negative_binomial(r, p_null),
            "zero_gene": np.zeros(70, dtype=int),
        },
        index=[f"cell_{index}" for index in range(70)],
    )
    return counts, pd.Series(t, index=counts.index), pd.DataFrame(probabilities, index=counts.index, columns=["A", "B"])


def test_fit_persistence_and_plots(tmp_path: Path):
    counts, t, probabilities = _simulated_counts()
    result = divergede.fit(
        counts,
        t,
        probabilities,
        tau_grid_size=5,
        n_starts=2,
        max_iter=80,
        n_jobs=1,
        verbose=0,
    )
    assert list(result.summary.columns) == [
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
    zero = result.summary.set_index("gene").loc["zero_gene"]
    assert not bool(zero["converged"])
    assert np.isnan(zero["delta_bic"])
    converged = result.summary.loc[result.summary["converged"], "gene"].tolist()
    assert converged
    figure = divergede.plot_gene(result, converged[0])
    assert figure.axes
    pages = divergede.plot_genes(result, genes=converged, max_per_page=1)
    assert len(pages) == len(converged)
    bic_figure = divergede.plot_bic_vs_terminal_fc(result)
    assert bic_figure.axes
    path = tmp_path / "fit.joblib"
    result.save(path)
    restored = divergede.load_result(path)
    pd.testing.assert_frame_equal(restored.summary, result.summary)

