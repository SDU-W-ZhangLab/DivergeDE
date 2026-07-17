from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

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
        "mean_posttau_log2fc",
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
    bic_figure = divergede.plot_bic_vs_posttau_fc(result)
    assert bic_figure.axes
    assert "Prespecified ΔBIC > 10" in bic_figure.axes[0].get_legend_handles_labels()[1]
    assert not hasattr(divergede, "plot_bic_vs_terminal_fc")
    fitted_gene = converged[0]
    row = result.summary.set_index("gene").loc[fitted_gene]
    fit = result.fits[fitted_gene]
    grid = np.linspace(float(row["tau"]), result.common_terminal, 100)
    pointwise = (
        (fit.delta1 - fit.delta2)
        * divergede._model.gate(grid - fit.tau, result.settings["kappa"])
        / np.log(2.0)
    )
    expected = trapezoid(pointwise, grid) / (result.common_terminal - fit.tau)
    assert np.isclose(float(row["mean_posttau_log2fc"]), expected)
    path = tmp_path / "fit.joblib"
    result.save(path)
    restored = divergede.load_result(path)
    pd.testing.assert_frame_equal(restored.summary, result.summary)


def test_uniform_prior_fit_uses_expression_only_symmetry_breaking():
    counts, t, probabilities = _simulated_counts()
    uniform = pd.DataFrame(0.5, index=counts.index, columns=probabilities.columns)
    endpoint_labels = pd.Series(
        np.r_[np.zeros(35, dtype=int), np.ones(35, dtype=int)],
        index=counts.index,
    )
    result = divergede.fit(
        counts[["de_gene"]],
        t,
        uniform,
        endpoint_branch_labels=endpoint_labels,
        tau_grid_size=5,
        n_starts=2,
        max_iter=80,
        n_jobs=1,
        verbose=0,
    )
    row = result.summary.iloc[0]
    assert bool(row["converged"])
    assert np.isfinite(row["tau"])
    assert abs(float(row["mean_posttau_log2fc"])) > 0.1
    assert np.allclose(result.branch_probabilities, 0.5)


def test_three_branch_summary_uses_stage_mean_posttau_effects():
    rng = np.random.default_rng(27)
    branch_time = np.linspace(0.0, 1.0, 30)
    t = np.tile(branch_time, 3)
    labels = np.repeat(["A", "B", "C"], 30)
    probabilities = np.full((90, 3), 0.05)
    probabilities[np.arange(90), np.repeat(np.arange(3), 30)] = 0.90
    baseline = np.exp(1.8 + 0.4 * t)
    branch_effect = np.select(
        [labels == "A", labels == "B", labels == "C"],
        [1.2 * np.maximum(t - 0.3, 0.0), 1.0 * np.maximum(t - 0.6, 0.0), -1.0 * np.maximum(t - 0.6, 0.0)],
    )
    mean = baseline * np.exp(branch_effect)
    dispersion = 25.0
    counts = pd.DataFrame(
        {"three_branch_gene": rng.negative_binomial(dispersion, dispersion / (dispersion + mean))},
        index=[f"cell_{index}" for index in range(90)],
    )
    probability_frame = pd.DataFrame(
        probabilities,
        index=counts.index,
        columns=["A", "B", "C"],
    )
    result = divergede.fit_three_branch(
        counts,
        pd.Series(t, index=counts.index),
        probability_frame,
        topology=("A", ("B", "C")),
        cell_types=pd.Series(labels, index=counts.index),
        tau_grid_size=5,
        n_starts=2,
        max_iter=80,
        n_jobs=1,
        verbose=0,
    )
    assert "stage1_mean_posttau_log2fc" in result.summary.columns
    assert "stage2_mean_posttau_log2fc" in result.summary.columns
    assert "stage1_terminal_log2fc" not in result.summary.columns
    assert "stage2_terminal_log2fc" not in result.summary.columns
    assert np.isfinite(result.summary.loc[0, "stage1_mean_posttau_log2fc"])
    assert np.isfinite(result.summary.loc[0, "stage2_mean_posttau_log2fc"])
