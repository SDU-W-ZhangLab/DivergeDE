from types import SimpleNamespace

import numpy as np

from divergede import _model
from divergede._model import (
    R_MAX,
    branch_log_weights,
    fit_alternative_start,
    fit_null,
    gate,
)


def test_gate_is_zero_before_tau_and_increases_after_tau():
    values = gate(np.array([-0.2, 0.0, 0.1, 0.5]), 12.0)
    assert np.allclose(values[:2], 0.0)
    assert 0.0 < values[2] < values[3] < 1.0


def test_branch_weights_are_normalized():
    t = np.array([0.0, 0.5, 1.0])
    probabilities = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
    log_q1, log_q2 = branch_log_weights(t, 0.5, probabilities)
    assert np.allclose(np.exp(log_q1) + np.exp(log_q2), 1.0)


def test_null_r_upper_bound_does_not_force_nonconvergence(monkeypatch):
    def fake_minimize(*args, **kwargs):
        return SimpleNamespace(
            x=np.array([0.0, np.log(R_MAX)]),
            success=True,
            message="converged at upper r bound",
        )

    monkeypatch.setattr(_model.optimize, "minimize", fake_minimize)
    result = fit_null(
        y=np.array([0.0, 1.0, 2.0]),
        X=np.ones((3, 1)),
        log_size_factor=np.zeros(3),
        max_iter=10,
        likelihood_tolerance=1e-6,
    )
    assert np.isclose(result.r, R_MAX)
    assert result.converged


def test_alternative_r_upper_bound_does_not_force_nonconvergence(monkeypatch):
    monkeypatch.setattr(_model, "observed_loglik", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        _model,
        "e_step",
        lambda y, *args, **kwargs: (
            np.full_like(y, 0.5, dtype=float),
            np.full_like(y, 0.5, dtype=float),
        ),
    )
    monkeypatch.setattr(
        _model,
        "update_delta",
        lambda y, mu0, activation, weights, r, initial: initial,
    )
    monkeypatch.setattr(_model, "update_r", lambda *args, **kwargs: R_MAX)
    monkeypatch.setattr(_model, "update_tau", lambda *args, **kwargs: 0.5)

    result = fit_alternative_start(
        y=np.array([0.0, 1.0, 2.0]),
        mu0=np.ones(3),
        t=np.array([0.2, 0.5, 0.8]),
        probabilities=np.full((3, 2), 0.5),
        start=(0.0, 0.5, 0.1, -0.1, R_MAX),
        kappa=12.0,
        tau_bounds=(0.1, 0.9),
        max_iter=2,
        likelihood_tolerance=1e-6,
        parameter_tolerance=1e-4,
    )
    assert np.isclose(result.r, R_MAX)
    assert result.converged
