from types import SimpleNamespace

import numpy as np

from divergede import _model
from divergede._model import (
    DELTA_MAX,
    DELTA_MIN,
    R_MIN,
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


def test_null_r_bounds_do_not_force_nonconvergence(monkeypatch):
    boundary = {"r": R_MIN}

    def fake_minimize(*args, **kwargs):
        return SimpleNamespace(
            x=np.array([0.0, np.log(boundary["r"])]),
            success=True,
            message="optimizer converged",
        )

    monkeypatch.setattr(_model.optimize, "minimize", fake_minimize)
    for r_value in (R_MIN, R_MAX):
        boundary["r"] = r_value
        result = fit_null(
            y=np.array([0.0, 1.0, 2.0]),
            X=np.ones((3, 1)),
            log_size_factor=np.zeros(3),
            max_iter=10,
            likelihood_tolerance=1e-6,
        )
        assert np.isclose(result.r, r_value)
        assert result.converged


def test_alternative_parameter_bounds_do_not_force_nonconvergence(monkeypatch):
    boundary = {"r": R_MIN}
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
    monkeypatch.setattr(_model, "update_r", lambda *args, **kwargs: boundary["r"])
    monkeypatch.setattr(_model, "update_tau", lambda *args, **kwargs: 0.5)

    cases = (
        (R_MIN, 0.1, -0.1),
        (R_MAX, 0.1, -0.1),
        (1.0, DELTA_MIN, DELTA_MAX),
    )
    for r_value, delta1, delta2 in cases:
        boundary["r"] = r_value
        result = fit_alternative_start(
            y=np.array([0.0, 1.0, 2.0]),
            mu0=np.ones(3),
            t=np.array([0.2, 0.5, 0.8]),
            probabilities=np.full((3, 2), 0.5),
            start=(0.0, 0.5, delta1, delta2, r_value),
            kappa=12.0,
            tau_bounds=(0.1, 0.9),
            max_iter=2,
            likelihood_tolerance=1e-6,
            parameter_tolerance=1e-4,
        )
        assert np.isclose(result.r, r_value)
        assert np.isclose(result.delta1, delta1)
        assert np.isclose(result.delta2, delta2)
        assert result.converged
