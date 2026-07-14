import numpy as np

from divergede._model import branch_log_weights, gate


def test_gate_is_zero_before_tau_and_increases_after_tau():
    values = gate(np.array([-0.2, 0.0, 0.1, 0.5]), 12.0)
    assert np.allclose(values[:2], 0.0)
    assert 0.0 < values[2] < values[3] < 1.0


def test_branch_weights_are_normalized():
    t = np.array([0.0, 0.5, 1.0])
    probabilities = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
    log_q1, log_q2 = branch_log_weights(t, 0.5, probabilities)
    assert np.allclose(np.exp(log_q1) + np.exp(log_q2), 1.0)

