import numpy as np
import pandas as pd
import pytest

from divergede._validation import prepare_data


def test_prepare_data_normalizes_probabilities_and_selects_common_range():
    counts = pd.DataFrame(
        [[1, 0], [2, 1], [3, 0], [1, 2]],
        index=["c1", "c2", "c3", "c4"],
        columns=["g1", "g2"],
    )
    t = pd.Series([0.0, 0.3, 0.7, 1.0], index=counts.index)
    probabilities = pd.DataFrame(
        [[9, 1], [2, 8], [8, 2], [1, 9]],
        index=counts.index,
        columns=["A", "B"],
    )
    prepared = prepare_data(counts, t, probabilities, None, None, None)
    assert np.allclose(prepared.probabilities.sum(axis=1), 1.0)
    assert prepared.branch_names == ("A", "B")
    assert prepared.common_terminal == 0.7


def test_transformed_counts_are_rejected():
    with pytest.raises(ValueError, match="integer"):
        prepare_data(
            np.array([[0.1], [1.0]]),
            np.array([0.0, 1.0]),
            np.array([[0.9, 0.1], [0.1, 0.9]]),
            None,
            None,
            None,
        )
