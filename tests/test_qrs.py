import numpy as np
import pytest

from qrs import match_peaks


def test_match_peaks_counts_one_to_one_matches():
    assert match_peaks(
        np.array([100, 103, 250]),
        np.array([101, 250]),
        tolerance_samples=5,
    ) == (2, 1, 0, 0.5)


def test_match_peaks_rejects_negative_tolerance():
    with pytest.raises(ValueError, match="tolerance_samples"):
        match_peaks(np.array([]), np.array([]), tolerance_samples=-1)
