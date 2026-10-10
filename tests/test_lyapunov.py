"""The Lyapunov spectrum, and the exact identity that licenses trusting it.

``sum_i lambda_i`` must equal the time-average of ``tr J``, which for Lorenz is
the constant ``-(sigma + 1 + beta) = -13.6667``. Nothing is fitted to make that
come out; if the variational integration or the QR bookkeeping were wrong, the
sum would miss. This is the report's warrant for quoting lambda_1.
"""

from __future__ import annotations

import numpy as np
import pytest

from lorenz63.lyapunov import (
    benettin_spectrum,
    finite_time_exponents,
    kaplan_yorke_dimension,
)
from lorenz63.system import LorenzParams, divergence

CLASSICAL = LorenzParams()

#: Published value for the classical parameters (Sprott, "Chaos and Time-Series
#: Analysis", 2003, table 4.1). Our runs must land on it independently.
LAMBDA_1_REFERENCE = 0.9056


@pytest.fixture(scope="module")
def spectrum():
    return benettin_spectrum(CLASSICAL, total_time=400.0, renorm_interval=1.0)


def test_spectrum_sums_to_the_divergence(spectrum) -> None:
    """The exact identity. Tight tolerance on purpose -- this is not a fit."""
    assert spectrum.total == pytest.approx(divergence(CLASSICAL), abs=1e-6)


def test_largest_exponent_matches_the_published_value(spectrum) -> None:
    assert spectrum.largest == pytest.approx(LAMBDA_1_REFERENCE, abs=0.02)


def test_middle_exponent_vanishes(spectrum) -> None:
    """Every bounded, non-equilibrium flow has a zero exponent along the flow."""
    assert spectrum.spectrum[1] == pytest.approx(0.0, abs=0.02)


def test_third_exponent_is_strongly_negative(spectrum) -> None:
    assert spectrum.spectrum[2] == pytest.approx(-14.57, abs=0.05)


def test_spectrum_is_sorted_descending(spectrum) -> None:
    assert np.all(np.diff(spectrum.spectrum) <= 0.0)


def test_running_estimate_converges(spectrum) -> None:
    """The tail of the history must be closer to the answer than the head."""
    early = spectrum.history[: len(spectrum.history) // 10, 0]
    late = spectrum.history[-len(spectrum.history) // 10 :, 0]
    assert abs(late.mean() - spectrum.largest) < abs(early.mean() - spectrum.largest)
    assert late.std() < early.std()


def test_derived_quantities(spectrum) -> None:
    assert spectrum.kaplan_yorke == pytest.approx(2.06, abs=0.02)
    assert spectrum.doubling_time == pytest.approx(np.log(2.0) / spectrum.largest)
    # Double-precision round-off sets the practical prediction horizon.
    assert spectrum.predictability_horizon(1e-16) == pytest.approx(41.0, abs=2.0)


def test_benettin_rejects_bad_windows() -> None:
    with pytest.raises(ValueError):
        benettin_spectrum(CLASSICAL, total_time=10.0, renorm_interval=0.0)
    with pytest.raises(ValueError):
        benettin_spectrum(CLASSICAL, total_time=0.5, renorm_interval=1.0)


def test_kaplan_yorke_closed_form() -> None:
    # Hand-checkable case: 2 + 0.9/15 = 2.06.
    assert kaplan_yorke_dimension(np.array([0.9, 0.0, -15.0])) == pytest.approx(2.06)
    # Every exponent negative -> dimension 0 (a stable fixed point).
    assert kaplan_yorke_dimension(np.array([-1.0, -2.0, -3.0])) == 0.0
    # Unsorted input must give the same answer as sorted.
    assert kaplan_yorke_dimension(np.array([-15.0, 0.9, 0.0])) == pytest.approx(2.06)


@pytest.mark.slow
def test_finite_time_exponents_average_to_the_global_one() -> None:
    """Local stretching varies widely; only its mean is lambda_1."""
    ftle = finite_time_exponents(CLASSICAL, total_time=300.0, window=0.5)
    assert ftle.mean == pytest.approx(LAMBDA_1_REFERENCE, abs=0.1)
    # The spread is the substance: instantaneous rates are nothing like 0.906,
    # and a good fraction of windows are locally contracting.
    assert ftle.values.std() > 0.5
    assert (ftle.values < 0.0).mean() > 0.05
