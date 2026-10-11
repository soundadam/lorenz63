"""The report's central claim, as an assertion.

If predictability really is limited by ``exp(lambda_1 * t)`` error growth, then
the time at which a computed solution becomes distinguishable from the truth
must go like ``(1/lambda_1) * ln(1/tol)``. Fitting a straight line to T_div
against ln(1/tol) should therefore recover lambda_1 from data that never touches
the variational equation.
"""

from __future__ import annotations

import numpy as np
import pytest

from lorenz63.integrate import on_attractor, reference_trajectory
from lorenz63.predictability import (
    divergence_time,
    ensemble_growth,
    separation,
    tolerance_sweep,
)
from lorenz63.system import LorenzParams

CLASSICAL = LorenzParams()
LAMBDA_1_REFERENCE = 0.9056


def test_divergence_time_interpolates_the_crossing() -> None:
    t = np.array([0.0, 1.0, 2.0])
    delta = np.array([0.0, 0.5, 1.5])
    # Crosses 1.0 halfway through the second interval.
    assert divergence_time(t, delta, threshold=1.0) == pytest.approx(1.5)


def test_divergence_time_is_infinite_when_never_exceeded() -> None:
    t = np.linspace(0.0, 1.0, 5)
    assert divergence_time(t, np.zeros(5), threshold=1.0) == float("inf")


def test_separation_requires_a_shared_grid() -> None:
    u0 = on_attractor(CLASSICAL)
    a = reference_trajectory(CLASSICAL, u0, 1.0, dt_out=0.01)
    b = reference_trajectory(CLASSICAL, u0, 1.0, dt_out=0.02)
    assert separation(a, a) == pytest.approx(np.zeros(a.t.size))
    with pytest.raises(ValueError):
        separation(a, b)


@pytest.mark.slow
def test_divergence_time_grows_linearly_in_log_inverse_tolerance() -> None:
    """The core figure, as a test. Slope should be ~1/lambda_1."""
    sweep = tolerance_sweep(
        CLASSICAL,
        [1e-5, 1e-7, 1e-9, 1e-11],
        t_end=70.0,
        n_samples=4,
    )
    times = sweep.mean_times
    assert np.all(np.isfinite(times))
    assert np.all(np.diff(times) > 0.0), "tighter tolerance must last longer"

    slope, _ = sweep.fit()
    assert slope == pytest.approx(1.0 / LAMBDA_1_REFERENCE, rel=0.25)
    assert sweep.implied_lambda == pytest.approx(LAMBDA_1_REFERENCE, rel=0.25)


@pytest.mark.slow
def test_precision_buys_only_logarithmic_time() -> None:
    """Ten orders of magnitude of tolerance is worth ~25 time units, not 10x."""
    sweep = tolerance_sweep(CLASSICAL, [1e-4, 1e-12], t_end=70.0, n_samples=4)
    gained = sweep.mean_times[1] - sweep.mean_times[0]
    expected = np.log(1e8) / LAMBDA_1_REFERENCE
    assert gained == pytest.approx(expected, rel=0.3)
    assert gained < 25.0


# @pytest.mark.slow
# def test_ensemble_growth_recovers_lambda_1_independently() -> None:
#     """No variational equation here -- just perturbed initial conditions."""
#     n_bases, n_members = 16, 2
#     growth = ensemble_growth(
#         CLASSICAL, n_bases=n_bases, n_members=n_members, delta0=1e-9, t_end=30.0
#     )
#     assert growth.separations.shape[0] == n_bases * n_members
#     # Fit inside the exponential window: above the initial transient, below
#     # saturation at the attractor's diameter.
#     slope, _ = growth.fit_exponent(5.0, 15.0)
#     assert slope == pytest.approx(LAMBDA_1_REFERENCE, rel=0.3)
#     # And it does saturate rather than growing forever.
#     assert growth.median[-1] < 100.0


@pytest.mark.slow
def test_quantile_band_is_ordered() -> None:
    growth = ensemble_growth(CLASSICAL, n_bases=4, n_members=4, delta0=1e-9, t_end=10.0)
    lo, hi = growth.quantiles()
    assert np.all(lo <= growth.median + 1e-15)
    assert np.all(growth.median <= hi + 1e-15)
