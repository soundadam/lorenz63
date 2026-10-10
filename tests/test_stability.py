"""RK4's stability region, the Jacobian spectrum, and the bifurcation sweep.

The point being defended: on Lorenz, linear stability is *not* what limits the
step size. Accuracy is. A step ten times smaller than the stability bound is
still nowhere near enough to track a trajectory for long.
"""

from __future__ import annotations

import numpy as np
import pytest

from lorenz63.bifurcation import c_plus_stability, rho_sweep
from lorenz63.stability import (
    RK4_IMAG_AXIS_LIMIT,
    RK4_REAL_AXIS_LIMIT,
    attractor_eigenvalues,
    max_stable_step,
    rk4_stability_function,
    rk4_stability_mask,
    spectral_extent,
)
from lorenz63.system import LorenzParams, hopf_rho

CLASSICAL = LorenzParams()


def test_stability_function_is_the_exponential_truncated() -> None:
    z = np.array([0.0, 0.1, -0.3 + 0.2j])
    assert rk4_stability_function(z)[0] == pytest.approx(1.0)
    # 4th-order accurate: R(z) - exp(z) = O(z^5).
    small = 1e-3
    err = abs(rk4_stability_function(np.array([small]))[0] - np.exp(small))
    assert err < 1e-14


def test_axis_limits_are_where_the_modulus_crosses_one() -> None:
    for limit, direction in ((RK4_REAL_AXIS_LIMIT, -1.0), (RK4_IMAG_AXIS_LIMIT, 1j)):
        inside = abs(rk4_stability_function(np.array([0.99 * limit * direction]))[0])
        outside = abs(rk4_stability_function(np.array([1.01 * limit * direction]))[0])
        assert inside < 1.0 < outside


def test_stability_mask_contains_the_origin() -> None:
    re = np.linspace(-4.0, 2.0, 61)
    im = np.linspace(-4.0, 4.0, 81)
    mask = rk4_stability_mask(re, im)
    assert mask.shape == (im.size, re.size)
    assert mask[np.argmin(abs(im)), np.argmin(abs(re))]


def test_attractor_spectrum_has_both_signs() -> None:
    """A third of the spectrum is expanding -- these are physical, not numerical."""
    extent = spectral_extent(CLASSICAL, t_end=100.0)
    assert extent["max_real"] > 0.0
    assert extent["min_real"] < -20.0
    assert 0.2 < extent["fraction_expanding"] < 0.5
    assert 20.0 < extent["max_abs"] < 30.0


def test_stability_bound_is_far_looser_than_practice() -> None:
    """h < ~0.12 suffices for stability; nobody integrating Lorenz uses 0.12."""
    h_max = max_stable_step(CLASSICAL, t_end=100.0)
    assert 0.05 < h_max < 0.2
    # Consistent with the textbook estimate from the most damped mode alone.
    extent = spectral_extent(CLASSICAL, t_end=100.0)
    crude = RK4_REAL_AXIS_LIMIT / abs(extent["min_real"])
    assert h_max == pytest.approx(crude, rel=0.3)


def test_expanding_modes_do_not_veto_every_step() -> None:
    """Growing directions must not drive the stable step to zero.

    Requiring |R| <= 1 over the *whole* spectrum yields h = 0, because Lorenz
    genuinely has growing directions. Restricting to the damped modes is what
    makes the question well posed.
    """
    eigs = attractor_eigenvalues(CLASSICAL, t_end=50.0)
    assert np.any(np.abs(rk4_stability_function(1e-6 * eigs)) > 1.0)
    assert max_stable_step(CLASSICAL, t_end=50.0) > 0.01


def test_c_plus_real_part_crosses_zero_at_the_hopf_point() -> None:
    rho_h = hopf_rho(CLASSICAL)
    branch = c_plus_stability(np.linspace(5.0, 30.0, 60))
    finite = np.isfinite(branch.max_real)
    crossings = np.flatnonzero(np.diff(np.sign(branch.max_real[finite])))
    assert crossings.size == 1
    rho_at_crossing = branch.rho[finite][crossings[0]]
    assert rho_at_crossing == pytest.approx(rho_h, abs=0.6)
    assert branch.hopf == pytest.approx(rho_h)


def test_c_plus_undefined_below_the_pitchfork() -> None:
    branch = c_plus_stability(np.array([0.5, 0.9, 2.0]))
    assert np.isnan(branch.max_real[:2]).all()
    assert np.isfinite(branch.max_real[2])


@pytest.mark.slow
def test_bifurcation_sweep_collapses_onto_c_plus_below_the_hopf_point() -> None:
    """Below rho_H the maxima converge to z = rho - 1, the height of C+.

    Note that they do not become *fewer*: the approach to C+ is a damped
    spiral, so z keeps oscillating and keeps having local maxima. What changes
    at the Hopf point is that the maxima stop converging, not that they stop
    existing.
    """
    rhos = np.array([18.0, 20.0, 22.0])
    quiet = rho_sweep(rhos, t_end=40.0, transient=150.0)
    for rho in rhos:
        settled = quiet.z_max[quiet.rho == rho]
        assert settled.size > 0
        assert settled == pytest.approx(np.full(settled.size, rho - 1.0), abs=1e-3)


@pytest.mark.slow
def test_bifurcation_sweep_spreads_into_a_band_above_the_hopf_point() -> None:
    chaotic = rho_sweep(np.array([26.0, 28.0, 30.0]), t_end=40.0, transient=150.0)
    for rho in (26.0, 28.0, 30.0):
        band = chaotic.z_max[chaotic.rho == rho]
        assert band.std() > 1.0
        assert band.min() > rho - 1.0  # every maximum overshoots C+
