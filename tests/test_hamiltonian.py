"""The conservative control case: what a symplectic method does and does not buy."""

from __future__ import annotations

import numpy as np
import pytest

from lorenz63.hamiltonian import (
    KEPLER_PERIOD,
    RK4_STAGES,
    angular_momentum,
    error_growth_exponent,
    horizon,
    kepler_energy,
    kepler_exact,
    kepler_initial,
    kepler_rk4,
    kepler_verlet,
    position_error,
)


def test_exact_orbit_starts_at_the_initial_state_and_keeps_its_invariants() -> None:
    q0, p0 = kepler_initial()
    t = np.linspace(0.0, 3.0 * KEPLER_PERIOD, 301)
    q, p = kepler_exact(t)
    np.testing.assert_allclose(q[0], q0, atol=1e-14)
    np.testing.assert_allclose(p[0], p0, atol=1e-14)
    np.testing.assert_allclose(q[100], q0, atol=1e-12)  # one full period
    np.testing.assert_allclose(kepler_energy(q, p), kepler_energy(q0, p0), atol=1e-12)
    np.testing.assert_allclose(angular_momentum(q, p), angular_momentum(q0, p0), atol=1e-12)


def test_verlet_is_second_order() -> None:
    errors = [position_error(kepler_verlet(KEPLER_PERIOD, h))[-1] for h in (0.004, 0.002, 0.001)]
    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        assert coarse / fine == pytest.approx(4.0, rel=0.05)


def test_verlet_keeps_angular_momentum_to_roundoff() -> None:
    """Stormer-Verlet conserves quadratic invariants of a central force exactly."""
    traj = kepler_verlet(50 * KEPLER_PERIOD, 0.02)
    momentum = angular_momentum(traj.q, traj.p)
    assert np.ptp(momentum) < 1e-12


def test_verlet_energy_stays_bounded_while_rk4_drifts() -> None:
    """Same cost; compare the worst energy error in the first and second half."""
    t_end, h = 100 * KEPLER_PERIOD, 0.02

    def growth(traj) -> float:
        error = np.abs(kepler_energy(traj.q, traj.p) - kepler_energy(traj.q[0], traj.p[0]))
        half = traj.t.size // 2
        return error[half:].max() / error[:half].max()

    assert growth(kepler_verlet(t_end, h)) < 1.05
    assert growth(kepler_rk4(t_end, RK4_STAGES * h)) > 1.5


def test_symplectic_error_grows_linearly_and_rk4_faster() -> None:
    """The phase error of an integrable orbit: ~t for Verlet, towards t^2 for RK4."""
    window = (5 * KEPLER_PERIOD, 20 * KEPLER_PERIOD)
    h = 0.005
    verlet = error_growth_exponent(kepler_verlet(window[1], h), window)
    rk4 = error_growth_exponent(kepler_rk4(window[1], RK4_STAGES * h), window)
    assert verlet == pytest.approx(1.0, abs=0.05)
    assert rk4 > 1.5


def test_verlet_horizon_is_a_power_law_in_the_step() -> None:
    """Error ~ h^2 t, so halving h quadruples the time until the phase is lost."""
    t_end = 40 * KEPLER_PERIOD
    coarse = horizon(kepler_verlet(t_end, 0.02), 0.1)
    fine = horizon(kepler_verlet(t_end, 0.01), 0.1)
    assert fine / coarse == pytest.approx(4.0, rel=0.1)
