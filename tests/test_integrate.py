"""Integrator correctness: RK4's order, and the adaptive wrapper's contract."""

from __future__ import annotations

import numpy as np
import pytest

from lorenz63.integrate import (
    DEFAULT_TRANSIENT,
    lorenz_rk4,
    on_attractor,
    reference_trajectory,
    rk4_integrate,
    solve_adaptive,
)
from lorenz63.system import LorenzParams, trapping_function_derivative

CLASSICAL = LorenzParams()


def test_rk4_is_fourth_order() -> None:
    """Halving h must cut the error by ~16 on a problem with a known solution."""
    lam = -0.7

    def f(t: float, u: np.ndarray) -> np.ndarray:
        del t
        return lam * u

    t_end = 2.0
    exact = np.exp(lam * t_end)
    errors = []
    for h in (0.1, 0.05, 0.025):
        _, u = rk4_integrate(f, np.array([1.0]), t_end, h)
        errors.append(abs(u[-1, 0] - exact))

    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        assert coarse / fine == pytest.approx(16.0, rel=0.05)


def test_rk4_rejects_bad_steps() -> None:
    f = lambda t, u: u  # noqa: E731
    with pytest.raises(ValueError):
        rk4_integrate(f, np.array([1.0]), 1.0, 0.0)
    with pytest.raises(ValueError):
        rk4_integrate(f, np.array([1.0]), 0.001, 1.0)


def test_rk4_keeps_a_truly_constant_step() -> None:
    t, _ = rk4_integrate(lambda tt, uu: -uu, np.array([1.0]), 1.0, 0.01)
    steps = np.diff(t)
    assert steps.max() - steps.min() < 1e-12


def test_adaptive_meets_its_tolerance_against_a_tighter_run() -> None:
    """Over a short span, before chaos has had time to amplify anything."""
    u0 = on_attractor(CLASSICAL)
    truth = reference_trajectory(CLASSICAL, u0, 1.0, dt_out=0.01)
    loose = solve_adaptive(CLASSICAL, u0, 1.0, method="RK45", rtol=1e-8, atol=1e-11, dt_out=0.01)
    scale = np.abs(truth.u).max()
    assert np.linalg.norm(truth.u - loose.u, axis=1).max() < 1e-5 * scale


def test_tighter_tolerance_is_more_accurate_before_saturation() -> None:
    """Note the short horizon: it is load-bearing.

    Run the same comparison to t=30 instead and the ordering disappears -- all
    three tolerances have by then saturated at the attractor's diameter (~50)
    and their errors are in essentially random order. Tolerance only buys
    accuracy while the error is still small; that is the whole subject of the
    report, and it makes naive error comparisons over long windows meaningless.
    """
    u0 = on_attractor(CLASSICAL)
    truth = reference_trajectory(CLASSICAL, u0, 8.0, dt_out=0.01)
    errors = []
    for rtol in (1e-6, 1e-9, 1e-12):
        test = solve_adaptive(
            CLASSICAL, u0, 8.0, method="RK45", rtol=rtol, atol=rtol * 1e-3, dt_out=0.01
        )
        errors.append(np.linalg.norm(truth.u - test.u, axis=1).max())
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] < 1.0  # still far from saturation


def test_adaptive_exposes_its_internal_steps() -> None:
    traj = solve_adaptive(CLASSICAL, on_attractor(CLASSICAL), 10.0, method="RK45", rtol=1e-9)
    steps = traj.solver_steps
    assert steps is not None
    assert steps.size > 10
    assert np.all(steps > 0.0)
    # The controller does vary the step; it is not secretly running fixed-step.
    assert steps.max() / steps.min() > 1.5


def test_fixed_step_run_reports_no_solver_steps() -> None:
    traj = lorenz_rk4(CLASSICAL, on_attractor(CLASSICAL), 5.0, 0.005)
    assert traj.solver_steps is None
    assert traj.t.size == 1001


def test_burn_in_lands_inside_the_trapping_region() -> None:
    u = on_attractor(CLASSICAL)
    assert np.all(np.isfinite(u))
    # Inside the bounding ellipsoid, V is no longer decreasing.
    assert trapping_function_derivative(u, CLASSICAL) > 0.0


def test_zero_transient_is_a_passthrough() -> None:
    u0 = np.array([2.0, 3.0, 4.0])
    assert on_attractor(CLASSICAL, u0=u0, transient=0.0) == pytest.approx(u0)
    assert DEFAULT_TRANSIENT > 0.0
