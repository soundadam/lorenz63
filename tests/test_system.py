"""Closed-form facts about the Lorenz vector field.

These are the numbers the report quotes in section 2. Every one of them is
checked here against an independent computation (finite differences, direct
substitution, an eigensolve) rather than taken on trust.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lorenz63.system import (
    LorenzParams,
    divergence,
    fixed_point_eigenvalues,
    fixed_points,
    hopf_rho,
    jacobian,
    rhs,
    trapping_function_derivative,
    volume_contraction,
)

CLASSICAL = LorenzParams()


def test_rejects_nonpositive_parameters() -> None:
    with pytest.raises(ValueError):
        LorenzParams(sigma=0.0)
    with pytest.raises(ValueError):
        LorenzParams(beta=-1.0)


def test_fixed_points_are_roots_of_the_vector_field() -> None:
    for fp in fixed_points(CLASSICAL):
        assert np.linalg.norm(rhs(0.0, fp, CLASSICAL)) < 1e-12


def test_c_plus_matches_closed_form() -> None:
    fps = fixed_points(CLASSICAL)
    assert fps.shape == (3, 3)
    s = math.sqrt(CLASSICAL.beta * (CLASSICAL.rho - 1.0))
    assert s == pytest.approx(math.sqrt(72.0))
    assert fps[1] == pytest.approx([s, s, 27.0])
    assert fps[2] == pytest.approx([-s, -s, 27.0])


def test_only_the_origin_below_the_pitchfork() -> None:
    assert fixed_points(LorenzParams(rho=0.5)).shape == (1, 3)
    assert fixed_points(LorenzParams(rho=1.0)).shape == (1, 3)
    assert fixed_points(LorenzParams(rho=1.5)).shape == (3, 3)


def test_jacobian_matches_finite_differences() -> None:
    rng = np.random.default_rng(20260922)
    eps = 1e-6
    for u in rng.uniform(-25.0, 25.0, size=(12, 3)):
        analytic = jacobian(u, CLASSICAL)
        numeric = np.empty((3, 3))
        for j in range(3):
            step = np.zeros(3)
            step[j] = eps
            numeric[:, j] = (
                rhs(0.0, u + step, CLASSICAL) - rhs(0.0, u - step, CLASSICAL)
            ) / (2.0 * eps)
        assert analytic == pytest.approx(numeric, abs=1e-6)


def test_trace_is_constant_everywhere() -> None:
    """The identity the whole Lyapunov calculation is later validated against."""
    rng = np.random.default_rng(7)
    expected = divergence(CLASSICAL)
    assert expected == pytest.approx(-(10.0 + 1.0 + 8.0 / 3.0))
    for u in rng.uniform(-60.0, 60.0, size=(40, 3)):
        assert np.trace(jacobian(u, CLASSICAL)) == pytest.approx(expected, rel=1e-14)


def test_volume_contraction_is_exponential() -> None:
    assert volume_contraction(CLASSICAL, 0.0) == pytest.approx(1.0)
    assert volume_contraction(CLASSICAL, 1.0) == pytest.approx(math.exp(-13.0 - 2.0 / 3.0))
    assert volume_contraction(CLASSICAL, 2.0) == pytest.approx(
        volume_contraction(CLASSICAL, 1.0) ** 2
    )


def test_hopf_threshold() -> None:
    assert hopf_rho(CLASSICAL) == pytest.approx(24.7368421, rel=1e-6)
    # sigma <= beta + 1: C+/C- never destabilise, no classical strange attractor.
    assert hopf_rho(LorenzParams(sigma=1.0, beta=8.0 / 3.0)) == math.inf


def test_c_plus_stability_switches_at_the_hopf_point() -> None:
    rho_h = hopf_rho(CLASSICAL)
    below = fixed_point_eigenvalues(LorenzParams(rho=rho_h - 1.0))[1]
    above = fixed_point_eigenvalues(LorenzParams(rho=rho_h + 1.0))[1]
    assert below.real.max() < 0.0
    assert above.real.max() > 0.0


def test_origin_is_a_saddle_at_the_classical_parameters() -> None:
    eigs = fixed_point_eigenvalues(CLASSICAL)[0]
    assert np.all(np.abs(eigs.imag) < 1e-12)
    assert (eigs.real > 0.0).sum() == 1
    assert (eigs.real < 0.0).sum() == 2


def test_trajectories_are_trapped_in_a_bounded_region() -> None:
    """dV/dt < 0 far from the origin, so nothing escapes to infinity."""
    rng = np.random.default_rng(11)
    directions = rng.normal(size=(30, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for direction in directions * 500.0:
        assert trapping_function_derivative(direction, CLASSICAL) < 0.0
