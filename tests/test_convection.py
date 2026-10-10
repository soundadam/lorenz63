"""The three Lorenz variables read back as a convecting fluid layer.

The checks are physical balances computed from the rebuilt fields, not the
closed forms in ``convection.py`` restated.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar

from lorenz63.convection import (
    convection_field,
    critical_rayleigh,
    nusselt,
    wavenumber_sq,
)
from lorenz63.system import LorenzParams, fixed_points

CLASSICAL = LorenzParams()


def _layer_mean(values: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    area = (x[0, -1] - x[0, 0]) * (z[-1, 0] - z[0, 0])
    return float(trapezoid(trapezoid(values, x[0], axis=1), z[:, 0]) / area)


def test_classical_beta_is_the_roll_that_goes_unstable_first() -> None:
    best = minimize_scalar(critical_rayleigh, bounds=(0.05, 5.0), method="bounded")
    assert 4.0 / (1.0 + best.x) == pytest.approx(CLASSICAL.beta, rel=1e-5)
    assert wavenumber_sq(CLASSICAL) == pytest.approx(best.x, rel=1e-5)


@pytest.mark.parametrize("rho", [2.0, 10.0, 28.0])
def test_plate_heat_flux_is_the_nusselt_number(rho: float) -> None:
    p = LorenzParams(rho=rho)
    state = fixed_points(p)[1]
    f = convection_field(state, p, nz=2001)
    width = f.x[0, -1] - f.x[0, 0]
    profile = trapezoid(f.temperature, f.x[0], axis=1) / width
    dz = f.z[1, 0] - f.z[0, 0]
    gradient_at_plate = -(-3.0 * profile[0] + 4.0 * profile[1] - profile[2]) / (2.0 * dz)
    assert gradient_at_plate == pytest.approx(nusselt(state, p), rel=1e-3)


@pytest.mark.parametrize("rho", [2.0, 10.0, 28.0])
def test_steady_rolls_carry_exactly_the_extra_heat(rho: float) -> None:
    # Layer-averaged energy balance for steady convection: conduction carries 1,
    # rising warm / sinking cool fluid carries <w T>, together the plate flux Nu.
    p = LorenzParams(rho=rho)
    state = fixed_points(p)[1]
    f = convection_field(state, p)
    convective = _layer_mean(f.w * f.temperature, f.x, f.z)
    assert convective > 0.0
    assert 1.0 + convective == pytest.approx(nusselt(state, p), rel=1e-6)
