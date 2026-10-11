"""Where the three Lorenz variables come from: Rayleigh–Bénard convection.

A fluid layer of depth H is heated from below with temperature difference
dT, between two stress-free plates. In the Boussinesq approximation a 2-D flow
is described by a stream function psi(x, z) and the temperature deviation
theta(x, z) from the linear conduction profile. Lorenz (1963), following
Saltzman (1962), keeps exactly three Fourier modes. In units H = kappa = dT = 1
(kappa the thermal diffusivity):

    psi   = sqrt(2) (1 + a^2) / a * X * sin(pi a x) sin(pi z)
    theta = [sqrt(2) Y cos(pi a x) sin(pi z) - Z sin(2 pi z)] / (pi r)

with ``a`` the horizontal wavenumber of the roll in units of pi/H and
``r = rho = Ra / Ra_c``. Substituting into the Boussinesq equations and
projecting onto the three modes gives the Lorenz system in the time
``tau = pi^2 (1 + a^2) t`` with

    sigma = nu / kappa             (Prandtl number)
    beta  = 4 / (1 + a^2)          (roll geometry)

So each variable is the amplitude of one spatial pattern:

    X  how fast the rolls turn (sign = sense of rotation)
    Y  temperature difference between rising and sinking fluid
    Z  how far the mean temperature profile has bent away from linear,
       i.e. how much heat convection carries: Nu = 1 + 2 Z / r

Nothing here integrates the Lorenz system; these are the closed-form maps from
a state (X, Y, Z) back to the fields it stands for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lorenz63.system import LorenzParams


def wavenumber_sq(p: LorenzParams) -> float:
    """``a^2`` implied by ``beta = 4 / (1 + a^2)``. Needs ``beta < 4``."""
    if p.beta >= 4.0:
        raise ValueError("beta >= 4 corresponds to no real roll wavenumber")
    return 4.0 / p.beta - 1.0


def critical_rayleigh(a_sq: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
    """Onset Rayleigh number of a roll with wavenumber ``a`` (stress-free plates).

        Ra_c(a) = pi^4 (1 + a^2)^3 / a^2

    Its minimum over ``a`` is the roll that goes unstable first; Lorenz took
    ``beta`` at that roll.
    """
    return math.pi**4 * (1.0 + a_sq) ** 3 / a_sq


def nusselt(u: NDArray[np.floating], p: LorenzParams) -> float:
    """Heat flux through the plates relative to pure conduction: ``1 + 2 Z / rho``."""
    return 1.0 + 2.0 * float(u[2]) / p.rho


@dataclass(frozen=True, slots=True)
class ConvectionField:
    """Fields on a grid covering one wavelength (two counter-rotating rolls)."""

    x: NDArray[np.floating]  # (nz, nx)
    z: NDArray[np.floating]
    psi: NDArray[np.floating]
    u: NDArray[np.floating]  # horizontal velocity, -d psi / dz
    w: NDArray[np.floating]  # vertical velocity, d psi / dx
    temperature: NDArray[np.floating]  # total: conduction profile + theta


def convection_field(
    state: NDArray[np.floating],
    p: LorenzParams,
    nx: int = 241,
    nz: int = 121,
) -> ConvectionField:
    """Rebuild the flow and temperature that the state ``(X, Y, Z)`` stands for."""
    a_sq = wavenumber_sq(p)
    a = math.sqrt(a_sq)
    big_x, big_y, big_z = (float(v) for v in state)

    x, z = np.meshgrid(np.linspace(0.0, 2.0 / a, nx), np.linspace(0.0, 1.0, nz))
    kx, kz = math.pi * a * x, math.pi * z
    amp = math.sqrt(2.0) * (1.0 + a_sq) / a * big_x

    psi = amp * np.sin(kx) * np.sin(kz)
    u = -amp * math.pi * np.sin(kx) * np.cos(kz)
    w = amp * math.pi * a * np.cos(kx) * np.sin(kz)
    theta = (
        math.sqrt(2.0) * big_y * np.cos(kx) * np.sin(kz) - big_z * np.sin(2.0 * kz)
    ) / (math.pi * p.rho)
    return ConvectionField(x=x, z=z, psi=psi, u=u, w=w, temperature=1.0 - z + theta)
