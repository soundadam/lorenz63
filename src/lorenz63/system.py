"""Lorenz (1963) vector field, Jacobian, fixed points and bifurcation thresholds.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

Everything here is closed-form. No integration, no plotting — so the report's
qualitative claims (dissipation rate, fixed-point locations, Hopf threshold)
rest on algebra that the tests check against finite differences, not on a
numerical experiment that could be quietly wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Homoclinic explosion for the classical (sigma=10, beta=8/3) section of
# parameter space. Unlike rho=1 (pitchfork) and rho_hopf (Hopf) this one has no
# closed form; it is a numerically located value quoted from the literature
# (Sparrow 1982, "The Lorenz Equations", ch. 5) and is used only to annotate
# the bifurcation diagram.
RHO_HOMOCLINIC = 13.926


@dataclass(frozen=True, slots=True)
class LorenzParams:
    """The three Lorenz parameters. Defaults are Lorenz's own 1963 values."""

    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0

    def __post_init__(self) -> None:
        if self.sigma <= 0.0 or self.beta <= 0.0:
            raise ValueError("sigma and beta must be positive")


def rhs(t: float, u: NDArray[np.floating], p: LorenzParams) -> NDArray[np.floating]:
    """Right-hand side. ``t`` is unused (autonomous) but kept for solve_ivp."""
    del t
    x, y, z = u
    return np.array(
        [
            p.sigma * (y - x),
            x * (p.rho - z) - y,
            x * y - p.beta * z,
        ]
    )


def jacobian(u: NDArray[np.floating], p: LorenzParams) -> NDArray[np.floating]:
    """Jacobian df/du at the state ``u``.

    Its trace is ``-(sigma + 1 + beta)`` at *every* point — a constant. That is
    the whole of section 2 of the report in one line: phase-space volume
    contracts at a uniform exponential rate, so there is no conserved quantity
    and no symplectic structure to preserve.
    """
    x, y, z = u
    return np.array(
        [
            [-p.sigma, p.sigma, 0.0],
            [p.rho - z, -1.0, -x],
            [y, x, -p.beta],
        ]
    )


def divergence(p: LorenzParams) -> float:
    """Phase-space divergence ``tr J = -(sigma + 1 + beta)``; state-independent."""
    return -(p.sigma + 1.0 + p.beta)


def volume_contraction(p: LorenzParams, t: float) -> float:
    """Factor by which any phase-space volume shrinks after time ``t``."""
    return math.exp(divergence(p) * t)


def fixed_points(p: LorenzParams) -> NDArray[np.floating]:
    """Equilibria: the origin always, plus C+/C- once ``rho > 1``.

    Returns an array of shape (1, 3) or (3, 3), ordered ``[origin, C+, C-]``.
    """
    origin = np.zeros(3)
    if p.rho <= 1.0:
        return origin.reshape(1, 3)
    s = math.sqrt(p.beta * (p.rho - 1.0))
    return np.array(
        [
            origin,
            [s, s, p.rho - 1.0],
            [-s, -s, p.rho - 1.0],
        ]
    )


def fixed_point_eigenvalues(p: LorenzParams) -> list[NDArray[np.complexfloating]]:
    """Eigenvalues of J at each fixed point, in the order ``fixed_points`` returns."""
    return [np.linalg.eigvals(jacobian(fp, p)) for fp in fixed_points(p)]


def hopf_rho(p: LorenzParams) -> float:
    """rho at which C+/C- lose stability through a subcritical Hopf bifurcation.

        rho_H = sigma * (sigma + beta + 3) / (sigma - beta - 1)

    Only meaningful when ``sigma > beta + 1``; otherwise C+/C- never destabilise
    and the classical strange attractor does not appear. Returns ``inf`` there.
    """
    denominator = p.sigma - p.beta - 1.0
    if denominator <= 0.0:
        return math.inf
    return p.sigma * (p.sigma + p.beta + 3.0) / denominator


def trapping_function(u: NDArray[np.floating], p: LorenzParams) -> float:
    """Lyapunov function ``V = rho*x^2 + sigma*y^2 + sigma*(z - 2*rho)^2``.

    Differentiating along the flow gives exactly

        dV/dt = -2*sigma * (rho*x^2 + y^2 + beta*(z - rho)^2 - beta*rho^2)

    which is negative everywhere outside the ellipsoid
    ``rho*x^2 + y^2 + beta*(z - rho)^2 = beta*rho^2``. So every trajectory is
    eventually trapped in a bounded region: solutions never escape to infinity,
    however chaotic they look. See :func:`trapping_function_derivative`.
    """
    x, y, z = u
    return p.rho * x**2 + p.sigma * y**2 + p.sigma * (z - 2.0 * p.rho) ** 2


def trapping_function_derivative(u: NDArray[np.floating], p: LorenzParams) -> float:
    """``dV/dt`` along the flow. Negative outside the bounding ellipsoid."""
    x, y, z = u
    return -2.0 * p.sigma * (
        p.rho * x**2 + y**2 + p.beta * (z - p.rho) ** 2 - p.beta * p.rho**2
    )


def summary(p: LorenzParams) -> dict[str, float]:
    """Closed-form quantities the report quotes. Used by ``lorenz63 info``."""
    fps = fixed_points(p)
    return {
        "sigma": p.sigma,
        "rho": p.rho,
        "beta": p.beta,
        "divergence": divergence(p),
        "n_fixed_points": float(len(fps)),
        "c_plus_x": float(fps[1, 0]) if len(fps) == 3 else math.nan,
        "c_plus_z": float(fps[1, 2]) if len(fps) == 3 else math.nan,
        "rho_pitchfork": 1.0,
        "rho_homoclinic": RHO_HOMOCLINIC,
        "rho_hopf": hopf_rho(p),
    }
