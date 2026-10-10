"""The conservative control case: the Kepler problem, RK4 against Stormer-Verlet.

Purpose in the report: the first row of the table in section 7 claims that on a
Hamiltonian system a long single trajectory *is* trustworthy, and that the check
is "does the energy stay bounded". Lorenz cannot test that claim -- it has no
energy -- so this module supplies a system that can.

The Kepler problem is the standard choice (Hairer, Lubich & Wanner, ch. I and X):
it has a closed-form solution to measure the global error against, and three
invariants (energy, angular momentum, the orbit's shape) to watch.

Units: semi-major axis 1, GM = 1, so the period is 2*pi and the energy is -1/2
for every eccentricity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lorenz63.integrate import rk4_integrate

KEPLER_PERIOD = 2.0 * np.pi

#: Hairer-Lubich-Wanner's eccentricity: eccentric enough that the perihelion
#: passage dominates the error, not so eccentric that tiny steps are needed.
DEFAULT_ECCENTRICITY = 0.6

#: Classical RK4 evaluates the force four times per step, Verlet once. Comparing
#: them at ``h_rk4 = RK4_STAGES * h_verlet`` gives both the same cost.
RK4_STAGES = 4


@dataclass(frozen=True, slots=True)
class PhaseTrajectory:
    """Positions and momenta of a planar orbit on a uniform time grid."""

    t: NDArray[np.floating]
    q: NDArray[np.floating]  # shape (n, 2)
    p: NDArray[np.floating]  # shape (n, 2)
    method: str


def kepler_initial(eccentricity: float = DEFAULT_ECCENTRICITY) -> tuple[NDArray, NDArray]:
    """Start at perihelion, moving in +y."""
    e = eccentricity
    q0 = np.array([1.0 - e, 0.0])
    p0 = np.array([0.0, np.sqrt((1.0 + e) / (1.0 - e))])
    return q0, p0


def kepler_force(q: NDArray[np.floating]) -> NDArray[np.floating]:
    """``-dV/dq`` for ``V = -1/|q|``."""
    r = np.linalg.norm(q, axis=-1, keepdims=True)
    return -q / r**3


def kepler_energy(q: NDArray[np.floating], p: NDArray[np.floating]) -> NDArray[np.floating]:
    return 0.5 * np.sum(p**2, axis=-1) - 1.0 / np.linalg.norm(q, axis=-1)


def angular_momentum(q: NDArray[np.floating], p: NDArray[np.floating]) -> NDArray[np.floating]:
    return q[..., 0] * p[..., 1] - q[..., 1] * p[..., 0]


def kepler_exact(
    t: NDArray[np.floating], eccentricity: float = DEFAULT_ECCENTRICITY
) -> tuple[NDArray, NDArray]:
    """Closed-form ``(q, p)`` at times ``t`` for the orbit of :func:`kepler_initial`.

    Solves Kepler's equation ``M = E - e sin E`` for the eccentric anomaly by
    Newton's method; with mean motion 1 the mean anomaly is simply ``t``.
    """
    e = eccentricity
    mean = np.asarray(t, dtype=float)
    ecc = mean + e * np.sin(mean)
    for _ in range(50):
        delta = (ecc - e * np.sin(ecc) - mean) / (1.0 - e * np.cos(ecc))
        ecc = ecc - delta
        if np.max(np.abs(delta), initial=0.0) < 1e-15:
            break
    b = np.sqrt(1.0 - e * e)
    rate = 1.0 / (1.0 - e * np.cos(ecc))  # dE/dt
    q = np.stack([np.cos(ecc) - e, b * np.sin(ecc)], axis=-1)
    p = np.stack([-np.sin(ecc) * rate, b * np.cos(ecc) * rate], axis=-1)
    return q, p


def stormer_verlet(
    force: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    q0: NDArray[np.floating],
    p0: NDArray[np.floating],
    t_end: float,
    h: float,
) -> PhaseTrajectory:
    """Kick-drift-kick leapfrog for ``H = |p|^2/2 + V(q)``. Second order, symplectic.

    One force evaluation per step: the force at the end of a step is reused as
    the first kick of the next.
    """
    if h <= 0.0:
        raise ValueError("h must be positive")
    n_steps = int(round(t_end / h))
    if n_steps < 1:
        raise ValueError("t_end must span at least one step")

    q = np.empty((n_steps + 1, np.size(q0)))
    p = np.empty_like(q)
    q[0], p[0] = q0, p0
    f = force(q[0])
    for n in range(n_steps):
        p_half = p[n] + 0.5 * h * f
        q[n + 1] = q[n] + h * p_half
        f = force(q[n + 1])
        p[n + 1] = p_half + 0.5 * h * f
    return PhaseTrajectory(t=h * np.arange(n_steps + 1), q=q, p=p, method=f"Verlet h={h:g}")


def kepler_verlet(
    t_end: float, h: float, eccentricity: float = DEFAULT_ECCENTRICITY
) -> PhaseTrajectory:
    q0, p0 = kepler_initial(eccentricity)
    return stormer_verlet(kepler_force, q0, p0, t_end, h)


def kepler_rk4(
    t_end: float, h: float, eccentricity: float = DEFAULT_ECCENTRICITY
) -> PhaseTrajectory:
    q0, p0 = kepler_initial(eccentricity)

    def f(t: float, u: NDArray[np.floating]) -> NDArray[np.floating]:
        del t
        return np.concatenate([u[2:], kepler_force(u[:2])])

    t, u = rk4_integrate(f, np.concatenate([q0, p0]), t_end, h)
    return PhaseTrajectory(t=t, q=u[:, :2], p=u[:, 2:], method=f"RK4 h={h:g}")


def position_error(
    traj: PhaseTrajectory, eccentricity: float = DEFAULT_ECCENTRICITY
) -> NDArray[np.floating]:
    """``|q_numerical(t) - q_exact(t)|`` along the trajectory."""
    q_exact, _ = kepler_exact(traj.t, eccentricity)
    return np.linalg.norm(traj.q - q_exact, axis=1)


def running_max(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Envelope of an oscillating error: the worst value seen so far."""
    return np.maximum.accumulate(values)


def error_growth_exponent(
    traj: PhaseTrajectory,
    t_window: tuple[float, float],
    eccentricity: float = DEFAULT_ECCENTRICITY,
) -> float:
    """Log-log slope of the position-error envelope over ``t_window``.

    The envelope is sampled once per period so the perihelion spikes inside an
    orbit do not bias the fit. Symplectic methods give 1 on this integrable
    problem; non-symplectic ones drift towards 2 as the energy error feeds a
    frequency error (Hairer-Lubich-Wanner, Thm X.3.1).
    """
    envelope = running_max(position_error(traj, eccentricity))
    marks = np.arange(t_window[0], t_window[1] + 1e-9, KEPLER_PERIOD)
    idx = np.searchsorted(traj.t, marks)
    idx = idx[idx < traj.t.size]
    slope, _ = np.polyfit(np.log(traj.t[idx]), np.log(envelope[idx]), 1)
    return float(slope)


def horizon(
    traj: PhaseTrajectory, threshold: float, eccentricity: float = DEFAULT_ECCENTRICITY
) -> float:
    """First time the position error exceeds ``threshold``; ``inf`` if it never does."""
    above = np.flatnonzero(position_error(traj, eccentricity) > threshold)
    return float(traj.t[above[0]]) if above.size else float("inf")
