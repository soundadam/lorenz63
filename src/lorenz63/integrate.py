"""Integrators: a hand-rolled fixed-step RK4 and thin wrappers over ``solve_ivp``.

Two things this module deliberately exposes that a bare ``solve_ivp`` call does
not:

* the adaptive solver's *internal* step sizes ``h(t)``, so the report can show
  where the controller chooses to work harder (figure 06);
* a ``transient`` burn-in run at reference accuracy, so every statistic in the
  report is computed on the attractor rather than on the approach to it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from lorenz63.system import LorenzParams, rhs

#: Tightest tolerance we treat as "truth". scipy warns below ~100*eps, so this
#: is about as far as double precision usefully goes with DOP853.
REFERENCE_RTOL = 1e-13
REFERENCE_ATOL = 1e-15
REFERENCE_METHOD = "DOP853"

#: Fixed starting point for every run in the report, so figures are comparable.
DEFAULT_U0 = np.array([1.0, 1.0, 1.0])

#: Time discarded before any statistic is collected (well past the transient).
DEFAULT_TRANSIENT = 50.0


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A sampled solution plus the provenance needed to reproduce it."""

    t: NDArray[np.floating]
    u: NDArray[np.floating]  # shape (n, 3)
    method: str
    rtol: float
    atol: float
    #: Times of the solver's own internal steps (adaptive runs only).
    solver_t: NDArray[np.floating] | None = field(default=None)

    @property
    def x(self) -> NDArray[np.floating]:
        return self.u[:, 0]

    @property
    def y(self) -> NDArray[np.floating]:
        return self.u[:, 1]

    @property
    def z(self) -> NDArray[np.floating]:
        return self.u[:, 2]

    @property
    def solver_steps(self) -> NDArray[np.floating] | None:
        """Internal step sizes ``h_n``; ``None`` for fixed-step runs."""
        if self.solver_t is None:
            return None
        return np.diff(self.solver_t)

    @property
    def label(self) -> str:
        return f"{self.method} rtol={self.rtol:.0e}"


# ---------------------------------------------------------------------------
# Fixed-step RK4 (generic, so the order-of-convergence test can use a problem
# with a known closed-form solution rather than Lorenz)
# ---------------------------------------------------------------------------


def rk4_step(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    t: float,
    u: NDArray[np.floating],
    h: float,
) -> NDArray[np.floating]:
    """One classical 4-stage, 4th-order Runge-Kutta step."""
    k1 = f(t, u)
    k2 = f(t + 0.5 * h, u + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, u + 0.5 * h * k2)
    k4 = f(t + h, u + h * k3)
    return u + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_integrate(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    u0: NDArray[np.floating],
    t_end: float,
    h: float,
    *,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Integrate ``u' = f(t, u)`` with a fixed step. Returns ``(t, u)``.

    The final step is *not* trimmed to land exactly on ``t_end``; instead the
    number of steps is rounded so that the step stays exactly ``h`` throughout.
    Keeping ``h`` constant matters for any backward-error argument, and it is
    also what makes the 4th-order convergence test clean.
    """
    if h <= 0.0:
        raise ValueError("h must be positive")
    n_steps = int(round((t_end - t0) / h))
    if n_steps < 1:
        raise ValueError("t_end - t0 must span at least one step")

    u0 = np.asarray(u0, dtype=float)
    t = t0 + h * np.arange(n_steps + 1)
    u = np.empty((n_steps + 1, u0.size))
    u[0] = u0
    for n in range(n_steps):
        u[n + 1] = rk4_step(f, t[n], u[n], h)
    return t, u


def lorenz_rk4(
    p: LorenzParams,
    u0: NDArray[np.floating],
    t_end: float,
    h: float,
) -> Trajectory:
    """Fixed-step RK4 applied to Lorenz."""
    t, u = rk4_integrate(lambda tt, uu: rhs(tt, uu, p), u0, t_end, h)
    return Trajectory(t=t, u=u, method=f"RK4 h={h:g}", rtol=float("nan"), atol=float("nan"))


# ---------------------------------------------------------------------------
# Adaptive
# ---------------------------------------------------------------------------


def solve_adaptive(
    p: LorenzParams,
    u0: NDArray[np.floating],
    t_end: float,
    *,
    method: str = REFERENCE_METHOD,
    rtol: float = REFERENCE_RTOL,
    atol: float = REFERENCE_ATOL,
    dt_out: float = 0.005,
    t0: float = 0.0,
) -> Trajectory:
    """Adaptive integration, sampled onto a uniform output grid.

    The solver runs with ``dense_output=True`` and *no* ``t_eval``, so its own
    step selection is untouched; the uniform grid is produced afterwards by
    interpolating the dense output. That is what lets us report both a smooth
    curve and the true ``h(t)`` the controller chose.
    """
    u0 = np.asarray(u0, dtype=float)
    sol = solve_ivp(
        rhs,
        (t0, t_end),
        u0,
        args=(p,),
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    n_out = int(round((t_end - t0) / dt_out)) + 1
    t = np.linspace(t0, t_end, n_out)
    u = sol.sol(t).T
    return Trajectory(t=t, u=u, method=method, rtol=rtol, atol=atol, solver_t=sol.t)


def on_attractor(
    p: LorenzParams,
    *,
    u0: NDArray[np.floating] | None = None,
    transient: float = DEFAULT_TRANSIENT,
) -> NDArray[np.floating]:
    """Burn in at reference accuracy and return a point on the attractor.

    Every statistic in the report starts from the output of this function, so
    that "the attractor" means the same set of states in every figure.
    """
    if u0 is None:
        u0 = DEFAULT_U0
    if transient <= 0.0:
        return np.asarray(u0, dtype=float)
    sol = solve_ivp(
        rhs,
        (0.0, transient),
        np.asarray(u0, dtype=float),
        args=(p,),
        method=REFERENCE_METHOD,
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"transient burn-in failed: {sol.message}")
    return sol.y[:, -1]


def reference_trajectory(
    p: LorenzParams,
    u0: NDArray[np.floating],
    t_end: float,
    *,
    dt_out: float = 0.005,
) -> Trajectory:
    """DOP853 at the tightest tolerance we trust. Used as the yardstick."""
    return solve_adaptive(
        p,
        u0,
        t_end,
        method=REFERENCE_METHOD,
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
        dt_out=dt_out,
    )
