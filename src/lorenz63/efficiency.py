"""How much work each explicit RK pair spends for a given accuracy.

Purpose in the report: the order of a method pays off only when accuracy sets
the step. A p-th order step has local error ~ h^(p+1), so reaching a tolerance
needs h ~ tol^(1/p) and costs s/h ~ s * tol^(-1/p) evaluations of f, with s the
evaluations per step. On log-log axes that is a line of slope -1/p: higher order
is flatter but starts higher, so it wins once the tolerance is tight enough.

When stability sets the step instead (a stiff problem), h ~ limit/|lambda| and
the work ~ s/limit no longer depends on tol or on the order. The stiff test
problem here is y' = -lam*(y - cos t): the solution follows cos t slowly, the
deviation from it decays at rate lam.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from lorenz63.integrate import REFERENCE_ATOL, REFERENCE_METHOD, REFERENCE_RTOL, on_attractor
from lorenz63.system import LorenzParams, rhs

#: Short enough that the e^(lambda_1 t) amplification of the global error stays
#: O(1), so the measured error is the method's, not chaos's.
WORK_T_END = 2.0

#: Stiffness of the test problem and its span: lam * t_end decay times.
STIFF_LAMBDA = 1000.0
STIFF_T_END = 10.0


@dataclass(frozen=True, slots=True)
class WorkPrecision:
    """Evaluations of f against the global error they bought, one row per rtol."""

    rtol: NDArray[np.floating]
    nfev: NDArray[np.integer]
    error: NDArray[np.floating]

    def slope(self) -> float:
        """d log(nfev) / d log(error); about -1/p for a p-th order method."""
        return float(np.polyfit(np.log(self.error), np.log(self.nfev), 1)[0])


def work_precision(
    p: LorenzParams, method: str, rtols: NDArray[np.floating], *, t_end: float = WORK_T_END
) -> WorkPrecision:
    """Relative error at ``t_end`` against the reference run, and the nfev it took."""
    u0 = on_attractor(p)
    ref = solve_ivp(
        rhs, (0.0, t_end), u0, args=(p,), method=REFERENCE_METHOD,
        rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL,
    ).y[:, -1]
    nfev, error = [], []
    for rtol in rtols:
        sol = solve_ivp(
            rhs, (0.0, t_end), u0, args=(p,), method=method, rtol=rtol, atol=REFERENCE_ATOL
        )
        nfev.append(sol.nfev)
        error.append(np.linalg.norm(sol.y[:, -1] - ref) / np.linalg.norm(ref))
    return WorkPrecision(np.asarray(rtols, float), np.array(nfev), np.array(error))


def _stiff_rhs(t: float, y: NDArray[np.floating], lam: float) -> NDArray[np.floating]:
    return -lam * (y - np.cos(t))


@dataclass(frozen=True, slots=True)
class StiffRun:
    rtol: NDArray[np.floating]
    nfev: NDArray[np.integer]
    #: Accepted steps only; nfev also pays for the rejected ones.
    steps: NDArray[np.integer]


def stiff_work(
    method: str,
    rtols: NDArray[np.floating],
    *,
    lam: float = STIFF_LAMBDA,
    t_end: float = STIFF_T_END,
) -> StiffRun:
    """Work the controller spends on y' = -lam*(y - cos t) at each rtol."""
    nfev, steps = [], []
    for rtol in rtols:
        sol = solve_ivp(
            _stiff_rhs, (0.0, t_end), [0.0], args=(lam,), method=method,
            rtol=rtol, atol=REFERENCE_ATOL,
        )
        nfev.append(sol.nfev)
        steps.append(sol.t.size - 1)
    return StiffRun(np.asarray(rtols, float), np.array(nfev), np.array(steps))
