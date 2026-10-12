"""Linear stability of explicit RK4, and where Lorenz's Jacobian eigenvalues sit.

Purpose in the report: to separate two things that are easy to conflate.

* *Numerical stability* is a property of method + step size: h*lambda_J must lie
  inside the stability region or the discrete map blows up.
* *Lyapunov exponents* are a property of the flow. The step size cannot change
  them; the true solution separates exponentially whatever you do.

On the Lorenz attractor the most strongly damped Jacobian eigenvalue has real
part about -21.9, so RK4 needs h < 2.785/21.9 ~ 0.13 to stay stable. The
fixed-step runs use h = 0.01, and DOP853 at the report's tolerances steps well
inside its own, wider region. So stability is never the binding constraint
here -- accuracy is. Saying "I must limit h to stop it blowing up"
gets the right action for the wrong reason.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.rk import DOP853, RK45
from scipy.optimize import brentq

from lorenz63.integrate import (
    REFERENCE_ATOL,
    on_attractor,
    reference_trajectory,
    solve_adaptive,
)
from lorenz63.system import LorenzParams, jacobian, rhs

#: |R(z)| = 1 crossings for classical RK4 along each axis.
RK4_REAL_AXIS_LIMIT = 2.7852935634
RK4_IMAG_AXIS_LIMIT = 2.0 * np.sqrt(2.0)


def rk4_stability_function(z: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """R(z) for classical RK4 — the 4th-order Taylor truncation of exp(z)."""
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def rk4_stability_mask(
    re: NDArray[np.floating], im: NDArray[np.floating]
) -> NDArray[np.bool_]:
    """Boolean grid of ``|R(z)| <= 1`` over the complex plane."""
    z = re[None, :] + 1j * im[:, None]
    return np.abs(rk4_stability_function(z)) <= 1.0


def attractor_eigenvalues(
    p: LorenzParams,
    *,
    t_end: float = 200.0,
    dt_out: float = 0.02,
) -> NDArray[np.complexfloating]:
    """Jacobian eigenvalues sampled along a trajectory on the attractor."""
    start = on_attractor(p)
    traj = reference_trajectory(p, start, t_end, dt_out=dt_out)
    return np.array([np.linalg.eigvals(jacobian(u, p)) for u in traj.u]).ravel()


def max_stable_step(p: LorenzParams, **kwargs: float) -> float:
    """Largest h keeping the *damped* modes inside RK4's stability region.

    The qualifier matters, and it is the point of this function. About a third
    of the Jacobian eigenvalues sampled on the attractor have positive real
    part: the flow genuinely has expanding directions, and ``|R(h*lambda)| > 1``
    there is the scheme correctly reproducing growth, not an instability. If we
    demanded ``|R| <= 1`` for the whole spectrum, no positive h would qualify
    and the answer would be a meaningless zero.

    What actually limits an explicit method is the strongly *damped* end of the
    spectrum -- the modes the exact flow kills fast and that RK4 will amplify
    instead if h*lambda leaves the region. So the search is restricted to
    eigenvalues with ``Re lambda < 0``.
    """
    eigs = attractor_eigenvalues(p, **kwargs)  # type: ignore[arg-type]
    damped = eigs[eigs.real < 0.0]
    if damped.size == 0:
        return float("inf")
    # Bisect on the scale factor; the region is star-shaped about the origin
    # along each ray, so the search is well posed.
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.all(np.abs(rk4_stability_function(mid * damped)) <= 1.0):
            lo = mid
        else:
            hi = mid
    return lo


def spectral_extent(p: LorenzParams, **kwargs: float) -> dict[str, float]:
    """Summary of where the Jacobian spectrum sits on the attractor."""
    eigs = attractor_eigenvalues(p, **kwargs)  # type: ignore[arg-type]
    return {
        "max_abs": float(np.abs(eigs).max()),
        "min_real": float(eigs.real.min()),
        "max_real": float(eigs.real.max()),
        "fraction_expanding": float((eigs.real > 0.0).mean()),
    }


#: Butcher tableaux (A, b) of the stage that each method propagates. Classical
#: RK4 is written out; RK45 (Dormand-Prince 5) and DOP853 are read from scipy, so
#: the polynomials below are those of the solvers the report actually runs.
_TABLEAUX: dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]] = {
    "RK4": (
        np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]], float),
        np.array([1, 2, 2, 1], float) / 6.0,
    ),
    "RK45": (np.asarray(RK45.A, float), np.asarray(RK45.B, float)),
    "DOP853": (
        np.asarray(DOP853.A, float)[: DOP853.n_stages, : DOP853.n_stages],
        np.asarray(DOP853.B, float),
    ),
}


def stability_polynomial(method: str) -> NDArray[np.floating]:
    """Coefficients c_k of R(z) = sum_k c_k z^k, lowest power first.

    For an explicit method A is strictly lower triangular, so
    (I - zA)^{-1} = sum_k z^k A^k terminates and
    R(z) = 1 + z b^T (I - zA)^{-1} 1 = 1 + sum_k (b^T A^k 1) z^{k+1}.
    """
    a, b = _TABLEAUX[method]
    s = b.size
    a_sq = np.zeros((s, s))
    a_sq[: a.shape[0], : a.shape[1]] = a
    coeffs = [1.0]
    v = np.ones(s)
    for _ in range(s):
        coeffs.append(float(b @ v))
        v = a_sq @ v
    return np.array(coeffs)


def stability_function(method: str, z: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """R(z) of ``method`` ("RK4", "RK45" or "DOP853") evaluated at ``z``."""
    return np.polynomial.polynomial.polyval(z, stability_polynomial(method))


def real_axis_limit(method: str) -> float:
    """Distance from 0 to where the region's boundary crosses the negative real axis."""
    # Bracket the first crossing on a grid that starts just off the origin,
    # where |R| = 1 trivially, then refine it.
    xs = np.linspace(1e-3, 20.0, 20000)
    i = int(np.argmax(np.abs(stability_function(method, -xs)) > 1.0))
    return float(brentq(lambda x: abs(stability_function(method, -x)) - 1.0, xs[i - 1], xs[i]))


#: scipy classes of the embedded pairs, for their error-estimate weights.
_EMBEDDED = {"RK45": RK45, "DOP853": DOP853}


def _stage_values(method: str, z: NDArray[np.complexfloating]) -> list[NDArray[np.complexfloating]]:
    """Stage values y_i / y_n on y' = lambda*y, plus y_{n+1} / y_n as the last entry."""
    a, b = _TABLEAUX[method]
    z = np.asarray(z, complex)
    ys = [np.ones_like(z)]
    for i in range(1, b.size):
        ys.append(1.0 + z * sum(a[i, j] * ys[j] for j in range(i)))
    ys.append(1.0 + z * sum(b[j] * ys[j] for j in range(b.size)))
    return ys


def error_function(method: str, z: NDArray[np.complexfloating]) -> NDArray[np.floating]:
    """|E(z)|: the error norm scipy's controller computes on y' = lambda*y, per unit |y|.

    The step is accepted when the norm, measured against ``atol + rtol*|y|``, is
    at most 1. With atol negligible that is ``|E(z)| <= rtol``, the accuracy
    region. The stage after y_{n+1} is f at the new point (scipy's FSAL row),
    which on the test equation is lambda * R(z) * y_n. DOP853 blends a 5th- and
    a 3rd-order difference exactly as ``DOP853._estimate_error_norm`` does.
    """
    ys = _stage_values(method, z)
    cls = _EMBEDDED[method]
    if method == "DOP853":
        e5 = np.abs(z * sum(w * y for w, y in zip(cls.E5, ys, strict=True)))
        e3 = np.abs(z * sum(w * y for w, y in zip(cls.E3, ys, strict=True)))
        return e5**2 / np.sqrt(e5**2 + 0.01 * e3**2)
    return np.abs(z * sum(w * y for w, y in zip(cls.E, ys, strict=True)))


def controller_spectrum(
    p: LorenzParams, method: str, rtol: float, *, t_end: float = 100.0
) -> NDArray[np.complexfloating]:
    """h_n * lambda_J(u_n) for every step the controller takes on the attractor.

    These are the points the method actually multiplies by R: one triple of
    eigenvalues per accepted step, scaled by that step's h.
    """
    sol = solve_ivp(
        rhs, (0.0, t_end), on_attractor(p), args=(p,), method=method, rtol=rtol, atol=REFERENCE_ATOL
    )
    h = np.diff(sol.t)
    eigs = np.array([np.linalg.eigvals(jacobian(u, p)) for u in sol.y.T[:-1]])
    return (h[:, None] * eigs).ravel()


def adaptive_step_usage(
    p: LorenzParams, method: str, rtol: float, *, t_end: float = 100.0
) -> float:
    """Largest step the controller takes, as a fraction of the stability bound.

    The bound is ``real_axis_limit / |min Re lambda_J|`` over the attractor: the
    most damped mode anywhere, so the fraction errs on the high side. Below 1 the
    controller's accuracy demand keeps every step inside the region and stability
    never binds; near 1 the error estimate is what holds the step at the edge.
    """
    traj = solve_adaptive(p, on_attractor(p), t_end, method=method, rtol=rtol, dt_out=t_end)
    assert traj.solver_t is not None
    h_max = float(np.diff(traj.solver_t).max())
    damping = abs(spectral_extent(p)["min_real"])
    return h_max * damping / real_axis_limit(method)
