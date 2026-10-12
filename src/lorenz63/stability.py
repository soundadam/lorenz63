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

from lorenz63.integrate import on_attractor, reference_trajectory
from lorenz63.system import LorenzParams, jacobian

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
