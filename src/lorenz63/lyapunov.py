"""Lyapunov exponents by the Benettin algorithm, plus the Kaplan-Yorke dimension.

The exponents are themselves numerical results, so the report needs a reason to
believe them. There is an exact one available here. For any flow,

    sum_i lambda_i = time-average of tr J

and for Lorenz ``tr J = -(sigma + 1 + beta)`` is a *constant*. So the computed
spectrum must sum to -13.6667 with no fitting and no tuning. That identity is
checked in ``tests/test_lyapunov.py``. It constrains the sum only -- each
window's ``sum log|R_ii|`` is ``log det``, which Liouville fixes -- so it
vouches for the variational equation, the integration accuracy and the QR
bookkeeping, not for how the sum splits between the exponents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from lorenz63.integrate import (
    REFERENCE_ATOL,
    REFERENCE_METHOD,
    REFERENCE_RTOL,
    on_attractor,
)
from lorenz63.system import LorenzParams, jacobian, rhs

DEFAULT_RENORM_INTERVAL = 1.0
DEFAULT_TOTAL_TIME = 1000.0


def _coupled_rhs(t: float, w: NDArray[np.floating], p: LorenzParams) -> NDArray[np.floating]:
    """State and tangent matrix advanced together: u' = f(u), Q' = J(u) Q."""
    u = w[:3]
    q = w[3:].reshape(3, 3)
    du = rhs(t, u, p)
    dq = jacobian(u, p) @ q
    out = np.empty(12)
    out[:3] = du
    out[3:] = dq.ravel()
    return out


@dataclass(frozen=True, slots=True)
class LyapunovResult:
    """Full spectrum plus the running estimates that show it converging."""

    spectrum: NDArray[np.floating]  # (3,) sorted descending
    times: NDArray[np.floating]  # renormalisation times
    history: NDArray[np.floating]  # (n, 3) running estimate at each time
    params: LorenzParams
    total_time: float

    @property
    def largest(self) -> float:
        return float(self.spectrum[0])

    @property
    def total(self) -> float:
        """Sum of the exponents; must equal ``divergence(params)``."""
        return float(self.spectrum.sum())

    @property
    def kaplan_yorke(self) -> float:
        return kaplan_yorke_dimension(self.spectrum)

    @property
    def doubling_time(self) -> float:
        """Time for an infinitesimal error to double: ``ln 2 / lambda_1``."""
        return float(np.log(2.0) / self.spectrum[0])

    def predictability_horizon(self, delta0: float, threshold: float = 1.0) -> float:
        """``(1/lambda_1) * ln(threshold / delta0)`` — the log-in-precision limit."""
        return float(np.log(threshold / delta0) / self.spectrum[0])


def benettin_spectrum(
    p: LorenzParams,
    *,
    total_time: float = DEFAULT_TOTAL_TIME,
    renorm_interval: float = DEFAULT_RENORM_INTERVAL,
    u0: NDArray[np.floating] | None = None,
    transient: float | None = None,
    rtol: float = REFERENCE_RTOL,
    atol: float = REFERENCE_ATOL,
) -> LyapunovResult:
    """Full Lyapunov spectrum by repeated QR re-orthonormalisation.

    Between renormalisations the tangent matrix Q is advanced by the variational
    equation; at each renormalisation ``Q = Q_new R`` and ``log|R_ii|`` is
    accumulated. The exponents are the accumulated logs divided by elapsed time.

    Re-orthonormalising is what keeps the columns from all collapsing onto the
    most unstable direction -- without it only ``lambda_1`` would be recoverable.
    """
    if renorm_interval <= 0.0 or total_time < renorm_interval:
        raise ValueError("need 0 < renorm_interval <= total_time")

    start = on_attractor(p, u0=u0) if transient is None else on_attractor(
        p, u0=u0, transient=transient
    )

    n_windows = int(round(total_time / renorm_interval))
    accumulated = np.zeros(3)
    history = np.empty((n_windows, 3))
    times = np.empty(n_windows)

    w = np.empty(12)
    w[:3] = start
    w[3:] = np.eye(3).ravel()

    for k in range(n_windows):
        sol = solve_ivp(
            _coupled_rhs,
            (0.0, renorm_interval),
            w,
            args=(p,),
            method=REFERENCE_METHOD,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"variational integration failed: {sol.message}")
        w = sol.y[:, -1].copy()

        q, r = np.linalg.qr(w[3:].reshape(3, 3))
        # Force positive diagonal so log|R_ii| is unambiguous and Q keeps a
        # consistent orientation across windows.
        signs = np.sign(np.diag(r))
        signs[signs == 0.0] = 1.0
        q = q * signs
        diag = np.abs(np.diag(r))

        accumulated += np.log(diag)
        elapsed = (k + 1) * renorm_interval
        times[k] = elapsed
        history[k] = accumulated / elapsed
        w[3:] = q.ravel()

    spectrum = np.sort(accumulated / (n_windows * renorm_interval))[::-1]
    return LyapunovResult(
        spectrum=spectrum,
        times=times,
        history=history,
        params=p,
        total_time=n_windows * renorm_interval,
    )


def kaplan_yorke_dimension(spectrum: NDArray[np.floating]) -> float:
    """Lyapunov (Kaplan-Yorke) dimension of the attractor.

    ``D = j + (sum_{i<=j} lambda_i) / |lambda_{j+1}|`` where j is the largest
    index whose partial sum is still non-negative. For classical Lorenz the
    spectrum is roughly (0.906, 0, -14.57), giving D ~ 2.06: the attractor is
    a set of zero volume but more than a surface.
    """
    ordered = np.sort(np.asarray(spectrum, dtype=float))[::-1]
    partial = np.cumsum(ordered)
    j = int(np.searchsorted(-partial, 0.0, side="left"))
    if j == 0:
        return 0.0
    if j >= ordered.size:
        return float(ordered.size)
    return float(j + partial[j - 1] / abs(ordered[j]))


@dataclass(frozen=True, slots=True)
class FrameHistory:
    """The orthonormal frame and its per-window stretching, kept for plotting.

    ``benettin_spectrum`` throws all of this away and keeps only the running
    average. It is what the algorithm actually does at each point: carry three
    orthonormal tangent vectors along the flow, measure how much each one grew,
    straighten them out again.
    """

    times: NDArray[np.floating]  # (n,) window end times
    states: NDArray[np.floating]  # (n, 3) state at each window end
    frames: NDArray[np.floating]  # (n, 3, 3) columns are the orthonormal vectors
    log_stretch: NDArray[np.floating]  # (n, 3) log|R_ii| over that window
    renorm_interval: float
    params: LorenzParams

    @property
    def spectrum(self) -> NDArray[np.floating]:
        """Exponents implied by this history; matches ``benettin_spectrum``."""
        total = self.log_stretch.sum(axis=0) / self.times[-1]
        return np.sort(total)[::-1]

    @property
    def instantaneous_rates(self) -> NDArray[np.floating]:
        """``q_k . J(u) q_k`` at each window end, shape (n, 3).

        For ``delta u' = J delta u`` and ``e = delta u / |delta u|``,
        ``d/dt ln|delta u| = e . J e``. With ``e = q_k`` this is how fast the
        k-th frame vector grows at that instant, after Gram-Schmidt has removed
        its part along the first k-1. Its time average is ``lambda_k``. The three rates are the
        diagonal of ``Q^T J Q``, so they sum to ``tr J`` at every instant, not
        just on average -- an orthogonal change of basis leaves the trace alone.
        """
        jac = np.stack([jacobian(u, self.params) for u in self.states])
        return np.einsum("nik,nij,njk->nk", self.frames, jac, self.frames)


def tangent_frames(
    p: LorenzParams,
    *,
    total_time: float = 200.0,
    renorm_interval: float = 0.5,
    u0: NDArray[np.floating] | None = None,
) -> FrameHistory:
    """Run the Benettin loop but keep the frame and the per-window stretch.

    Same computation as :func:`benettin_spectrum`; only the bookkeeping differs.
    Used by the figure that draws the moving frame along the trajectory.
    """
    if renorm_interval <= 0.0 or total_time < renorm_interval:
        raise ValueError("need 0 < renorm_interval <= total_time")

    n_windows = int(round(total_time / renorm_interval))
    times = np.empty(n_windows)
    states = np.empty((n_windows, 3))
    frames = np.empty((n_windows, 3, 3))
    log_stretch = np.empty((n_windows, 3))

    w = np.empty(12)
    w[:3] = on_attractor(p, u0=u0)
    w[3:] = np.eye(3).ravel()

    for k in range(n_windows):
        sol = solve_ivp(
            _coupled_rhs,
            (0.0, renorm_interval),
            w,
            args=(p,),
            method=REFERENCE_METHOD,
            rtol=REFERENCE_RTOL,
            atol=REFERENCE_ATOL,
        )
        if not sol.success:
            raise RuntimeError(f"variational integration failed: {sol.message}")
        w = sol.y[:, -1].copy()

        q, r = np.linalg.qr(w[3:].reshape(3, 3))
        signs = np.sign(np.diag(r))
        signs[signs == 0.0] = 1.0
        q = q * signs

        times[k] = (k + 1) * renorm_interval
        states[k] = w[:3]
        frames[k] = q
        log_stretch[k] = np.log(np.abs(np.diag(r)))
        w[3:] = q.ravel()

    return FrameHistory(
        times=times,
        states=states,
        frames=frames,
        log_stretch=log_stretch,
        renorm_interval=renorm_interval,
        params=p,
    )


@dataclass(frozen=True, slots=True)
class TangentCollapse:
    """What happens to three tangent vectors when they are *never* re-orthonormalised."""

    times: NDArray[np.floating]  # (n,)
    log_norms: NDArray[np.floating]  # (n, 3) log|v_i(t)|
    angles: NDArray[np.floating]  # (n, 3) angles (rad) between pairs 12, 13, 23
    naive_exponents: NDArray[np.floating]  # (3,) log|v_i(T)| / T

    @property
    def spread(self) -> float:
        """Relative spread of the three naive estimates; ~0 means all three collapsed."""
        values = self.naive_exponents
        return float(np.ptp(values) / np.abs(values).max())


def unrenormalised_tangents(
    p: LorenzParams,
    *,
    total_time: float = 30.0,
    n_out: int = 600,
    u0: NDArray[np.floating] | None = None,
) -> TangentCollapse:
    """Integrate the variational equation with no QR at all, and watch it fail.

    Three initially orthonormal tangent vectors are carried along the flow
    untouched. Because every generic vector has a component along the leading
    eigendirection, and that component grows like ``exp(lambda_1 t)`` while the
    others decay, all three vectors rotate onto the *same* direction at rate
    ``lambda_1 - lambda_2``. After that:

    * the angles between them are zero to machine precision, so the frame no
      longer spans anything;
    * all three log-norms have the same slope, so the computation returns
      ``lambda_1`` three times instead of the spectrum.

    This is the reason for the re-orthonormalisation in
    :func:`benettin_spectrum`, stated as a measurement rather than an assertion.
    ``total_time`` is deliberately short: by t = 30 the angles are already down
    at 1e-12 and the curve would soon be measuring rounding, not dynamics.
    """
    w = np.empty(12)
    w[:3] = on_attractor(p, u0=u0)
    w[3:] = np.eye(3).ravel()

    t_eval = np.linspace(0.0, total_time, n_out)
    sol = solve_ivp(
        _coupled_rhs,
        (0.0, total_time),
        w,
        args=(p,),
        method=REFERENCE_METHOD,
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
        t_eval=t_eval,
    )
    if not sol.success:
        raise RuntimeError(f"variational integration failed: {sol.message}")

    tangents = sol.y[3:].T.reshape(-1, 3, 3)  # (n, row, column)
    norms = np.linalg.norm(tangents, axis=1)  # (n, 3), one per column
    unit = tangents / norms[:, None, :]

    pairs = ((0, 1), (0, 2), (1, 2))
    angles = np.empty((sol.t.size, 3))
    for k, (i, j) in enumerate(pairs):
        a, b = unit[:, :, i], unit[:, :, j]
        # arccos(dot) would be useless here: the angles reach 1e-12, and near
        # dot = 1 the arccos of a double has no resolution below ~1e-8. The
        # chord form theta = 2 asin(|a - b| / 2) stays accurate all the way
        # down. Collapse can be onto +v or -v, so take the angle to the *line*.
        chord = np.minimum(
            np.linalg.norm(a - b, axis=1),
            np.linalg.norm(a + b, axis=1),
        )
        angles[:, k] = 2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))

    return TangentCollapse(
        times=sol.t,
        log_norms=np.log(norms),
        angles=angles,
        naive_exponents=np.log(norms[-1]) / sol.t[-1],
    )


@dataclass(frozen=True, slots=True)
class FiniteTimeExponents:
    """Local stretching rates sampled along a trajectory on the attractor."""

    points: NDArray[np.floating]  # (n, 3) window start states
    values: NDArray[np.floating]  # (n,) exponent over the following window
    window: float

    @property
    def mean(self) -> float:
        return float(self.values.mean())


def finite_time_exponents(
    p: LorenzParams,
    *,
    total_time: float = 400.0,
    window: float = 0.5,
    u0: NDArray[np.floating] | None = None,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> FiniteTimeExponents:
    """Largest finite-time Lyapunov exponent over successive short windows.

    This is the quantity people mean by "the local Lyapunov exponent": it varies
    a lot over the attractor even though the long-time average is a single
    number. Where it is large, any error injected there gets amplified hardest
    downstream -- which is a reason to take small steps *there*, but note that
    it is a property of the flow, not something the step size can change.
    """
    start = on_attractor(p, u0=u0)
    n_windows = int(round(total_time / window))

    points = np.empty((n_windows, 3))
    values = np.empty(n_windows)

    w = np.empty(12)
    w[:3] = start
    w[3:] = np.eye(3).ravel()

    for k in range(n_windows):
        points[k] = w[:3]
        sol = solve_ivp(
            _coupled_rhs,
            (0.0, window),
            w,
            args=(p,),
            method=REFERENCE_METHOD,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"variational integration failed: {sol.message}")
        w = sol.y[:, -1].copy()
        q, r = np.linalg.qr(w[3:].reshape(3, 3))
        values[k] = np.log(abs(r[0, 0])) / window
        signs = np.sign(np.diag(r))
        signs[signs == 0.0] = 1.0
        w[3:] = (q * signs).ravel()

    return FiniteTimeExponents(points=points, values=values, window=window)
