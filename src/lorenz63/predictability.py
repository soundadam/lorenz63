"""How long two nearby solutions stay together, and how that scales with effort.

The central experiment of the report lives here: ``tolerance_sweep`` measures the
time at which a solution computed at tolerance ``tol`` becomes distinguishable
from a reference solution. Because errors grow like ``exp(lambda_1 * t)``,

    T_div = (1 / lambda_1) * ln(threshold / delta_0),   delta_0 ~ tol

so T_div is *linear in ln(1/tol)* with slope ``1 / lambda_1``. Tightening the
tolerance by ten orders of magnitude buys about 25 extra time units. That is the
quantitative form of "precision buys only logarithmic predictability".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lorenz63.integrate import (
    REFERENCE_ATOL,
    REFERENCE_RTOL,
    Trajectory,
    on_attractor,
    reference_trajectory,
    solve_adaptive,
)
from lorenz63.system import LorenzParams

#: Separation counted as "these are different trajectories now". The attractor
#: spans roughly 40 units in x, so 1.0 is a visible but not saturated gap.
DEFAULT_THRESHOLD = 1.0


def separation(a: Trajectory, b: Trajectory) -> NDArray[np.floating]:
    """Euclidean distance between two trajectories sampled on the same grid."""
    if a.t.shape != b.t.shape or not np.allclose(a.t, b.t):
        raise ValueError("trajectories must share an output grid")
    return np.linalg.norm(a.u - b.u, axis=1)


def divergence_time(
    t: NDArray[np.floating],
    delta: NDArray[np.floating],
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """First time the separation exceeds ``threshold``; ``inf`` if it never does.

    Linearly interpolated within the crossing interval so the result is not
    quantised by the output sampling.
    """
    above = np.flatnonzero(delta > threshold)
    if above.size == 0:
        return float("inf")
    i = int(above[0])
    if i == 0:
        return float(t[0])
    d0, d1 = delta[i - 1], delta[i]
    frac = (threshold - d0) / (d1 - d0)
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


@dataclass(frozen=True, slots=True)
class ToleranceSweep:
    """T_div versus tolerance, plus the straight-line fit the report quotes."""

    tolerances: NDArray[np.floating]
    divergence_times: NDArray[np.floating]  # (n_tol, n_samples)
    threshold: float
    method: str

    @property
    def mean_times(self) -> NDArray[np.floating]:
        return self.divergence_times.mean(axis=1)

    @property
    def std_times(self) -> NDArray[np.floating]:
        return self.divergence_times.std(axis=1)

    @property
    def log_inverse_tol(self) -> NDArray[np.floating]:
        """Natural log, so the fitted slope is ``1 / lambda_1`` directly."""
        return np.log(1.0 / self.tolerances)

    def fit(self) -> tuple[float, float]:
        """Least-squares ``(slope, intercept)`` of T_div against ln(1/tol)."""
        slope, intercept = np.polyfit(self.log_inverse_tol, self.mean_times, 1)
        return float(slope), float(intercept)

    @property
    def implied_lambda(self) -> float:
        """``1 / slope`` — an estimate of lambda_1 from predictability alone."""
        slope, _ = self.fit()
        return 1.0 / slope


def tolerance_sweep(
    p: LorenzParams,
    tolerances: NDArray[np.floating] | list[float],
    *,
    t_end: float = 80.0,
    n_samples: int = 8,
    method: str = "DOP853",
    threshold: float = DEFAULT_THRESHOLD,
    dt_out: float = 0.01,
    seed: int = 20260922,
) -> ToleranceSweep:
    """Measure T_div at each tolerance, averaged over several attractor states.

    Each sample uses a different starting point on the attractor (reached by a
    randomised burn-in), because T_div depends on where on the attractor you
    happen to start -- a trajectory heading into a lobe switch loses coherence
    sooner than one mid-lobe.
    """
    tolerances = np.asarray(tolerances, dtype=float)
    rng = np.random.default_rng(seed)
    starts = [
        on_attractor(p, u0=rng.uniform(-10.0, 10.0, size=3), transient=60.0)
        for _ in range(n_samples)
    ]

    times = np.empty((tolerances.size, n_samples))
    for j, start in enumerate(starts):
        truth = reference_trajectory(p, start, t_end, dt_out=dt_out)
        for i, tol in enumerate(tolerances):
            test = solve_adaptive(
                p,
                start,
                t_end,
                method=method,
                rtol=float(tol),
                atol=float(tol) * 1e-3,
                dt_out=dt_out,
            )
            times[i, j] = divergence_time(truth.t, separation(truth, test), threshold)

    return ToleranceSweep(
        tolerances=tolerances,
        divergence_times=times,
        threshold=threshold,
        method=method,
    )


@dataclass(frozen=True, slots=True)
class EnsembleGrowth:
    """Separation of slightly perturbed copies of several attractor states."""

    t: NDArray[np.floating]
    separations: NDArray[np.floating]  # (n_bases * n_members, n_times)
    delta0: float

    @property
    def median(self) -> NDArray[np.floating]:
        return np.median(self.separations, axis=0)

    def quantiles(self, lo: float = 0.1, hi: float = 0.9) -> tuple[NDArray, NDArray]:
        return (
            np.quantile(self.separations, lo, axis=0),
            np.quantile(self.separations, hi, axis=0),
        )

    def fit_exponent(self, lo: float, hi: float) -> tuple[float, float]:
        """Fit ``ln(median separation) = lambda * t + c`` over ``t in [lo, hi]``.

        The window must sit above the noise floor and below saturation: outside
        the linear-growth regime the slope means nothing.
        """
        mask = (self.t >= lo) & (self.t <= hi)
        slope, intercept = np.polyfit(self.t[mask], np.log(self.median[mask]), 1)
        return float(slope), float(intercept)


def ensemble_growth(
    p: LorenzParams,
    *,
    n_bases: int = 16,
    n_members: int = 4,
    base_spacing: float = 5.0,
    delta0: float = 1e-9,
    t_end: float = 40.0,
    dt_out: float = 0.02,
    seed: int = 20260922,
) -> EnsembleGrowth:
    """Perturb several attractor states in random directions and watch them part.

    An independent route to lambda_1: the exponential envelope of the median
    separation. It should agree with the Benettin value without sharing any code
    with it.

    The base states are ``n_bases`` points ``base_spacing`` apart along one
    trajectory, each perturbed in ``n_members`` directions. A single base state
    measures the finite-time exponent of the stretch of attractor its 10-20
    time units of growth happen to cover, which ranges from about 0.4 to 1.3;
    pooling bases spread over the attractor is what makes the median slope an
    estimate of lambda_1. Random directions add little on their own, since
    members of one base all align with its unstable direction within a few
    time units.
    """
    rng = np.random.default_rng(seed)
    path = reference_trajectory(
        p, on_attractor(p), base_spacing * (n_bases - 1), dt_out=base_spacing
    )

    seps = []
    for base_u0 in path.u:
        base = reference_trajectory(p, base_u0, t_end, dt_out=dt_out)
        directions = rng.normal(size=(n_members, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        for direction in directions:
            member = solve_adaptive(
                p,
                base_u0 + delta0 * direction,
                t_end,
                rtol=REFERENCE_RTOL,
                atol=REFERENCE_ATOL,
                dt_out=dt_out,
            )
            seps.append(separation(base, member))

    return EnsembleGrowth(t=base.t, separations=np.array(seps), delta0=delta0)
