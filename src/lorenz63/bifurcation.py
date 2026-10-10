"""Sweep rho and record the attractor's z-maxima: the bifurcation diagram."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from numpy.typing import NDArray

from lorenz63.integrate import on_attractor, solve_adaptive
from lorenz63.statistics import local_maxima
from lorenz63.system import LorenzParams, fixed_point_eigenvalues, hopf_rho


@dataclass(frozen=True, slots=True)
class BifurcationScan:
    """Flattened (rho, z_max) point cloud, ready to scatter-plot."""

    rho: NDArray[np.floating]
    z_max: NDArray[np.floating]


def rho_sweep(
    rho_values: NDArray[np.floating] | list[float],
    *,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    t_end: float = 60.0,
    transient: float = 120.0,
    dt_out: float = 0.005,
    max_points_per_rho: int = 60,
) -> BifurcationScan:
    """For each rho: burn in, integrate, keep the local maxima of z.

    Below the Hopf threshold the trajectory settles onto a fixed point and the
    maxima collapse to a single value (or vanish entirely, when z approaches its
    limit monotonically); above it they spread into a band. That contrast is the
    diagram.
    """
    rho_values = np.asarray(rho_values, dtype=float)
    all_rho: list[NDArray[np.floating]] = []
    all_z: list[NDArray[np.floating]] = []

    for rho in rho_values:
        p = LorenzParams(sigma=sigma, rho=float(rho), beta=beta)
        start = on_attractor(p, u0=np.array([1.0, 1.0, 1.0]), transient=transient)
        traj = solve_adaptive(p, start, t_end, dt_out=dt_out)
        _, peaks = local_maxima(traj.t, traj.z)
        if peaks.size == 0:
            # Monotone approach to equilibrium: record the settled z instead, so
            # the pre-Hopf branch is still visible on the diagram.
            peaks = traj.z[-1:]
        if peaks.size > max_points_per_rho:
            peaks = peaks[-max_points_per_rho:]
        all_rho.append(np.full(peaks.size, rho))
        all_z.append(peaks)

    return BifurcationScan(
        rho=np.concatenate(all_rho),
        z_max=np.concatenate(all_z),
    )


@dataclass(frozen=True, slots=True)
class FixedPointBranch:
    """Leading eigenvalue real parts of C+ as rho varies."""

    rho: NDArray[np.floating]
    max_real: NDArray[np.floating]
    hopf: float


def c_plus_stability(
    rho_values: NDArray[np.floating] | list[float],
    *,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
) -> FixedPointBranch:
    """Largest eigenvalue real part at C+, which crosses zero at the Hopf point.

    This is pure linear algebra on the closed-form fixed point -- no integration
    -- so it pins the Hopf threshold independently of any solver.
    """
    rho_values = np.asarray(rho_values, dtype=float)
    max_real = np.empty(rho_values.size)
    for i, rho in enumerate(rho_values):
        p = LorenzParams(sigma=sigma, rho=float(rho), beta=beta)
        eigs = fixed_point_eigenvalues(p)
        # Index 1 is C+; below the pitchfork only the origin exists.
        max_real[i] = float(eigs[1].real.max()) if len(eigs) == 3 else np.nan
    return FixedPointBranch(
        rho=rho_values,
        max_real=max_real,
        hopf=hopf_rho(LorenzParams(sigma=sigma, rho=28.0, beta=beta)),
    )


@dataclass(frozen=True, slots=True)
class EigenLocus:
    """Three eigenvalue branches tracked continuously as rho varies."""

    rho: NDArray[np.floating]
    values: NDArray[np.complexfloating]  # (n_rho, 3), column k is one branch
    which: str

    def branch(self, k: int) -> NDArray[np.complexfloating]:
        return self.values[:, k]


def eigenvalue_locus(
    rho_values: NDArray[np.floating] | list[float],
    *,
    which: str = "C+",
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
) -> EigenLocus:
    """Path traced in the complex plane by J's eigenvalues as rho varies.

    LAPACK returns eigenvalues in an unspecified order, so a column of the raw
    output is not a branch: it can swap partners between neighbouring rho and
    the plotted lines then jump between branches. Each rho is therefore matched
    to the previous one by nearest-neighbour assignment -- with three
    eigenvalues, checking all six permutations is exact and costs nothing.

    On the classical (sigma=10, beta=8/3) section the ordering that comes back
    happens to be continuous already, so the matching changes nothing here; it
    is what makes that a fact rather than a hope.
    """
    rho_values = np.asarray(rho_values, dtype=float)
    index = {"origin": 0, "C+": 1, "C-": 2}[which]

    values = np.empty((rho_values.size, 3), dtype=complex)
    previous: NDArray[np.complexfloating] | None = None
    for i, rho in enumerate(rho_values):
        p = LorenzParams(sigma=sigma, rho=float(rho), beta=beta)
        eigs = fixed_point_eigenvalues(p)
        if index >= len(eigs):
            values[i] = np.nan
            previous = None
            continue
        current = np.asarray(eigs[index], dtype=complex)
        if previous is None:
            current = current[np.argsort(-current.real)]
        else:
            current = _match_to(previous, current)
        values[i] = current
        previous = current

    return EigenLocus(rho=rho_values, values=values, which=which)


def _match_to(
    previous: NDArray[np.complexfloating], current: NDArray[np.complexfloating]
) -> NDArray[np.complexfloating]:
    """Permute ``current`` so each entry continues the nearest ``previous`` branch."""
    best_cost = np.inf
    best = current
    for perm in permutations(range(current.size)):
        candidate = current[list(perm)]
        cost = float(np.abs(candidate - previous).sum())
        if cost < best_cost:
            best_cost = cost
            best = candidate
    return best
