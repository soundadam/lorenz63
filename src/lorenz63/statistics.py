"""Invariant statistics: the Lorenz map, Poincare sections, invariant densities.

This is the other half of the report's argument. Individual trajectories from
different solvers stop agreeing after ~40 time units, but the objects in this
module are computed from *long* runs and come out identical whichever solver
produced them. They are what a chaotic computation can legitimately claim.

The Lorenz map -- plot each local maximum of z against the next one -- is
Lorenz's own construction from the 1963 paper. The points fall on a thin cusped
curve whose slope exceeds 1 in magnitude everywhere, which is why the dynamics
is chaotic: no fixed point of the map can be stable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lorenz63.integrate import Trajectory


def local_maxima(
    t: NDArray[np.floating], values: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Interior local maxima with sub-sample parabolic refinement.

    Refinement matters here. Taking the raw sampled maximum smears the Lorenz
    map into a visible band whose width is set by the output step rather than by
    the dynamics, which would undercut the whole point of the figure.
    """
    interior = np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])) + 1
    if interior.size == 0:
        return np.empty(0), np.empty(0)

    left = values[interior - 1]
    mid = values[interior]
    right = values[interior + 1]
    curvature = left - 2.0 * mid + right

    # Fall back to the sampled point where the parabola is degenerate.
    offset = np.zeros_like(mid)
    ok = curvature != 0.0
    offset[ok] = 0.5 * (left[ok] - right[ok]) / curvature[ok]
    offset = np.clip(offset, -0.5, 0.5)

    peak = mid - 0.25 * (left - right) * offset
    dt = np.diff(t)
    spacing = np.where(offset >= 0.0, dt[interior], dt[interior - 1])
    peak_t = t[interior] + offset * spacing
    return peak_t, peak


@dataclass(frozen=True, slots=True)
class LorenzMap:
    """Successive maxima of z, the (z_n, z_{n+1}) pairs and their source."""

    z_n: NDArray[np.floating]
    z_next: NDArray[np.floating]
    label: str

    @property
    def size(self) -> int:
        return int(self.z_n.size)


def lorenz_map(traj: Trajectory, *, label: str | None = None) -> LorenzMap:
    """Build the return map of successive local maxima of z."""
    _, peaks = local_maxima(traj.t, traj.z)
    if peaks.size < 2:
        raise ValueError("trajectory too short to build a return map")
    return LorenzMap(
        z_n=peaks[:-1],
        z_next=peaks[1:],
        label=label if label is not None else traj.label,
    )


def poincare_section(
    traj: Trajectory,
    *,
    plane_z: float,
    rising: bool = True,
) -> NDArray[np.floating]:
    """Crossings of the plane ``z = plane_z``, linearly interpolated.

    Returns an (n, 3) array of crossing states. ``rising`` selects the direction
    of travel through the plane, so each loop contributes one point.
    """
    z = traj.z - plane_z
    if rising:
        idx = np.flatnonzero((z[:-1] <= 0.0) & (z[1:] > 0.0))
    else:
        idx = np.flatnonzero((z[:-1] >= 0.0) & (z[1:] < 0.0))
    if idx.size == 0:
        return np.empty((0, 3))

    frac = (-z[idx] / (z[idx + 1] - z[idx]))[:, None]
    return traj.u[idx] + frac * (traj.u[idx + 1] - traj.u[idx])


def section_return_map(
    section: NDArray[np.floating], *, coord: int = 0
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """First-return map on a Poincare section: ``(s_n, s_{n+1})`` for one coordinate.

    The section reduces the three-dimensional flow to a two-dimensional map, and
    because the attractor is so thin transverse to its own sheets (D_KY ~ 2.06)
    that map in turn collapses onto a one-dimensional curve. Plotting one
    coordinate of successive crossings makes that collapse visible: the points
    fall on a curve, not into a cloud.

    This is the same construction as :func:`lorenz_map`, but on crossings of a
    plane rather than on maxima of z -- two independent reductions that have to
    agree, and do.
    """
    if section.shape[0] < 2:
        raise ValueError("need at least two crossings to build a return map")
    values = section[:, coord]
    return values[:-1], values[1:]


def wing_switch_threshold(
    s_n: NDArray[np.floating], s_next: NDArray[np.floating]
) -> float:
    """The |s| that separates crossings which stay on their wing from those that switch.

    On the classical attractor the choice is not statistical: every crossing
    inside a cutoff |x| stays on its wing (s_{n+1} has the same sign) and every
    crossing outside it switches. Returns the midpoint of the gap between the
    outermost staying crossing and the innermost switching one; raises if the
    two sets overlap, because then no single cutoff exists.
    """
    reach = np.abs(s_n)
    stays = np.sign(s_n) == np.sign(s_next)
    if stays.all() or not stays.any():
        raise ValueError("need both staying and switching crossings")
    outer_stay = float(reach[stays].max())
    inner_switch = float(reach[~stays].min())
    if outer_stay >= inner_switch:
        raise ValueError("staying and switching crossings overlap; no single cutoff")
    return 0.5 * (outer_stay + inner_switch)


def invariant_density(
    values: NDArray[np.floating],
    *,
    bins: int = 120,
    value_range: tuple[float, float] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Normalised histogram (bin centres, density). Integrates to 1."""
    density, edges = np.histogram(values, bins=bins, range=value_range, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, density


def map_thickness(lmap: LorenzMap, *, bins: int = 40, min_per_bin: int = 8) -> float:
    """Thickness of the cusped curve, as a fraction of the ``z_{n+1}`` range.

    Measuring raw within-bin spread would be wrong: the Lorenz map is *steep*
    (|F'| > 1 everywhere -- that is precisely why the dynamics is chaotic), so a
    bin of finite width maps to a wider interval even for a perfect function.
    The slope has to be divided out first.

    So within each bin we fit ``z_{n+1} = a + b*z_n`` and take the standard
    deviation of the residuals. What remains is curve thickness, independent of
    slope. The result is small -- a couple of percent -- but not zero: the true
    return map is a fractal of dimension slightly above 1, and it is also being
    sampled by a finite-length trajectory.
    """
    edges = np.linspace(lmap.z_n.min(), lmap.z_n.max(), bins + 1)
    which = np.clip(np.digitize(lmap.z_n, edges) - 1, 0, bins - 1)
    full_range = float(lmap.z_next.max() - lmap.z_next.min())

    residual_stds: list[float] = []
    for b in range(bins):
        members = which == b
        if int(members.sum()) < min_per_bin:
            continue
        xs = lmap.z_n[members]
        ys = lmap.z_next[members]
        if xs.max() - xs.min() < 1e-12:
            continue
        slope, intercept = np.polyfit(xs, ys, 1)
        residual_stds.append(float(np.std(ys - (slope * xs + intercept))))

    if not residual_stds:
        raise ValueError("no bin had enough points; use a longer trajectory")
    return float(np.median(residual_stds)) / full_range
