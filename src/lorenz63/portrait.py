"""Phase-space geometry: regime portraits and the eigenstructure at a fixed point.

The rest of the package answers "how far can this computation be trusted"; this
module answers the prior question "what does the flow actually look like". Both
halves are needed -- the eigenvalues here are what *predict* the behaviour the
portraits show, and they are closed-form, so the portraits can be checked
against algebra rather than against intuition.

Nothing here is a new numerical method. Trajectories come from
:func:`lorenz63.integrate.solve_adaptive` at reference accuracy and the
eigenstructure comes from ``numpy.linalg.eig`` applied to
:func:`lorenz63.system.jacobian`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from lorenz63.integrate import (
    REFERENCE_ATOL,
    REFERENCE_METHOD,
    REFERENCE_RTOL,
    Trajectory,
    solve_adaptive,
)
from lorenz63.system import LorenzParams, fixed_points, jacobian

#: Which fixed point: the origin, or one of the two symmetric ones.
FixedPointName = Literal["origin", "C+", "C-"]

_FIXED_POINT_INDEX: dict[str, int] = {"origin": 0, "C+": 1, "C-": 2}


def regime_trajectory(
    rho: float,
    u0: NDArray[np.floating],
    t_end: float,
    *,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    dt_out: float = 0.002,
) -> Trajectory:
    """One trajectory at a given rho, *transient included*.

    Everywhere else in the package the transient is burned off first, because
    statistics of the transient are meaningless. Here it is the point: the way a
    trajectory falls onto the origin (rho < 1) or spirals into C+ (1 < rho <
    rho_H) is exactly what the portrait has to show.
    """
    p = LorenzParams(sigma=sigma, rho=rho, beta=beta)
    return solve_adaptive(
        p,
        np.asarray(u0, dtype=float),
        t_end,
        method=REFERENCE_METHOD,
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
        dt_out=dt_out,
    )


@dataclass(frozen=True, slots=True)
class EigenStructure:
    """The linearisation at one fixed point, in a form that can be drawn.

    ``kind`` is ``"real"`` when all three eigenvalues are real -- then
    ``directions`` holds three eigenvectors, one per eigenvalue. It is
    ``"spiral"`` when a complex-conjugate pair is present: a pair spans a real
    *plane* rather than two real lines, so ``directions`` holds the real
    eigenvector first and then ``Re(v)``, ``Im(v)`` of the pair, which are the
    two axes of the plane the trajectory spirals in.
    """

    name: str
    point: NDArray[np.floating]  # (3,)
    eigenvalues: NDArray[np.complexfloating]  # (3,) sorted by descending Re
    directions: NDArray[np.floating]  # (3, 3) unit row vectors
    kind: Literal["real", "spiral"]

    @property
    def is_stable(self) -> bool:
        return bool(self.eigenvalues.real.max() < 0.0)

    @property
    def n_unstable(self) -> int:
        """Dimension of the unstable subspace (counting a complex pair twice)."""
        return int((self.eigenvalues.real > 0.0).sum())

    @property
    def spiral_plane(self) -> NDArray[np.floating] | None:
        """The two axes spanning the rotation plane, or ``None`` if there is none."""
        if self.kind != "spiral":
            return None
        return self.directions[1:]


def eigen_structure(p: LorenzParams, which: FixedPointName) -> EigenStructure:
    """Eigenvalues and real invariant directions of J at one fixed point.

    Sorted by descending real part, so ``eigenvalues[0]`` is the least stable
    mode and ``directions[0]`` the direction a perturbation grows along fastest.
    """
    index = _FIXED_POINT_INDEX[which]
    fps = fixed_points(p)
    if index >= len(fps):
        raise ValueError(f"{which} does not exist for rho = {p.rho}")

    point = fps[index]
    values, vectors = np.linalg.eig(jacobian(point, p))
    order = np.argsort(-values.real)
    values = values[order]
    vectors = vectors[:, order]

    complex_mask = np.abs(values.imag) > 1e-12
    if not complex_mask.any():
        directions = np.real(vectors).T.copy()
        kind: Literal["real", "spiral"] = "real"
    else:
        # A conjugate pair carries no pair of real eigenvectors, only the plane
        # spanned by Re(v) and Im(v); the flow rotates inside that plane while
        # growing or decaying at the rate Re(lambda).
        real_index = int(np.flatnonzero(~complex_mask)[0])
        pair_index = int(np.flatnonzero(complex_mask)[0])
        v = vectors[:, pair_index]
        directions = np.array([np.real(vectors[:, real_index]), v.real, v.imag])
        # Order eigenvalues to match: real mode first, then the pair.
        values = np.concatenate(
            [values[[real_index]], values[complex_mask]]
        )
        kind = "spiral"

    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.where(norms > 0.0, norms, 1.0)
    return EigenStructure(
        name=which,
        point=point,
        eigenvalues=values,
        directions=directions,
        kind=kind,
    )


def all_eigen_structures(p: LorenzParams) -> list[EigenStructure]:
    """Eigenstructure at every fixed point that exists at this rho."""
    names: list[FixedPointName] = ["origin", "C+", "C-"]
    return [eigen_structure(p, name) for name in names[: len(fixed_points(p))]]


def plane_patch(
    origin: NDArray[np.floating],
    axis_u: NDArray[np.floating],
    axis_v: NDArray[np.floating],
    half_width: float,
    *,
    n: int = 2,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """``(X, Y, Z)`` meshes for the patch spanned by two axes, for ``plot_surface``.

    Used to draw an invariant plane (the spiral plane of C+) and the Poincare
    cutting plane. ``n=2`` is enough for a flat quad; larger ``n`` only matters
    if the patch is to be shaded.
    """
    s = np.linspace(-half_width, half_width, n)
    a, b = np.meshgrid(s, s)
    grid = (
        np.asarray(origin, dtype=float)[None, None, :]
        + a[..., None] * np.asarray(axis_u, dtype=float)[None, None, :]
        + b[..., None] * np.asarray(axis_v, dtype=float)[None, None, :]
    )
    return grid[..., 0], grid[..., 1], grid[..., 2]


#: The most irrational rotation number: its continued fraction is all ones, so no
#: rational p/q approximates it well and the section circle fills most evenly.
GOLDEN_ROTATION = (np.sqrt(5.0) - 1.0) / 2.0


def torus_orbit(
    omega: float,
    t: NDArray[np.floating],
    *,
    major: float = 3.0,
    minor: float = 1.0,
) -> NDArray[np.floating]:
    """``(n, 3)`` points of the flow ``theta_1' = 1, theta_2' = omega`` on a torus.

    The Lorenz system has no quasi-periodic regime, so the section plate borrows
    the textbook one: two independent rotations, the long way round at rate 1
    and the short way round at rate ``omega``. Rational ``omega`` closes the
    orbit; irrational ``omega`` never does and covers the torus densely.
    """
    t = np.asarray(t, dtype=float)
    a, b = t, omega * t
    ring = major + minor * np.cos(b)
    return np.column_stack([ring * np.cos(a), ring * np.sin(a), minor * np.sin(b)])


def torus_section(
    omega: float,
    n_returns: int,
    *,
    major: float = 3.0,
    minor: float = 1.0,
) -> NDArray[np.floating]:
    """Returns of :func:`torus_orbit` to the half-plane ``y = 0, x > 0``.

    Each lap of ``theta_1`` takes time ``2 pi``, during which ``theta_2``
    advances by ``2 pi omega``; so the return map is a rotation of the section
    circle by ``omega`` of a turn. Returned as ``(n, 2)`` points ``(x, z)``.
    """
    angle = 2.0 * np.pi * omega * np.arange(n_returns)
    return np.column_stack([major + minor * np.cos(angle), minor * np.sin(angle)])


def origin_eigenvalues_closed_form(p: LorenzParams) -> NDArray[np.floating]:
    """Eigenvalues at the origin, in closed form.

    The z-row of J decouples at the origin, leaving ``-beta`` plus the two roots
    of ``lambda^2 + (sigma + 1) lambda + sigma (1 - rho)``:

        lambda_pm = (-(sigma + 1) +- sqrt((sigma - 1)^2 + 4 sigma rho)) / 2

    For ``rho > 1`` the discriminant makes ``lambda_+`` positive: the origin is a
    saddle with a one-dimensional unstable direction. That single positive
    eigenvalue is where the whole attractor comes from -- trajectories are
    pushed off the origin along it and folded back by the nonlinearity.
    """
    root = np.sqrt((p.sigma - 1.0) ** 2 + 4.0 * p.sigma * p.rho)
    return np.array(
        [
            0.5 * (-(p.sigma + 1.0) + root),
            0.5 * (-(p.sigma + 1.0) - root),
            -p.beta,
        ]
    )
