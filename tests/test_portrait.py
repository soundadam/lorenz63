"""Phase portraits, the eigenstructure at each fixed point, and the tangent frame.

Three claims the report's phase-portrait section makes, written as assertions:

* the regimes are what the eigenvalues say they are (rho < 1: everything falls
  to the origin; 1 < rho < rho_H: everything spirals into C+/C-);
* the "eigen-directions" drawn in the figure really are invariant directions of
  J, not decorative arrows;
* re-orthonormalising is not a detail of the Lyapunov algorithm but the only
  thing that makes lambda_2 and lambda_3 recoverable at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from lorenz63.bifurcation import eigenvalue_locus
from lorenz63.integrate import reference_trajectory
from lorenz63.lyapunov import (
    benettin_spectrum,
    tangent_frames,
    unrenormalised_tangents,
)
from lorenz63.portrait import (
    all_eigen_structures,
    eigen_structure,
    origin_eigenvalues_closed_form,
    plane_patch,
    regime_trajectory,
)
from lorenz63.statistics import poincare_section, section_return_map
from lorenz63.system import (
    LorenzParams,
    fixed_point_eigenvalues,
    fixed_points,
    hopf_rho,
    jacobian,
)

CLASSICAL = LorenzParams()


# ---------------------------------------------------------------------------
# The regimes
# ---------------------------------------------------------------------------


def test_below_the_pitchfork_everything_falls_to_the_origin() -> None:
    for u0 in (np.array([12.0, 10.0, 22.0]), np.array([-9.0, -14.0, 5.0])):
        traj = regime_trajectory(0.5, u0, 30.0, dt_out=0.05)
        assert np.linalg.norm(traj.u[-1]) < 1e-6


def test_between_the_thresholds_everything_spirals_into_c_pm() -> None:
    """1 < rho < rho_H: C+/C- are stable, and the approach is oscillatory."""
    p = LorenzParams(rho=10.0)
    targets = fixed_points(p)[1:]
    for u0 in (np.array([1.0, 1.0, 1.0]), np.array([-1.0, -1.0, 1.0])):
        traj = regime_trajectory(10.0, u0, 60.0, dt_out=0.02)
        distances = np.linalg.norm(traj.u[-1] - targets, axis=1)
        assert distances.min() < 1e-6
        # Oscillatory, not monotone: z overshoots its final value repeatedly.
        final_z = traj.z[-1]
        crossings = np.flatnonzero(np.diff(np.sign(traj.z[: traj.z.size // 2] - final_z)) != 0)
        assert crossings.size >= 4


def test_the_classical_case_has_no_stable_fixed_point_left() -> None:
    assert all(not es.is_stable for es in all_eigen_structures(CLASSICAL))
    assert all(es.is_stable for es in all_eigen_structures(LorenzParams(rho=10.0))[1:])


# ---------------------------------------------------------------------------
# The eigenstructure
# ---------------------------------------------------------------------------


def test_origin_eigenvalues_match_the_closed_form() -> None:
    es = eigen_structure(CLASSICAL, "origin")
    assert es.kind == "real"
    computed = np.sort(es.eigenvalues.real)
    assert computed == pytest.approx(np.sort(origin_eigenvalues_closed_form(CLASSICAL)))
    # One positive, two negative: a saddle. The positive one is the source of
    # every trajectory on the attractor.
    assert (computed > 0.0).sum() == 1
    assert es.n_unstable == 1


def test_real_eigen_directions_are_eigenvectors() -> None:
    es = eigen_structure(CLASSICAL, "origin")
    j = jacobian(es.point, CLASSICAL)
    for value, direction in zip(es.eigenvalues, es.directions, strict=True):
        residual = j @ direction - value.real * direction
        assert np.linalg.norm(residual) < 1e-12


def test_the_spiral_plane_at_c_plus_is_invariant_under_j() -> None:
    """A complex pair spans a plane, and J maps that plane into itself."""
    es = eigen_structure(CLASSICAL, "C+")
    assert es.kind == "spiral"
    plane = es.spiral_plane
    assert plane is not None

    # Project J @ v back onto the plane; nothing may be left over.
    basis = np.linalg.qr(plane.T)[0]  # (3, 2) orthonormal columns
    for v in plane:
        image = jacobian(es.point, CLASSICAL) @ v
        leftover = image - basis @ (basis.T @ image)
        assert np.linalg.norm(leftover) < 1e-12

    # At rho = 28 the plane is the *unstable* one and the real direction contracts.
    assert es.eigenvalues[0].real < 0.0
    assert es.eigenvalues[1].real > 0.0
    assert abs(es.eigenvalues[1].imag) > 1.0


def test_c_plus_and_c_minus_are_mirror_images() -> None:
    plus, minus = eigen_structure(CLASSICAL, "C+"), eigen_structure(CLASSICAL, "C-")
    assert np.sort_complex(plus.eigenvalues) == pytest.approx(np.sort_complex(minus.eigenvalues))
    assert minus.point == pytest.approx(plus.point * np.array([-1.0, -1.0, 1.0]))


def test_plane_patch_lies_in_the_plane_it_was_given() -> None:
    origin = np.array([1.0, 2.0, 3.0])
    u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    xs, ys, zs = plane_patch(origin, u, v, 2.0, n=5)
    assert np.allclose(zs, 3.0)
    assert xs.min() == pytest.approx(-1.0)
    assert ys.max() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# The eigenvalue locus
# ---------------------------------------------------------------------------


def test_locus_branches_are_continuous_permutations_of_the_spectrum() -> None:
    """Each column is one branch (continuity) and nothing was lost (permutation)."""
    rho = np.linspace(1.05, 32.0, 600)
    locus = eigenvalue_locus(rho, which="C+")

    assert np.abs(np.diff(locus.values, axis=0)).max() < 1.0

    for i, r in enumerate(rho[::37]):
        raw = np.asarray(fixed_point_eigenvalues(LorenzParams(rho=float(r)))[1], dtype=complex)
        assert np.sort_complex(locus.values[i * 37]) == pytest.approx(np.sort_complex(raw))


def test_the_c_plus_pair_crosses_the_imaginary_axis_at_the_hopf_point() -> None:
    rho_h = hopf_rho(CLASSICAL)
    below = eigenvalue_locus(np.array([rho_h - 0.5]), which="C+").values[0]
    above = eigenvalue_locus(np.array([rho_h + 0.5]), which="C+").values[0]
    assert below.real.max() < 0.0
    assert above.real.max() > 0.0
    # It is the *complex* pair that crosses -- that is what makes it a Hopf.
    crossing = above[np.argmax(above.real)]
    assert abs(crossing.imag) > 1.0


def test_origin_locus_agrees_with_the_closed_form_everywhere() -> None:
    rho = np.linspace(0.1, 32.0, 50)
    locus = eigenvalue_locus(rho, which="origin")
    for i, r in enumerate(rho):
        expected = np.sort(origin_eigenvalues_closed_form(LorenzParams(rho=float(r))))
        assert np.sort(locus.values[i].real) == pytest.approx(expected)
        assert np.abs(locus.values[i].imag).max() < 1e-12


# ---------------------------------------------------------------------------
# The Poincare section
# ---------------------------------------------------------------------------


def test_section_return_map_stays_on_the_section() -> None:
    plane_z = CLASSICAL.rho - 1.0
    traj = reference_trajectory(CLASSICAL, np.array([1.0, 1.0, 1.0]), 200.0, dt_out=0.002)
    section = poincare_section(traj, plane_z=plane_z)
    assert section.shape[0] > 50
    assert np.abs(section[:, 2] - plane_z).max() < 1e-6

    current, following = section_return_map(section, coord=0)
    assert current.size == section.shape[0] - 1
    assert following[:-1] == pytest.approx(current[1:])


def test_a_crossing_never_lands_on_the_wrong_side_of_the_wing() -> None:
    """|x| at the section is bounded away from zero: the two wings never touch."""
    traj = reference_trajectory(CLASSICAL, np.array([1.0, 1.0, 1.0]), 400.0, dt_out=0.002)
    section = poincare_section(traj, plane_z=CLASSICAL.rho - 1.0)
    assert np.abs(section[:, 0]).min() > 1.0


# ---------------------------------------------------------------------------
# The tangent frame, and why it has to be straightened out
# ---------------------------------------------------------------------------


def test_tangent_frames_stay_orthonormal_and_reproduce_the_spectrum() -> None:
    history = tangent_frames(CLASSICAL, total_time=120.0, renorm_interval=0.5)
    for q in history.frames:
        assert q.T @ q == pytest.approx(np.eye(3), abs=1e-12)

    reference = benettin_spectrum(CLASSICAL, total_time=120.0, renorm_interval=0.5)
    # Same algorithm, same trajectory: the two must agree to round-off.
    assert history.spectrum == pytest.approx(reference.spectrum, abs=1e-9)


@pytest.mark.slow
def test_without_reorthonormalisation_only_lambda_1_survives() -> None:
    """The whole reason for the QR step, measured rather than asserted.

    Three tangent vectors carried along the flow untouched all rotate onto the
    leading direction, so all three log-norms end up with the *same* slope --
    lambda_1 three times over, with no trace of lambda_2 or lambda_3.
    """
    collapse = unrenormalised_tangents(CLASSICAL, total_time=30.0)
    lam1 = benettin_spectrum(CLASSICAL, total_time=500.0).largest

    assert collapse.angles[0] == pytest.approx(np.pi / 2, abs=1e-12)
    assert collapse.angles[-1].max() < 1e-10  # collapsed onto one line
    assert np.all(collapse.angles[-1] < collapse.angles[collapse.times.size // 2])

    # All three "exponents" are the same number, and that number is lambda_1
    # (to within the finite-time fluctuation of a 30-unit window).
    assert collapse.spread < 0.15
    assert collapse.naive_exponents.min() > 0.5 * lam1
    assert abs(collapse.naive_exponents.mean() - lam1) < 0.3

    # Nothing here resembles lambda_2 = 0 or lambda_3 = -14.6.
    assert collapse.naive_exponents.min() > 0.1
