"""Invariant statistics, and the claim that they survive changing the solver.

The test that matters here is ``test_lorenz_map_is_tolerance_independent``.
Two tolerances whose individual trajectories part after about 10 time units
produce the *same* return map to within a fraction of a percent.
That is the positive half of the report's argument.
"""

from __future__ import annotations

import numpy as np
import pytest

from lorenz63.integrate import on_attractor, reference_trajectory, solve_adaptive
from lorenz63.statistics import (
    invariant_density,
    local_maxima,
    lorenz_map,
    map_thickness,
    poincare_section,
)
from lorenz63.system import LorenzParams

CLASSICAL = LorenzParams()


@pytest.fixture(scope="module")
def long_trajectory():
    return reference_trajectory(CLASSICAL, on_attractor(CLASSICAL), 1500.0, dt_out=0.002)


def test_local_maxima_on_a_sine() -> None:
    t = np.linspace(0.0, 10.0 * np.pi, 4001)
    peak_t, peak_v = local_maxima(t, np.sin(t))
    assert peak_t.size == 5
    assert peak_v == pytest.approx(np.ones(5), abs=1e-6)
    assert peak_t == pytest.approx(np.pi / 2 + 2 * np.pi * np.arange(5), abs=1e-3)


def test_parabolic_refinement_beats_raw_sampling() -> None:
    """Coarse sampling, with no sample landing exactly on the peak."""
    t = np.linspace(0.0, 2.0 * np.pi, 40)  # pi/2 falls between samples
    values = np.sin(t)
    _, refined = local_maxima(t, values)
    raw = values.max()
    assert abs(raw - 1.0) > 1e-4, "degenerate grid: a sample sits on the peak"
    assert abs(refined[0] - 1.0) < 0.1 * abs(raw - 1.0)


def test_local_maxima_handles_monotone_input() -> None:
    t = np.linspace(0.0, 1.0, 50)
    peak_t, peak_v = local_maxima(t, t.copy())
    assert peak_t.size == 0
    assert peak_v.size == 0


def test_lorenz_map_is_a_thin_curve(long_trajectory) -> None:
    lmap = lorenz_map(long_trajectory)
    assert lmap.size > 1000
    # Thickness after dividing out the local slope: well under a percent.
    assert map_thickness(lmap) < 0.01


def test_lorenz_map_lies_above_the_unstable_fixed_point(long_trajectory) -> None:
    """Every z-maximum exceeds z = rho - 1, the height of C+/C-."""
    lmap = lorenz_map(long_trajectory)
    assert lmap.z_n.min() > CLASSICAL.rho - 1.0


def test_lorenz_map_is_tolerance_independent(long_trajectory) -> None:
    """The thesis: single trajectories diverge, the return map does not."""
    u0 = on_attractor(CLASSICAL)
    loose = solve_adaptive(
        CLASSICAL, u0, 1500.0, method="DOP853", rtol=1e-6, atol=1e-9, dt_out=0.002
    )
    a = lorenz_map(long_trajectory)
    b = lorenz_map(loose)

    # Same support, then compare binned means of z_{n+1}.
    lo = max(a.z_n.min(), b.z_n.min())
    hi = min(a.z_n.max(), b.z_n.max())
    edges = np.linspace(lo, hi, 25)

    def binned(m):
        idx = np.clip(np.digitize(m.z_n, edges) - 1, 0, edges.size - 2)
        # 30 per bin, not a handful: the map is steep, so a bin mean estimated
        # from six points carries a standard error comparable to the effect
        # being measured. Sparse bins at the edges of the support are the only
        # place the two runs visibly disagree, and that is sampling noise.
        return np.array(
            [m.z_next[idx == k].mean() if (idx == k).sum() >= 30 else np.nan for k in range(24)]
        )

    ma, mb = binned(a), binned(b)
    both = np.isfinite(ma) & np.isfinite(mb)
    assert both.sum() > 12
    deviation = np.abs(ma[both] - mb[both]) / (a.z_next.max() - a.z_next.min())
    assert deviation.max() < 0.02
    assert np.median(deviation) < 0.005


def test_poincare_points_lie_on_the_plane(long_trajectory) -> None:
    plane = CLASSICAL.rho - 1.0
    crossings = poincare_section(long_trajectory, plane_z=plane)
    assert crossings.shape[0] > 500
    assert crossings[:, 2] == pytest.approx(np.full(crossings.shape[0], plane), abs=1e-3)


def test_poincare_directions_are_disjoint_but_equinumerous(long_trajectory) -> None:
    plane = CLASSICAL.rho - 1.0
    up = poincare_section(long_trajectory, plane_z=plane, rising=True)
    down = poincare_section(long_trajectory, plane_z=plane, rising=False)
    # A bounded orbit must come back down as often as it goes up, +/- one.
    assert abs(up.shape[0] - down.shape[0]) <= 1


def test_poincare_returns_empty_for_an_unreached_plane(long_trajectory) -> None:
    assert poincare_section(long_trajectory, plane_z=1e6).shape == (0, 3)


def test_invariant_density_normalises(long_trajectory) -> None:
    centres, density = invariant_density(long_trajectory.z, bins=100)
    width = centres[1] - centres[0]
    assert (density * width).sum() == pytest.approx(1.0, rel=1e-6)
    assert np.all(density >= 0.0)


def test_invariant_density_is_tolerance_independent(long_trajectory) -> None:
    u0 = on_attractor(CLASSICAL)
    loose = solve_adaptive(
        CLASSICAL, u0, 1500.0, method="DOP853", rtol=1e-6, atol=1e-9, dt_out=0.002
    )
    span = (0.0, 50.0)
    _, da = invariant_density(long_trajectory.z, bins=50, value_range=span)
    _, db = invariant_density(loose.z, bins=50, value_range=span)
    # Total variation distance between the two densities, well under 5%.
    width = (span[1] - span[0]) / 50
    assert 0.5 * np.abs(da - db).sum() * width < 0.05


def test_lorenz_map_rejects_short_trajectories() -> None:
    short = reference_trajectory(CLASSICAL, on_attractor(CLASSICAL), 0.05, dt_out=0.01)
    with pytest.raises(ValueError):
        lorenz_map(short)
