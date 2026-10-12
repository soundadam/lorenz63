"""Work against accuracy: order pays when accuracy sets the step, not when stability does."""

from __future__ import annotations

import numpy as np
import pytest

from lorenz63.efficiency import STIFF_LAMBDA, STIFF_T_END, stiff_work, work_precision
from lorenz63.stability import real_axis_limit
from lorenz63.system import LorenzParams

CLASSICAL = LorenzParams()


def test_higher_order_work_grows_more_slowly_with_accuracy() -> None:
    rtols = np.logspace(-6, -12, 4)
    rk45 = work_precision(CLASSICAL, "RK45", rtols)
    dop853 = work_precision(CLASSICAL, "DOP853", rtols)
    assert rk45.slope() < dop853.slope() < 0.0
    assert dop853.nfev[-1] < rk45.nfev[-1]


@pytest.mark.parametrize("method", ["RK45", "DOP853"])
def test_stiff_explicit_steps_are_set_by_the_stability_region(method: str) -> None:
    run = stiff_work(method, np.array([1e-2, 1e-4]))
    predicted = STIFF_LAMBDA * STIFF_T_END / real_axis_limit(method)
    assert run.steps == pytest.approx(predicted, rel=0.02)
