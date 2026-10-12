"""Every figure in ``report/report.md``, one function each.

Each figure is a computation, not an illustration: the numbers annotated on the
axes are produced by the run that draws them. Expensive intermediates are cached
under ``.cache/`` so that re-drawing is fast, but deleting the cache and
re-running must reproduce the figures exactly -- every random draw is seeded.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from lorenz63.bifurcation import (  # noqa: E402
    c_plus_stability,
    eigenvalue_locus,
    rho_sweep,
)
from lorenz63.convection import convection_field, nusselt  # noqa: E402
from lorenz63.hamiltonian import (  # noqa: E402
    DEFAULT_ECCENTRICITY,
    KEPLER_PERIOD,
    RK4_STAGES,
    angular_momentum,
    horizon,
    kepler_energy,
    kepler_rk4,
    kepler_verlet,
    position_error,
    running_max,
)
from lorenz63.integrate import (  # noqa: E402
    on_attractor,
    reference_trajectory,
    solve_adaptive,
)
from lorenz63.lyapunov import (  # noqa: E402
    benettin_spectrum,
    finite_time_exponents,
    tangent_frames,
    unrenormalised_tangents,
)
from lorenz63.portrait import (  # noqa: E402
    GOLDEN_ROTATION,
    all_eigen_structures,
    plane_patch,
    regime_trajectory,
    torus_orbit,
    torus_section,
)
from lorenz63.predictability import (  # noqa: E402
    DEFAULT_THRESHOLD,
    divergence_time,
    ensemble_growth,
    separation,
    tolerance_sweep,
)
from lorenz63.stability import (  # noqa: E402
    attractor_eigenvalues,
    max_stable_step,
    rk4_stability_mask,
)
from lorenz63.statistics import (  # noqa: E402
    invariant_density,
    lorenz_map,
    poincare_section,
    section_return_map,
    wing_switch_threshold,
)
from lorenz63.system import (  # noqa: E402
    RHO_HOMOCLINIC,
    LorenzParams,
    divergence,
    fixed_points,
    hopf_rho,
    jacobian,
    rhs,
)

DPI = 150
CLASSICAL = LorenzParams()

#: Labels are Chinese, so matplotlib needs a CJK face or it renders tofu boxes.
#: Tried in order; the first one installed wins. If none is present the figures
#: still draw (with missing glyphs) and a warning is emitted -- the committed
#: PNGs in report/figures/ are the deliverable, so a fontless CI box is fine.
CJK_FONT_CANDIDATES = (
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
)


def setup_fonts() -> str | None:
    """Point matplotlib at an installed CJK font. Returns the one chosen."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in CJK_FONT_CANDIDATES:
        if candidate in installed:
            plt.rcParams["font.sans-serif"] = [candidate, *plt.rcParams["font.sans-serif"]]
            # A CJK face renders U+2212 as a full-width glyph; use ASCII hyphen.
            plt.rcParams["axes.unicode_minus"] = False
            return candidate

    import warnings

    warnings.warn(
        "No CJK font found; Chinese labels will render as boxes. "
        f"Install one of: {', '.join(CJK_FONT_CANDIDATES)}",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


setup_fonts()

# A consistent, colour-blind-safe set used across every figure.
C_REF = "#1f3b73"
C_LOOSE = "#c1440e"
C_MID = "#0d7d6f"
C_ACCENT = "#8b3a8b"
C_MUTED = "#8a8a8a"


def _style(ax: plt.Axes) -> plt.Axes:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    return ax


def _finish(fig: plt.Figure, output: Path, name: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _cache_dir() -> Path:
    return Path(os.environ.get("LORENZ63_CACHE", ".cache"))


def _cached(name: str, compute: Callable[[], dict[str, NDArray]]) -> dict[str, NDArray]:
    """Memoise an expensive array computation to ``.cache/<name>.npz``."""
    path = _cache_dir() / f"{name}.npz"
    if path.exists():
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return result


# ---------------------------------------------------------------------------
# 01 — the opening paradox
# ---------------------------------------------------------------------------


def fig01_paradox(output: Path) -> Path:
    """Same equation, same initial state, two tolerances, two different answers."""
    u0 = on_attractor(CLASSICAL)
    t_end = 60.0
    truth = reference_trajectory(CLASSICAL, u0, t_end, dt_out=0.005)
    loose = solve_adaptive(
        CLASSICAL, u0, t_end, method="DOP853", rtol=1e-6, atol=1e-9, dt_out=0.005
    )
    tight = solve_adaptive(
        CLASSICAL, u0, t_end, method="DOP853", rtol=1e-12, atol=1e-15, dt_out=0.005
    )

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)

    ax = _style(axes[0])
    ax.plot(loose.t, loose.x, color=C_LOOSE, lw=1.0, label="DOP853  rtol=1e-6")
    ax.plot(tight.t, tight.x, color=C_REF, lw=1.0, label="DOP853  rtol=1e-12")
    ax.set_ylabel("x(t)")
    ax.set_title("同一方程、同一初值、两个容差：谁是对的？")
    ax.legend(loc="upper left", fontsize=9)

    ax = _style(axes[1])
    ax.semilogy(truth.t, np.maximum(separation(truth, loose), 1e-18), color=C_LOOSE, lw=1.0)
    ax.semilogy(truth.t, np.maximum(separation(truth, tight), 1e-18), color=C_REF, lw=1.0)
    ax.axhline(1.0, color=C_MUTED, ls="--", lw=0.9)
    ax.text(1.0, 1.4, "可分辨阈值 = 1", fontsize=8, color=C_MUTED)
    ax.set_xlabel("t")
    ax.set_ylabel("与参考解的距离")
    ax.set_ylim(1e-16, 1e2)

    return _finish(fig, output, "fig01_paradox")


# ---------------------------------------------------------------------------
# 02 — the attractor
# ---------------------------------------------------------------------------


def fig02_attractor(output: Path) -> Path:
    traj = reference_trajectory(CLASSICAL, on_attractor(CLASSICAL), 100.0, dt_out=0.002)
    fps = np.array([[0, 0, 0], [np.sqrt(72), np.sqrt(72), 27], [-np.sqrt(72), -np.sqrt(72), 27]])

    fig = plt.figure(figsize=(10.0, 7.5))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(traj.x, traj.y, traj.z, color=C_REF, lw=0.35, alpha=0.85)
    ax3d.scatter(*fps.T, color=C_LOOSE, s=22, depthshade=False, zorder=5)
    ax3d.set_xlabel("x", labelpad=-6)
    ax3d.set_ylabel("y", labelpad=-6)
    ax3d.set_zlabel("z", labelpad=-6)
    ax3d.tick_params(labelsize=7)
    ax3d.set_title("吸引子", fontsize=10)

    for k, (i, j, li, lj) in enumerate(
        [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")], start=2
    ):
        ax = _style(fig.add_subplot(2, 2, k))
        ax.plot(traj.u[:, i], traj.u[:, j], color=C_REF, lw=0.3, alpha=0.8)
        ax.scatter(fps[:, i], fps[:, j], color=C_LOOSE, s=20, zorder=5)
        ax.set_xlabel(li)
        ax.set_ylabel(lj)
        ax.set_title(f"{li}–{lj} 投影", fontsize=10)

    fig.suptitle(
        "Lorenz 吸引子（σ=10, ρ=28, β=8/3），红点为三个不动点", fontsize=11
    )
    return _finish(fig, output, "fig02_attractor")


# ---------------------------------------------------------------------------
# 03 / 04 — bifurcation structure
# ---------------------------------------------------------------------------


def fig03_bifurcation(output: Path) -> Path:
    data = _cached(
        "bifurcation",
        lambda: _bifurcation_data(),
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    _style(ax)
    ax.plot(data["rho"], data["z_max"], ".", color=C_REF, ms=0.7, alpha=0.5)

    for value, label, colour in [
        (1.0, "ρ=1\n叉式分岔", C_MID),
        (RHO_HOMOCLINIC, f"ρ≈{RHO_HOMOCLINIC}\n同宿爆炸", C_ACCENT),
        (hopf_rho(CLASSICAL), f"ρ_H={hopf_rho(CLASSICAL):.3f}\n亚临界 Hopf", C_LOOSE),
    ]:
        ax.axvline(value, color=colour, ls="--", lw=1.1)
        ax.text(value + 0.25, ax.get_ylim()[1] * 0.02, label, fontsize=8, color=colour)

    ax.set_xlabel("ρ")
    ax.set_ylabel("z 的局部极大")
    ax.set_title("分岔图：σ=10, β=8/3 固定，ρ 扫描")
    return _finish(fig, output, "fig03_bifurcation")


def _bifurcation_data() -> dict[str, NDArray]:
    scan = rho_sweep(np.linspace(0.5, 32.0, 240), t_end=50.0, transient=140.0)
    return {"rho": scan.rho, "z_max": scan.z_max}


def fig04_fixed_point_stability(output: Path) -> Path:
    branch = c_plus_stability(np.linspace(1.01, 32.0, 800))
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    _style(ax)
    ax.plot(branch.rho, branch.max_real, color=C_REF, lw=1.6)
    ax.axhline(0.0, color="k", lw=0.9)
    ax.axvline(branch.hopf, color=C_LOOSE, ls="--", lw=1.2)
    ax.annotate(
        f"ρ_H = {branch.hopf:.4f}\n实部穿零，C± 失稳",
        xy=(branch.hopf, 0.0),
        xytext=(branch.hopf - 11.0, 1.2),
        fontsize=9,
        color=C_LOOSE,
        arrowprops={"arrowstyle": "->", "color": C_LOOSE, "lw": 1.0},
    )
    ax.set_xlabel("ρ")
    ax.set_ylabel("C+ 处特征值实部的最大值")
    ax.set_title("不动点 C± 的线性稳定性（纯代数，不含任何积分）")
    ax.set_ylim(-2.0, 2.5)
    return _finish(fig, output, "fig04_fixed_point_stability")


# ---------------------------------------------------------------------------
# 05 — there is no structure to preserve
# ---------------------------------------------------------------------------


def _volume_factor(
    f_jac: Callable[[NDArray], NDArray],
    f_rhs: Callable[[float, NDArray], NDArray],
    u0: NDArray,
    t_end: float,
    n_out: int = 400,
) -> tuple[NDArray, NDArray]:
    """log|det M(t)| where M' = J M, M(0) = I. This is the volume growth factor."""
    dim = u0.size

    def coupled(t: float, w: NDArray) -> NDArray:
        u = w[:dim]
        m = w[dim:].reshape(dim, dim)
        return np.concatenate([f_rhs(t, u), (f_jac(u) @ m).ravel()])

    w0 = np.concatenate([u0, np.eye(dim).ravel()])
    t_eval = np.linspace(0.0, t_end, n_out)
    sol = solve_ivp(
        coupled, (0.0, t_end), w0, t_eval=t_eval, method="DOP853", rtol=1e-12, atol=1e-14
    )
    dets = np.array(
        [np.linalg.slogdet(sol.y[dim:, k].reshape(dim, dim))[1] for k in range(sol.t.size)]
    )
    return sol.t, dets


def fig05_volume(output: Path) -> Path:
    """Hamiltonian: volume exactly preserved. Lorenz: exponentially destroyed."""
    omega = 1.0
    t_end = 2.0

    t_h, log_h = _volume_factor(
        lambda u: np.array([[0.0, 1.0], [-(omega**2), 0.0]]),
        lambda t, u: np.array([u[1], -(omega**2) * u[0]]),
        np.array([1.0, 0.0]),
        t_end,
    )
    t_l, log_l = _volume_factor(
        lambda u: jacobian(u, CLASSICAL),
        lambda t, u: rhs(t, u, CLASSICAL),
        on_attractor(CLASSICAL),
        t_end,
    )

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    _style(ax)
    ax.plot(t_h, log_h, color=C_MID, lw=2.0, label="谐振子（哈密顿）：tr J = 0")
    ax.plot(t_l, log_l, color=C_LOOSE, lw=2.0, label="Lorenz：数值计算的 log|det M|")
    exact = divergence(CLASSICAL) * t_l
    ax.plot(t_l, exact, color="k", ls="--", lw=1.1, label=f"精确 {divergence(CLASSICAL):.4f}·t")

    ax.set_xlabel("t")
    ax.set_ylabel("log |det M(t)|   （相空间体积的对数）")
    ax.set_title("有没有结构可保？谐振子体积守恒，Lorenz 体积以固定指数率坍缩")
    ax.legend(loc="lower left", fontsize=9)
    # Quote the fitted slope, not the raw deviation in the log: log|det M|
    # reaches -27 here, so an absolute gap of 1e-2 there is a relative error of
    # 5e-4 and reads as much worse than it is. The slope is the claim.
    fitted_slope = float(np.polyfit(t_l, log_l, 1)[0])
    ax.text(
        0.97,
        0.72,
        f"数值拟合斜率 {fitted_slope:.6f}\n"
        f"精确值      {divergence(CLASSICAL):.6f}   (相对误差 "
        f"{abs(fitted_slope - divergence(CLASSICAL)) / abs(divergence(CLASSICAL)):.1e})\n"
        f"2 个时间单位内体积缩小 {np.exp(-exact[-1]):.2e} 倍\n"
        f"守恒系统靠不变量验证这条路，在 Lorenz 上没有起点",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "fc": "white", "ec": C_MUTED, "alpha": 0.95},
    )
    return _finish(fig, output, "fig05_volume")


# ---------------------------------------------------------------------------
# 06 — stability is not the binding constraint
# ---------------------------------------------------------------------------


def fig06_stability(output: Path) -> Path:
    eigs = attractor_eigenvalues(CLASSICAL, t_end=150.0, dt_out=0.02)
    h_max = max_stable_step(CLASSICAL, t_end=150.0)

    re = np.linspace(-3.5, 1.5, 500)
    im = np.linspace(-3.5, 3.5, 700)
    mask = rk4_stability_mask(re, im)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9))
    for ax, h, title in (
        (axes[0], 0.01, "实用步长 h = 0.01"),
        (axes[1], h_max, f"稳定性上限 h = {h_max:.3f}"),
    ):
        _style(ax)
        ax.contourf(re, im, mask, levels=[0.5, 1.5], colors=[C_MID], alpha=0.18)
        ax.contour(re, im, mask, levels=[0.5], colors=[C_MID], linewidths=1.4)
        scaled = h * eigs
        damped = scaled[scaled.real < 0.0]
        growing = scaled[scaled.real >= 0.0]
        ax.plot(damped.real, damped.imag, ".", color=C_REF, ms=1.2, alpha=0.5)
        ax.plot(growing.real, growing.imag, ".", color=C_LOOSE, ms=1.2, alpha=0.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Re(h·λ_J)")
        ax.set_ylabel("Im(h·λ_J)")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(re[0], re[-1])
        ax.set_ylim(im[0], im[-1])

    axes[0].text(
        0.03,
        0.95,
        "蓝：衰减模 Re λ<0\n红：增长模 Re λ>0（物理的，\n步长改变不了）",
        transform=axes[0].transAxes,
        fontsize=8,
        va="top",
        bbox={"boxstyle": "round", "fc": "white", "ec": C_MUTED, "alpha": 0.9},
    )
    fig.suptitle(
        "RK4 稳定域与吸引子上的 Jacobian 谱：实用步长离稳定性边界还差一个数量级",
        fontsize=11,
    )
    return _finish(fig, output, "fig06_stability")


# ---------------------------------------------------------------------------
# 07 / 08 — how fast, and how do we know
# ---------------------------------------------------------------------------


def fig07_ensemble(output: Path) -> Path:
    data = _cached("ensemble", _ensemble_data)
    t = data["t"]
    median = data["median"]
    lo, hi = data["lo"], data["hi"]
    slope, intercept = float(data["slope"]), float(data["intercept"])

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    _style(ax)
    ax.fill_between(t, lo, hi, color=C_REF, alpha=0.18, label="10–90 分位")
    ax.semilogy(t, median, color=C_REF, lw=1.6, label="中位分离度")
    fit_window = (t >= 5.0) & (t <= 22.0)
    ax.semilogy(
        t[fit_window],
        np.exp(intercept + slope * t[fit_window]),
        color=C_LOOSE,
        ls="--",
        lw=1.8,
        label=f"指数拟合  λ = {slope:.4f}",
    )
    ax.set_xlabel("t")
    ax.set_ylabel("‖δ(t)‖")
    ax.set_title("16 个吸引子点 × 4 个方向（δ₀ = 1e-9）的分离：指数增长，然后在吸引子尺度饱和")
    ax.legend(loc="lower right", fontsize=9)
    return _finish(fig, output, "fig07_ensemble")


def _ensemble_data() -> dict[str, NDArray]:
    growth = ensemble_growth(
        CLASSICAL, n_bases=16, n_members=4, delta0=1e-9, t_end=40.0, dt_out=0.02
    )
    lo, hi = growth.quantiles()
    slope, intercept = growth.fit_exponent(5.0, 22.0)
    return {
        "t": growth.t,
        "median": growth.median,
        "lo": lo,
        "hi": hi,
        "slope": np.array(slope),
        "intercept": np.array(intercept),
    }


def fig08_lyapunov(output: Path) -> Path:
    data = _cached("lyapunov", _lyapunov_data)
    times, history, spectrum = data["times"], data["history"], data["spectrum"]

    fig = plt.figure(figsize=(11.5, 5.2))
    grid = fig.add_gridspec(2, 2, hspace=0.12, wspace=0.28)

    # lambda_3 is ~16x larger in magnitude than lambda_1, so a shared y-axis
    # would flatten the two interesting curves into a single line at zero.
    ax_top = _style(fig.add_subplot(grid[0, 0]))
    ax_bot = _style(fig.add_subplot(grid[1, 0], sharex=ax_top))

    for k, ax, colour in ((0, ax_top, C_LOOSE), (1, ax_top, C_MID), (2, ax_bot, C_REF)):
        ax.plot(times, history[:, k], color=colour, lw=1.2)
        ax.axhline(spectrum[k], color=colour, ls=":", lw=1.0)
        ax.text(
            times[-1] * 0.99,
            spectrum[k],
            f" λ{k + 1} = {spectrum[k]:.4f} ",
            fontsize=9,
            color=colour,
            va="bottom",
            ha="right",
        )
    ax_top.set_ylim(-0.45, 1.35)
    ax_bot.set_ylim(-15.3, -14.1)
    ax_top.tick_params(labelbottom=False)
    ax_bot.set_xlabel("积分时间")
    ax_top.set_ylabel("λ₁, λ₂ 的运行估计")
    ax_bot.set_ylabel("λ₃")
    ax_top.set_title("Benettin 算法的收敛", fontsize=10)

    ax = _style(fig.add_subplot(grid[:, 1]))
    running_sum = history.sum(axis=1)
    exact = divergence(CLASSICAL)
    ax.semilogy(times, np.abs(running_sum - exact), color=C_ACCENT, lw=1.3)
    ax.set_xlabel("积分时间")
    ax.set_ylabel("|Σλ − tr J|")
    ax.set_title("与精确恒等式的偏差", fontsize=10)
    ax.text(
        0.04,
        0.06,
        f"sum lambda = {spectrum.sum():.9f}\n"
        f"tr J       = {exact:.9f}\n"
        f"difference = {abs(spectrum.sum() - exact):.2e}",
        transform=ax.transAxes,
        fontsize=9,
        family="monospace",
        va="bottom",
        bbox={"boxstyle": "round", "fc": "white", "ec": C_MUTED, "alpha": 0.9},
    )

    fig.suptitle("Lyapunov 谱，以及凭什么相信它：Σλ 必须等于常数 tr J", fontsize=11)
    return _finish(fig, output, "fig08_lyapunov")


def _lyapunov_data() -> dict[str, NDArray]:
    res = benettin_spectrum(CLASSICAL, total_time=2000.0, renorm_interval=1.0)
    return {
        "times": res.times,
        "history": res.history,
        "spectrum": res.spectrum,
        "kaplan_yorke": np.array(res.kaplan_yorke),
    }


# ---------------------------------------------------------------------------
# 09 — the core figure
# ---------------------------------------------------------------------------


def fig09_tolerance(output: Path) -> Path:
    """Predictable time is linear in ln(1/tol). Slope is 1/lambda_1."""
    data = _cached("tolerance", _tolerance_data)
    x = data["log_inv_tol"]
    mean, std = data["mean"], data["std"]
    slope, intercept = float(data["slope"]), float(data["intercept"])
    lam = float(data["lambda_benettin"])

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    _style(ax)
    ax.errorbar(
        x,
        mean,
        yerr=std,
        fmt="o",
        color=C_REF,
        ms=6,
        capsize=3,
        lw=1.2,
        label="测量值（8 个初值平均）",
    )
    xs = np.linspace(x.min() - 1.0, x.max() + 1.0, 10)
    ax.plot(
        xs,
        intercept + slope * xs,
        color=C_LOOSE,
        lw=1.8,
        label=f"最小二乘拟合  斜率 = {slope:.4f}",
    )
    ax.plot(
        xs,
        intercept + xs / lam,
        color=C_MID,
        ls="--",
        lw=1.6,
        label=f"理论斜率 1/λ₁ = {1 / lam:.4f}",
    )

    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    decades = np.arange(4, 13, 2)
    top.set_xticks(np.log(10.0**decades))
    top.set_xticklabels([f"1e-{d}" for d in decades])
    top.set_xlabel("rtol")

    ax.set_xlabel("ln(1/rtol)")
    ax.set_ylabel("可分辨时间 T_div")
    ax.set_title("精度只换来对数的预报时间", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    gained = mean[-1] - mean[0]
    orders = np.log10(1.0 / data["tolerances"][-1]) - np.log10(1.0 / data["tolerances"][0])
    ax.text(
        0.97,
        0.06,
        f"容差收紧 {orders:.0f} 个数量级\n只买到 {gained:.1f} 个时间单位\n"
        f"（理论值 {orders * np.log(10) / lam:.1f}）",
        transform=ax.transAxes,
        fontsize=9.5,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "fc": "#fff6ef", "ec": C_LOOSE, "alpha": 0.95},
    )
    return _finish(fig, output, "fig09_tolerance")


def _tolerance_data() -> dict[str, NDArray]:
    tols = np.logspace(-4, -12, 9)
    sweep = tolerance_sweep(CLASSICAL, tols, t_end=80.0, n_samples=8)
    slope, intercept = sweep.fit()
    # Same integration length as `lorenz63 info` and figure 08, so the report
    # and the figures quote one lambda_1, not two slightly different ones.
    lam = benettin_spectrum(CLASSICAL, total_time=2000.0).largest
    return {
        "tolerances": sweep.tolerances,
        "log_inv_tol": sweep.log_inverse_tol,
        "mean": sweep.mean_times,
        "std": sweep.std_times,
        "slope": np.array(slope),
        "intercept": np.array(intercept),
        "lambda_benettin": np.array(lam),
    }


# ---------------------------------------------------------------------------
# 10 — the thesis in one picture
# ---------------------------------------------------------------------------


def fig10_trajectory_vs_statistics(output: Path) -> Path:
    """Left: the trajectories are unrelated. Right: the return map is identical."""
    data = _cached("thesis", _thesis_data)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    labels = ["DOP853 rtol=1e-6", "DOP853 rtol=1e-9", "DOP853 rtol=1e-13"]
    colours = [C_LOOSE, C_MID, C_REF]

    ax = _style(axes[0])
    t_div = data["t_div"]
    for k, (label, colour) in enumerate(zip(labels, colours, strict=True)):
        ax.plot(data["t_short"], data[f"x{k}"], color=colour, lw=1.0, label=label)
    # Mark where each run stops tracking the reference, measured not asserted.
    for k, (colour, height) in enumerate(zip(colours[:2], (19.8, 16.8), strict=True)):
        ax.axvline(t_div[k], color=colour, ls="--", lw=1.2, alpha=0.8)
        ax.text(
            t_div[k] + 0.5,
            height,
            f"T_div = {t_div[k]:.1f}",
            fontsize=8.5,
            color=colour,
            va="top",
        )
    ax.set_xlim(0.0, 45.0)
    ax.set_ylim(-21.0, 21.0)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title("单条轨道：各自在 T_div 后与参考解脱钩", fontsize=11)
    ax.legend(loc="lower left", fontsize=8.5, ncols=3, columnspacing=0.9)

    ax = _style(axes[1])
    for k, (label, colour) in enumerate(zip(labels, colours, strict=True)):
        ax.plot(
            data[f"zn{k}"], data[f"znext{k}"], ".", color=colour, ms=2.2, alpha=0.55, label=label
        )
    lo = min(data["zn0"].min(), data["zn2"].min())
    hi = max(data["zn0"].max(), data["zn2"].max())
    ax.plot([lo, hi], [lo, hi], color="k", ls=":", lw=0.9)
    ax.set_xlabel("z 的第 n 个极大")
    ax.set_ylabel("第 n+1 个极大")
    ax.set_title("同样三次计算的 Lorenz 映射：重合到线宽以内", fontsize=11)
    ax.legend(loc="lower center", fontsize=8.5, markerscale=4)

    fig.suptitle(
        "混沌计算里什么不可信、什么可信：同一组解，左边报废，右边完好", fontsize=12
    )
    return _finish(fig, output, "fig10_trajectory_vs_statistics")


def _thesis_data() -> dict[str, NDArray]:
    u0 = on_attractor(CLASSICAL)
    settings = [("DOP853", 1e-6), ("DOP853", 1e-9), ("DOP853", 1e-13)]
    truth = reference_trajectory(CLASSICAL, u0, 60.0, dt_out=0.005)
    out: dict[str, NDArray] = {"t_short": truth.t}
    t_div = []
    for k, (method, rtol) in enumerate(settings):
        short = solve_adaptive(
            CLASSICAL, u0, 60.0, method=method, rtol=rtol, atol=rtol * 1e-3, dt_out=0.005
        )
        long = solve_adaptive(
            CLASSICAL, u0, 3000.0, method=method, rtol=rtol, atol=rtol * 1e-3, dt_out=0.002
        )
        lmap = lorenz_map(long)
        out[f"x{k}"] = short.x
        out[f"zn{k}"] = lmap.z_n
        out[f"znext{k}"] = lmap.z_next
        t_div.append(divergence_time(truth.t, separation(truth, short)))
    out["t_div"] = np.array(t_div)
    return out


# ---------------------------------------------------------------------------
# 11 / 12 — the rest of the statistical case
# ---------------------------------------------------------------------------


def fig11_invariant_density(output: Path) -> Path:
    data = _cached("density", _density_data)
    labels = ["DOP853 rtol=1e-6", "DOP853 rtol=1e-9", "DOP853 rtol=1e-13"]
    colours = [C_LOOSE, C_MID, C_REF]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))

    ax = _style(axes[0])
    for k, (label, colour) in enumerate(zip(labels, colours, strict=True)):
        ax.plot(data["centres"], data[f"density{k}"], color=colour, lw=1.4, label=label)
    ax.set_xlabel("z")
    ax.set_ylabel("不变密度 p(z)")
    ax.set_title("z 的不变密度（T = 3000）", fontsize=10)
    ax.legend(fontsize=8.5)

    ax = _style(axes[1])
    ax.plot(data["sec_x"], data["sec_y"], ".", color=C_REF, ms=1.0, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Poincaré 截面 z = ρ−1 = {CLASSICAL.rho - 1:.0f}（上行穿越）", fontsize=10)

    tv = float(data["max_total_variation"])
    fig.suptitle(
        f"统计量对容差不敏感：三条密度的最大总变差距离 {tv:.4f}", fontsize=11
    )
    return _finish(fig, output, "fig11_invariant_density")


def _density_data() -> dict[str, NDArray]:
    u0 = on_attractor(CLASSICAL)
    settings = [("DOP853", 1e-6), ("DOP853", 1e-9), ("DOP853", 1e-13)]
    span = (0.0, 52.0)
    bins = 130
    out: dict[str, NDArray] = {}
    densities = []
    for k, (method, rtol) in enumerate(settings):
        traj = solve_adaptive(
            CLASSICAL, u0, 3000.0, method=method, rtol=rtol, atol=rtol * 1e-3, dt_out=0.002
        )
        centres, density = invariant_density(traj.z, bins=bins, value_range=span)
        out["centres"] = centres
        out[f"density{k}"] = density
        densities.append(density)
        if k == len(settings) - 1:
            section = poincare_section(traj, plane_z=CLASSICAL.rho - 1.0)
            out["sec_x"] = section[:, 0]
            out["sec_y"] = section[:, 1]

    width = (span[1] - span[0]) / bins
    worst = max(
        0.5 * np.abs(a - b).sum() * width
        for i, a in enumerate(densities)
        for b in densities[i + 1 :]
    )
    out["max_total_variation"] = np.array(worst)
    return out


def fig12_ftle(output: Path) -> Path:
    """Local stretching varies wildly even though its average is a single number."""
    data = _cached("ftle", _ftle_data)
    points, values = data["points"], data["values"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax = _style(axes[0])
    limit = float(np.abs(values).max())
    sc = ax.scatter(
        points[:, 0], points[:, 2], c=values, s=4.0, cmap="coolwarm", vmin=-limit, vmax=limit
    )
    fig.colorbar(sc, ax=ax, label="有限时间 λ₁（窗口 0.5）")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("局部拉伸率在吸引子上的分布", fontsize=10)

    ax = _style(axes[1])
    ax.hist(values, bins=70, color=C_REF, alpha=0.8)
    ax.axvline(values.mean(), color=C_LOOSE, lw=1.8)
    ax.text(
        values.mean() + 0.2,
        ax.get_ylim()[1] * 0.85,
        f"均值 = {values.mean():.3f}\n(= 全局 λ₁)",
        fontsize=9,
        color=C_LOOSE,
    )
    ax.axvline(0.0, color="k", lw=0.9, ls="--")
    ax.set_xlabel("有限时间 λ₁")
    ax.set_ylabel("计数")
    ax.set_title(
        f"标准差 {values.std():.2f}，{(values < 0).mean() * 100:.0f}% 的窗口局部收缩",
        fontsize=10,
    )

    fig.suptitle("「局部 Lyapunov 指数」：它是流的性质，步长改变不了它", fontsize=11)
    return _finish(fig, output, "fig12_ftle")


def _ftle_data() -> dict[str, NDArray]:
    ftle = finite_time_exponents(CLASSICAL, total_time=1500.0, window=0.5)
    return {"points": ftle.points, "values": ftle.values}


# ---------------------------------------------------------------------------
# 13-16 — the phase-portrait section
# ---------------------------------------------------------------------------


def _mark_fixed_points(ax: plt.Axes, p: LorenzParams) -> None:
    """Scatter the fixed points: filled if stable, hollow if not."""
    for es in all_eigen_structures(p):
        stable = es.is_stable
        ax.scatter(
            *es.point,
            s=46,
            facecolor=C_MID if stable else "none",
            edgecolor=C_MID if stable else C_LOOSE,
            linewidths=1.6,
            depthshade=False,
            zorder=6,
        )


def _portrait_axes(ax: plt.Axes, title: str) -> plt.Axes:
    ax.set_xlabel("x", labelpad=-8)
    ax.set_ylabel("y", labelpad=-8)
    ax.set_zlabel("z", labelpad=-8)
    ax.tick_params(labelsize=6.5, pad=-1)
    ax.set_title(title, fontsize=10)
    return ax


def fig13_phase_portraits(output: Path) -> Path:
    """Four regimes in one plate: what the flow does as rho crosses its thresholds."""
    fig = plt.figure(figsize=(11.0, 8.4))

    panels = [
        (
            0.5,
            "ρ = 0.5 < 1：原点全局稳定",
            [(np.array([12.0, 10.0, 22.0]), C_REF), (np.array([-9.0, -14.0, 5.0]), C_LOOSE)],
            8.0,
        ),
        (
            10.0,
            "ρ = 10：C± 稳定，衰减螺旋",
            [(np.array([1.0, 1.0, 1.0]), C_REF), (np.array([-1.0, -1.0, 1.0]), C_LOOSE)],
            30.0,
        ),
        (
            24.5,
            f"ρ = 24.5 < ρ_H={hopf_rho(CLASSICAL):.2f}：双稳\n蓝 = 混沌吸引子，绿 = 落回 C+",
            None,  # built below: one orbit to C+, one on the chaotic set
            70.0,
        ),
        (
            28.0,
            "ρ = 28 > ρ_H：三个不动点全不稳定",
            None,
            60.0,
        ),
    ]

    for k, (rho, title, starts, t_end) in enumerate(panels, start=1):
        p = LorenzParams(rho=rho)
        ax = _portrait_axes(fig.add_subplot(2, 2, k, projection="3d"), title)

        if starts is None and rho == 24.5:
            # Bistability: C+ is still linearly stable here, yet a chaotic
            # attractor already coexists with it. Which one a trajectory ends on
            # is a question about basins, not about rho. The converging orbit is
            # drawn last and thicker, or the chaotic one buries it.
            near_c_plus = all_eigen_structures(p)[1].point + np.array([4.0, 0.0, 0.0])
            starts = [(np.array([1.0, 1.0, 1.0]), C_REF), (near_c_plus, C_MID)]
        elif starts is None:
            starts = [(on_attractor(p), C_REF)]

        for j, (u0, colour) in enumerate(starts):
            traj = regime_trajectory(rho, u0, t_end)
            last = j == len(starts) - 1
            ax.plot(
                traj.x,
                traj.y,
                traj.z,
                color=colour,
                lw=1.4 if (rho == 24.5 and last) else 0.45,
                alpha=0.4 if (rho == 24.5 and not last) else 0.9,
            )
            ax.scatter(*u0, color=colour, s=14, marker="^", depthshade=False, zorder=6)
        _mark_fixed_points(ax, p)

    fig.suptitle(
        "四个 ρ 下的相图：▲ 起点，实心 = 稳定不动点，空心 = 不稳定（σ=10, β=8/3）",
        fontsize=11,
    )
    return _finish(fig, output, "fig13_phase_portraits")


def fig14_poincare(output: Path) -> Path:
    """The section turns the 3D flow into a 1D map; a cobweb walks the map."""
    data = _cached("section", _section_data)
    plane_z = CLASSICAL.rho - 1.0
    x_n, x_next = data["map_n"], data["map_next"]
    same_wing = np.sign(x_n) == np.sign(x_next)
    cutoff = wing_switch_threshold(x_n, x_next)
    lo = float(min(x_n.min(), x_next.min()))
    hi = float(max(x_n.max(), x_next.max()))
    branches = ((same_wing, C_MID, "留在同一叶"), (~same_wing, C_ACCENT, "换到另一叶"))

    fig = plt.figure(figsize=(15.0, 5.4))
    grid = fig.add_gridspec(1, 3)

    ax = _portrait_axes(
        fig.add_subplot(grid[0], projection="3d"),
        f"截面 z = ρ−1 = {plane_z:.0f}，按下一步是否换叶着色",
    )
    ax.plot(data["orbit_x"], data["orbit_y"], data["orbit_z"], color=C_MUTED, lw=0.3, alpha=0.6)
    xs, ys, zs = plane_patch(
        np.array([0.0, 0.0, plane_z]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        26.0,
    )
    ax.plot_surface(xs, ys, zs, color=C_MUTED, alpha=0.12, shade=False)
    # Each wing crosses the plane along one thin curve; the cutoff splits each
    # curve into an inner part that stays and an outer part that switches.
    for mask, colour, _ in branches:
        ax.scatter(
            data["sec_x"][:-1][mask],
            data["sec_y"][:-1][mask],
            np.full(int(mask.sum()), plane_z),
            color=colour,
            s=1.5,
            depthshade=False,
        )

    def draw_map(ax: plt.Axes, alpha: float, label: bool) -> None:
        for mask, colour, name in branches:
            ax.plot(
                x_n[mask],
                x_next[mask],
                ".",
                color=colour,
                ms=1.3,
                alpha=alpha,
                label=name if label else None,
            )
        ax.plot([lo, hi], [lo, hi], color="k", ls=":", lw=0.9)
        for side in (-1.0, 1.0):
            ax.axvline(side * cutoff, color=C_ACCENT, ls="--", lw=0.9)
            ax.axvspan(side * cutoff, side * hi * 1.05, color=C_ACCENT, alpha=0.06, lw=0)
        ax.set_xlabel("第 n 次穿越的 x")
        ax.set_ylabel("第 n+1 次穿越的 x")

    ax = _style(fig.add_subplot(grid[1]))
    draw_map(ax, 0.55, label=True)
    ax.plot([], [], color=C_ACCENT, ls="--", lw=0.9, label=f"门槛 |x| = {cutoff:.2f}")
    ax.set_title(f"一维回归映射：四段，{same_wing.mean() * 100:.0f}% 的穿越不换叶", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0.16, 0.98), fontsize=8, markerscale=4)

    # Cobweb: vertical to the map reads x_{n+1}, horizontal back to the diagonal
    # makes it the next x_n. These five steps from crossing 3 contain two stays,
    # a switch landing just past the cutoff, the far throw that follows, and a
    # far switch that lands near the other wing's centre.
    ax = _style(fig.add_subplot(grid[2]))
    draw_map(ax, 0.18, label=False)
    start, steps = 3, 5
    walk = np.concatenate([x_n[start : start + 1], x_next[start : start + steps]])
    for k in range(steps):
        a, b = walk[k], walk[k + 1]
        colour = C_MID if np.sign(a) == np.sign(b) else C_ACCENT
        ax.annotate(
            "",
            xy=(a, b),
            xytext=(a, a),
            arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.3, "mutation_scale": 9},
        )
        ax.plot([a, b], [b, b], color=colour, lw=1.3)
    ax.plot(walk, walk, "o", color="k", ms=3.5, zorder=5)
    for k, v in enumerate(walk):
        # Step numbers only, below-right of the diagonal where the panel is empty;
        # the values are listed in the report's walk-through table.
        ax.annotate(str(k), (v, v), xytext=(6, -12), textcoords="offset points", fontsize=9)
    ax.set_title("蛛网：竖到映射读下一步，横回 y=x 当新起点", fontsize=10)

    fig.suptitle(
        "Poincaré 截面：三维流 → 截面上两条细线 → 一条一维映射",
        fontsize=11,
    )
    return _finish(fig, output, "fig14_poincare")


def _section_data() -> dict[str, NDArray]:
    u0 = on_attractor(CLASSICAL)
    short = reference_trajectory(CLASSICAL, u0, 60.0, dt_out=0.002)
    long = reference_trajectory(CLASSICAL, u0, 3000.0, dt_out=0.002)
    section = poincare_section(long, plane_z=CLASSICAL.rho - 1.0)
    map_n, map_next = section_return_map(section, coord=0)
    return {
        "orbit_x": short.x,
        "orbit_y": short.y,
        "orbit_z": short.z,
        "sec_x": section[:, 0],
        "sec_y": section[:, 1],
        "map_n": map_n,
        "map_next": map_next,
    }


#: A stable periodic orbit of the Lorenz system: one loop round each wing, so
#: the rising section z = rho - 1 is crossed at two points, one per wing.
PERIODIC_RHO = 350.0


def fig19_section_types(output: Path) -> Path:
    """What a Poincare section shows for a periodic, a quasi-periodic and a chaotic orbit."""
    periodic = _cached("section_periodic", _periodic_section_data)
    chaotic = _cached("section", _section_data)
    golden_label = f"{GOLDEN_ROTATION:.3f}"
    columns = [
        ("周期：Lorenz ρ = 350", "截面上收拢到 2 个点"),
        ("周期：环面，ω = 2/5", "5 圈闭合 → 5 个点"),
        (f"准周期：环面，ω = (√5−1)/2 ≈ {golden_label}", "永不闭合 → 点铺满一条闭曲线"),
        ("混沌：Lorenz ρ = 28", "点落在两条细线上，次序乱跳"),
    ]

    fig = plt.figure(figsize=(17.0, 8.4))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0])
    tops = [fig.add_subplot(grid[0, k], projection="3d") for k in range(4)]
    bottoms = [_style(fig.add_subplot(grid[1, k])) for k in range(4)]
    for ax, (title, _) in zip(tops, columns, strict=True):
        _portrait_axes(ax, title)
    for ax, (_, title) in zip(bottoms, columns, strict=True):
        ax.set_title(title, fontsize=10)

    # Periodic Lorenz: the transient is kept, so the bottom panel shows the
    # crossings closing in on the two points of the orbit.
    plane_z = PERIODIC_RHO - 1.0
    ax = tops[0]
    ax.plot(periodic["orbit_x"], periodic["orbit_y"], periodic["orbit_z"], color=C_REF, lw=0.4)
    xs, ys, zs = plane_patch(
        np.array([0.0, 0.0, plane_z]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 70.0
    )
    ax.plot_surface(xs, ys, zs, color=C_MUTED, alpha=0.15, shade=False)
    ax.scatter(
        periodic["sec_x"][-2:], periodic["sec_y"][-2:], [plane_z] * 2,
        color=C_LOOSE, s=30, depthshade=False,
    )
    ax = bottoms[0]
    # Zoom on the right-wing point: the crossings march in along a line,
    # each one nearer than the last; the left-wing point is its mirror image.
    right = periodic["sec_x"] > 0
    sec_x, sec_y = periodic["sec_x"][right], periodic["sec_y"][right]
    index = np.flatnonzero(right)
    order = ax.scatter(sec_x, sec_y, c=index, cmap="viridis_r", s=14, zorder=3)
    ax.plot(sec_x[-1:], sec_y[-1:], "x", color=C_LOOSE, ms=10, mew=2, zorder=4)
    fig.colorbar(order, ax=ax, label="第几次穿越", fraction=0.05)
    ax.set_xlim(sec_x[-1] - 4.0, sec_x[-1] + 2.0)
    ax.set_ylim(sec_y[-1] - 12.0, sec_y[-1] + 4.0)
    ax.set_title(columns[0][1] + "（放大右叶那一个）", fontsize=10)
    ax.set_xlabel(f"x（截面 z = {plane_z:.0f}）")
    ax.set_ylabel("y")

    # Torus: the same surface twice; only the frequency ratio differs.
    for ax_top, ax_bottom, omega, laps, n_returns in (
        (tops[1], bottoms[1], 2.0 / 5.0, 5, 5),
        (tops[2], bottoms[2], GOLDEN_ROTATION, 30, 400),
    ):
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 60), np.linspace(0, 2 * np.pi, 30))
        ring = 3.0 + np.cos(v)
        ax_top.plot_surface(
            ring * np.cos(u), ring * np.sin(u), np.sin(v), color=C_MUTED, alpha=0.08, shade=False
        )
        orbit = torus_orbit(omega, np.linspace(0.0, 2 * np.pi * laps, 400 * laps))
        ax_top.plot(*orbit.T, color=C_REF, lw=0.7)
        half_x, half_z = np.meshgrid([1.6, 4.4], [-1.4, 1.4])
        ax_top.plot_surface(
            half_x, np.zeros_like(half_x), half_z, color=C_LOOSE, alpha=0.18, shade=False
        )
        points = torus_section(omega, n_returns)
        ax_top.scatter(points[:, 0], 0.0, points[:, 1], color=C_LOOSE, s=8, depthshade=False)
        ax_top.set_zlim(-3, 3)

        circle = np.linspace(0, 2 * np.pi, 200)
        ax_bottom.plot(3.0 + np.cos(circle), np.sin(circle), color=C_MUTED, lw=0.6, ls=":")
        ax_bottom.plot(points[:, 0], points[:, 1], "o", color=C_LOOSE, ms=3 if n_returns > 5 else 6)
        for k, (px, pz) in enumerate(points[:6]):
            ax_bottom.annotate(
                str(k), (px, pz), xytext=(3.0 + 1.22 * (px - 3.0), 1.22 * pz),
                ha="center", va="center", fontsize=9,
            )
        ax_bottom.set_aspect("equal")
        ax_bottom.set_xlim(1.6, 4.4)
        ax_bottom.set_ylim(-1.4, 1.4)
        ax_bottom.set_xlabel("x（截面 y = 0 半平面）")
        ax_bottom.set_ylabel("z")

    # Chaotic Lorenz: the same section as fig14, drawn flat.
    plane_z = CLASSICAL.rho - 1.0
    ax = tops[3]
    ax.plot(
        chaotic["orbit_x"], chaotic["orbit_y"], chaotic["orbit_z"], color=C_REF, lw=0.3, alpha=0.6
    )
    xs, ys, zs = plane_patch(
        np.array([0.0, 0.0, plane_z]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 26.0
    )
    ax.plot_surface(xs, ys, zs, color=C_MUTED, alpha=0.15, shade=False)
    ax.scatter(
        chaotic["sec_x"], chaotic["sec_y"], np.full(chaotic["sec_x"].size, plane_z),
        color=C_LOOSE, s=1.0, depthshade=False,
    )
    ax = bottoms[3]
    ax.plot(chaotic["sec_x"], chaotic["sec_y"], ".", color=C_LOOSE, ms=1.2, alpha=0.5)
    # The six crossings fig14's cobweb walks (crossings 3-8), renumbered from 0.
    start = 3
    offsets = [(-16, 4), (-16, -4), (-16, -12), (8, -4), (8, -4), (8, -4)]
    for k, offset in enumerate(offsets):
        px, py = chaotic["sec_x"][start + k], chaotic["sec_y"][start + k]
        ax.plot(px, py, "o", color="k", ms=3)
        ax.annotate(str(k), (px, py), xytext=offset, textcoords="offset points", fontsize=9)
    ax.set_xlabel(f"x（截面 z = {plane_z:.0f}）")
    ax.set_ylabel("y")

    fig.suptitle(
        "Poincaré 截面：每绕一圈记一个点。上：轨道与截面；下：截面上的点（数字 = 第几次穿越）",
        fontsize=11,
    )
    return _finish(fig, output, "fig19_section_types")


def _periodic_section_data() -> dict[str, NDArray]:
    traj = regime_trajectory(PERIODIC_RHO, np.array([1.0, 1.0, 1.0]), 40.0, dt_out=0.0005)
    section = poincare_section(traj, plane_z=PERIODIC_RHO - 1.0)
    return {
        "orbit_x": traj.x,
        "orbit_y": traj.y,
        "orbit_z": traj.z,
        "sec_x": section[:, 0],
        "sec_y": section[:, 1],
    }


def fig15_eigen_structure(output: Path) -> Path:
    """Left: where the eigenvalues go as rho grows. Right: where they point at rho=28."""
    rho_values = np.linspace(0.02, 32.0, 900)
    origin_locus = eigenvalue_locus(rho_values, which="origin")
    c_locus = eigenvalue_locus(rho_values[rho_values > 1.0], which="C+")
    rho_h = hopf_rho(CLASSICAL)

    fig = plt.figure(figsize=(12.8, 5.6))

    ax = _style(fig.add_subplot(1, 2, 1))
    ax.axvline(0.0, color="k", lw=1.0)
    for k in range(3):
        branch = origin_locus.branch(k)
        ax.plot(branch.real, branch.imag, color=C_MUTED, lw=2.6, alpha=0.35, solid_capstyle="butt")
        branch = c_locus.branch(k)
        ax.plot(branch.real, branch.imag, color=C_MUTED, lw=2.6, alpha=0.35)
        # Colour by rho so the direction of travel is readable without arrows.
        scatter = ax.scatter(
            c_locus.branch(k).real,
            c_locus.branch(k).imag,
            c=c_locus.rho,
            cmap="viridis",
            s=3.0,
            zorder=4,
        )
    fig.colorbar(scatter, ax=ax, label="ρ（C± 分支的着色）", pad=0.02)

    for rho_mark, colour, text, text_xy in [
        (1.0, C_MID, "ρ=1：原点的实特征值穿零\n（叉式分岔，C± 诞生）", (-22.0, 4.6)),
        (rho_h, C_LOOSE, f"ρ_H={rho_h:.3f}：复对穿越虚轴\n（亚临界 Hopf）", (2.6, 6.6)),
    ]:
        locus = origin_locus if rho_mark == 1.0 else c_locus
        i = int(np.argmin(np.abs(locus.rho - rho_mark)))
        pts = locus.values[i]
        # The one that is on the imaginary axis at this rho -- for the Hopf
        # point, the member of the pair with positive imaginary part.
        crossing = sorted(pts, key=lambda v: (abs(v.real), -v.imag))[0]
        ax.plot(pts.real, pts.imag, "o", color=colour, ms=6, mfc="none", mew=1.6, zorder=6)
        ax.annotate(
            text,
            xy=(float(crossing.real), float(crossing.imag)),
            xytext=text_xy,
            fontsize=8.5,
            color=colour,
            arrowprops={"arrowstyle": "->", "color": colour, "lw": 1.0},
        )

    for which, marker_label in (("origin", "原点 @ ρ=28"), ("C+", "C± @ ρ=28")):
        pts = eigenvalue_locus(np.array([CLASSICAL.rho]), which=which).values[0]
        ax.plot(
            pts.real,
            pts.imag,
            "*",
            color=C_ACCENT if which == "C+" else "k",
            ms=12,
            zorder=7,
            label=marker_label,
        )
    ax.set_xlabel("Re λ")
    ax.set_ylabel("Im λ")
    ax.set_xlim(-25.0, 14.0)
    ax.set_ylim(-13.5, 13.5)
    ax.set_title("特征值随 ρ 在复平面上的走向（灰=原点，彩=C±）", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8.5)

    ax = _portrait_axes(
        fig.add_subplot(1, 2, 2, projection="3d"), "ρ=28 处的本征方向与不变平面"
    )
    traj = reference_trajectory(CLASSICAL, on_attractor(CLASSICAL), 12.0, dt_out=0.002)
    ax.plot(traj.x, traj.y, traj.z, color=C_MUTED, lw=0.5, alpha=0.75)

    for es in all_eigen_structures(CLASSICAL):
        ax.scatter(*es.point, color="k", s=28, depthshade=False, zorder=6)
        if es.kind == "real":
            for value, direction in zip(es.eigenvalues, es.directions, strict=True):
                unstable = value.real > 0.0
                arrow = (16.0 if unstable else 11.0) * direction
                ax.plot(
                    *np.array([es.point - arrow, es.point + arrow]).T,
                    color=C_LOOSE if unstable else C_REF,
                    lw=2.0 if unstable else 1.4,
                    zorder=5,
                )
        else:
            u_axis, v_axis = _orthonormal_pair(es.spiral_plane)
            xs, ys, zs = plane_patch(es.point, u_axis, v_axis, 8.0)
            ax.plot_surface(xs, ys, zs, color=C_LOOSE, alpha=0.22, shade=False)
            arrow = 8.0 * es.directions[0]
            ax.plot(*np.array([es.point - arrow, es.point + arrow]).T, color=C_REF, lw=1.6)

    origin_eigs = all_eigen_structures(CLASSICAL)[0].eigenvalues.real
    c_eigs = all_eigen_structures(CLASSICAL)[1].eigenvalues
    handles = [
        (C_LOOSE, 2.0, f"原点的不稳定方向  λ={origin_eigs[0]:+.2f}"),
        (C_REF, 1.4, f"原点的稳定方向  λ={origin_eigs[1]:+.2f}, {origin_eigs[2]:+.2f}"),
        (C_REF, 1.6, f"C± 的收缩方向  λ={c_eigs[0].real:+.2f}"),
    ]
    for colour, width, label in handles:
        ax.plot([], [], color=colour, lw=width, label=label)
    ax.plot(
        [],
        [],
        color=C_LOOSE,
        lw=6,
        alpha=0.3,
        label=f"C± 的不变平面  Re λ={c_eigs[1].real:+.3f}, Im λ=±{abs(c_eigs[1].imag):.2f}",
    )
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)

    fig.suptitle(
        "原点是鞍点（一正二负的实特征值）；C± 有一个实收缩方向和一个向外旋的不变平面——"
        "轨道正是被这套结构推来推去",
        fontsize=11,
    )
    return _finish(fig, output, "fig15_eigen_structure")


def _orthonormal_pair(axes: NDArray) -> tuple[NDArray, NDArray]:
    """Gram-Schmidt the two spanning vectors, so the drawn patch is a square."""
    u = axes[0] / np.linalg.norm(axes[0])
    v = axes[1] - (axes[1] @ u) * u
    return u, v / np.linalg.norm(v)


def fig16_orthogonalisation(output: Path) -> Path:
    """Why the Benettin loop re-orthonormalises: without it the frame collapses."""
    data = _cached("frames", _frames_data)

    fig = plt.figure(figsize=(13.0, 4.7))

    ax = _portrait_axes(
        fig.add_subplot(1, 3, 1, projection="3d"), "沿轨道运送的正交标架"
    )
    ax.plot(data["orbit_x"], data["orbit_y"], data["orbit_z"], color=C_MUTED, lw=0.5, alpha=0.7)
    # Only the stretch of frames that lies on the drawn orbit segment.
    on_orbit = data["times"] <= float(data["collapse_t"][-1])
    states, frames = data["states"][on_orbit], data["frames"][on_orbit]
    every = max(1, states.shape[0] // 18)
    colours = (C_LOOSE, C_MID, C_REF)
    for k in range(0, states.shape[0], every):
        base = states[k]
        for j, colour in enumerate(colours):
            tip = base + 5.5 * frames[k][:, j]
            ax.plot(*np.array([base, tip]).T, color=colour, lw=1.1)
    for colour, label in zip(colours, ("e₁", "e₂", "e₃"), strict=True):
        ax.plot([], [], color=colour, lw=1.4, label=label)
    ax.legend(loc="upper left", fontsize=8)

    ax = _style(fig.add_subplot(1, 3, 2))
    pair_colours = (C_REF, C_LOOSE, C_ACCENT)
    for k, (label, colour) in enumerate(
        zip(("v₁∠v₂", "v₁∠v₃", "v₂∠v₃"), pair_colours, strict=True)
    ):
        ax.semilogy(data["collapse_t"], data["angles"][:, k], color=colour, lw=1.1, label=label)
    ax.axhline(np.pi / 2, color=C_MID, lw=1.8, ls="--")
    ax.text(9.0, 5.6, "重正交化后：恒为 90°", fontsize=8.5, color=C_MID)
    ax.set_xlabel("t")
    ax.set_ylabel("切向量之间的夹角 (rad)")
    ax.set_ylim(1e-15, 12.0)
    ax.set_title("不重正交化：标架塌成一条线", fontsize=10)
    ax.legend(loc="lower left", fontsize=8.5)

    ax = _style(fig.add_subplot(1, 3, 3))
    for k, colour in enumerate(colours):
        ax.plot(
            data["collapse_t"],
            data["log_norms"][:, k],
            color=colour,
            ls="--",
            lw=1.2,
            label="不正交化：log|v₁|,|v₂|,|v₃|（三条重合）" if k == 0 else None,
        )
    for k, colour in enumerate(colours):
        ax.plot(
            data["times"],
            data["accumulated"][:, k],
            color=colour,
            lw=1.6,
            label=f"Σ log|R_{k + 1}{k + 1}|",
        )
    ax.set_xlim(0.0, float(data["collapse_t"][-1]))
    ax.set_ylim(-470.0, 70.0)
    ax.set_xlabel("t")
    ax.set_ylabel("累积对数伸长")
    naive = data["naive"]
    ax.set_title(
        f"三条虚线斜率全是 λ₁（{naive.min():.2f}…{naive.max():.2f}）\n"
        f"实线斜率才是 {data['spectrum'][0]:.2f} / {data['spectrum'][1]:.3f} / "
        f"{data['spectrum'][2]:.2f}",
        fontsize=9.5,
    )
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(
        "每一步做正交化到底在做什么：不做，三个方向都倒向最不稳定的那个，只剩 λ₁",
        fontsize=11,
    )
    return _finish(fig, output, "fig16_orthogonalisation")


def _frames_data() -> dict[str, NDArray]:
    history = tangent_frames(CLASSICAL, total_time=200.0, renorm_interval=0.5)
    collapse = unrenormalised_tangents(CLASSICAL, total_time=30.0)
    orbit = reference_trajectory(CLASSICAL, history.states[0], 30.0, dt_out=0.002)
    return {
        "times": history.times,
        "states": history.states,
        "frames": history.frames,
        "accumulated": np.cumsum(history.log_stretch, axis=0),
        "spectrum": history.spectrum,
        "collapse_t": collapse.times,
        "angles": collapse.angles,
        "log_norms": collapse.log_norms,
        "naive": collapse.naive_exponents,
        "orbit_x": orbit.x,
        "orbit_y": orbit.y,
        "orbit_z": orbit.z,
    }


# ---------------------------------------------------------------------------
# 17 — the conservative control: where a long single trajectory is trustworthy
# ---------------------------------------------------------------------------

#: Verlet step for panels (a) and (b); RK4 runs at RK4_STAGES times this, so the
#: two methods spend the same number of force evaluations.
KEPLER_H = 0.01
KEPLER_PERIODS = 1000
#: Position error that counts as "the phase is lost" in panel (c).
KEPLER_THRESHOLD = 0.1


def fig17_conservative(output: Path) -> Path:
    """Kepler: Verlet's energy stays bounded, RK4's drifts; both horizons are power laws."""
    data = _cached("conservative", _conservative_data)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9))
    runs = (("verlet", "Störmer–Verlet（辛）", C_MID), ("rk4", "RK4（非辛）", C_LOOSE))
    cost = f"同等代价：h_Verlet = {KEPLER_H:g}，h_RK4 = {RK4_STAGES * KEPLER_H:g}"
    fig.suptitle(f"守恒系统对照：Kepler 问题（偏心率 {DEFAULT_ECCENTRICITY}）", fontsize=12.5)

    ax = _style(axes[0])
    for key, label, colour in runs:
        ax.loglog(data["periods"], data[f"{key}_energy"], color=colour, lw=1.6, label=label)
    cross = float(data["energy_crossover"])
    ax.axvline(cross, color=C_MUTED, ls=":", lw=1.0)
    ax.text(
        cross * 1.15,
        ax.get_ylim()[0] * 3,
        f"{cross:.0f} 周期后 RK4 的\n能量误差超过 Verlet",
        fontsize=8.5,
        color=C_MUTED,
    )
    ax.set_xlabel("时间（轨道周期）")
    ax.set_ylabel("|H − H₀| / |H₀|  （至今最大值）")
    ax.set_title("(a) 能量：有界震荡 vs 线性漂移", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, title=cost, title_fontsize=8)
    ax.text(
        0.97,
        0.04,
        f"Verlet 角动量漂移 {float(data['verlet_angular_drift']):.0e}\n（中心力下精确守恒）",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "fc": "#effaf7", "ec": C_MID, "alpha": 0.95},
    )

    ax = _style(axes[1])
    periods = data["periods"]
    for key, label, colour in runs:
        ax.loglog(periods, data[f"{key}_error"], color=colour, lw=1.6, label=label)
    # Slope guides sit a factor 3 below each curve, anchored at period 10 where
    # RK4 has left its short linear transient.
    guide = periods[(periods >= 3.0) & (periods <= 30.0)]
    for power, key, colour in ((1, "verlet", C_MID), (2, "rk4", C_LOOSE)):
        anchor = float(np.interp(10.0, periods, data[f"{key}_error"])) / 3.0
        line = anchor * (guide / 10.0) ** power
        ax.loglog(guide, line, color=colour, ls="--", lw=1.0, alpha=0.8)
        ax.text(guide[0] / 1.15, line[0], f"∝ t{'²' if power == 2 else ''}", fontsize=9.5,
                color=colour, ha="right", va="center")
    ax.axhline(2.0, color=C_MUTED, ls=":", lw=1.0)
    ax.text(periods[0], 2.3, "轨道长轴 = 2：相位全丢，轨道形状仍在", fontsize=8, color=C_MUTED)
    ax.set_ylim(top=8.0)
    ax.set_xlabel("时间（轨道周期）")
    ax.set_ylabel("位置误差 |q − q_exact|  （至今最大值）")
    ax.set_title("(b) 相位误差：多项式增长，不是指数", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)

    ax = _style(axes[2])
    for key, label, colour in runs:
        evals, reach = data[f"{key}_evals"], data[f"{key}_horizon"]
        slope = np.polyfit(np.log(evals), np.log(reach), 1)[0]
        ax.loglog(evals, reach, "o-", color=colour, ms=6, lw=1.4,
                  label=f"{label}  斜率 {slope:.1f}")
    ax.set_xticks(data["verlet_evals"])
    ax.set_xticklabels([f"{e:.0f}" for e in data["verlet_evals"]])
    ax.minorticks_off()
    ax.set_xlabel("每单位时间的力计算次数（代价）")
    ax.set_ylabel(f"位置误差到 {KEPLER_THRESHOLD:g} 的时间（周期）")
    ax.set_title("(c) 代价换来的时间是幂律", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.text(
        0.97,
        0.04,
        "同一个问题在 Lorenz 上\n只有对数关系（图 9）",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "fc": "#fff6ef", "ec": C_LOOSE, "alpha": 0.95},
    )
    return _finish(fig, output, "fig17_conservative")


def _conservative_data() -> dict[str, NDArray]:
    t_end = KEPLER_PERIODS * KEPLER_PERIOD
    runs = {
        "verlet": kepler_verlet(t_end, KEPLER_H),
        "rk4": kepler_rk4(t_end, RK4_STAGES * KEPLER_H),
    }
    # One sample per period: the envelopes are monotone, so nothing is lost.
    periods = np.arange(1, KEPLER_PERIODS + 1, dtype=float)
    out: dict[str, NDArray] = {"periods": periods}
    for key, traj in runs.items():
        idx = np.searchsorted(traj.t, periods * KEPLER_PERIOD - 1e-9)
        h0 = kepler_energy(traj.q[0], traj.p[0])
        energy = running_max(np.abs(kepler_energy(traj.q, traj.p) - h0) / abs(h0))
        out[f"{key}_energy"] = energy[idx]
        out[f"{key}_error"] = running_max(position_error(traj))[idx]
    ahead = np.flatnonzero(out["rk4_energy"] > out["verlet_energy"])
    out["energy_crossover"] = np.array(periods[ahead[0]] if ahead.size else np.inf)
    momentum = angular_momentum(runs["verlet"].q, runs["verlet"].p)
    out["verlet_angular_drift"] = np.array(np.ptp(momentum))

    # Horizon against cost. Each run is long enough to reach the threshold; the
    # last RK4 point needs ~700 periods.
    verlet_steps = np.array([0.02, 0.01, 0.005, 0.0025])
    out["verlet_evals"] = 1.0 / verlet_steps
    out["verlet_horizon"] = np.array(
        [horizon(kepler_verlet(400 * KEPLER_PERIOD, h), KEPLER_THRESHOLD) for h in verlet_steps]
    ) / KEPLER_PERIOD
    rk4_steps = RK4_STAGES * verlet_steps
    out["rk4_evals"] = RK4_STAGES / rk4_steps
    out["rk4_horizon"] = np.array(
        [horizon(kepler_rk4(1500 * KEPLER_PERIOD, h), KEPLER_THRESHOLD) for h in rk4_steps]
    ) / KEPLER_PERIOD
    return out


# ---------------------------------------------------------------------------
# 18 — what x, y, z are: one convecting layer
# ---------------------------------------------------------------------------


def fig18_convection(output: Path) -> Path:
    """Each Lorenz variable as the spatial pattern it multiplies, then all three at C+."""
    p = CLASSICAL
    c_plus = fixed_points(p)[1]
    fig = plt.figure(figsize=(11.0, 7.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25])

    def roll_axes(ax: plt.Axes, title: str) -> plt.Axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["底（热）", "顶（冷）"], fontsize=8)
        ax.set_title(title, fontsize=10)
        return ax

    # x alone: the stream function, i.e. the rolls themselves.
    f = convection_field(np.array([1.0, 0.0, 0.0]), p)
    ax = roll_axes(fig.add_subplot(grid[0, 0]), "x：对流卷的翻滚速度")
    ax.streamplot(f.x[0], f.z[:, 0], f.u, f.w, color=C_REF, density=0.9, linewidth=0.8)

    # y alone: warm where fluid rises, cool where it sinks.
    f = convection_field(np.array([0.0, 1.0, 0.0]), p)
    ax = roll_axes(fig.add_subplot(grid[0, 1]), "y：上升流与下沉流的温差")
    theta = f.temperature - (1.0 - f.z)
    lim = np.abs(theta).max()
    ax.pcolormesh(f.x, f.z, theta, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")

    # z alone: horizontally uniform, so draw the mean profile it produces.
    ax = _style(fig.add_subplot(grid[0, 2]))
    f = convection_field(c_plus, p)
    profile = f.temperature.mean(axis=1)
    z = f.z[:, 0]
    ax.plot(1.0 - z, z, color=C_MUTED, ls="--", lw=1.0, label="纯传导：线性")
    ax.plot(profile, z, color=C_LOOSE, lw=1.6, label=f"C+ 处，z = {c_plus[2]:.0f}")
    ax.set_xlabel("水平平均温度")
    ax.set_ylabel("高度")
    ax.set_title(f"z：剖面弯离线性 → Nu = 1 + 2z/ρ = {nusselt(c_plus, p):.2f}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    # All three together at C+: steady rolls carrying heat upward.
    ax = roll_axes(fig.add_subplot(grid[1, :]), "")
    mesh = ax.pcolormesh(
        f.x, f.z, f.temperature, cmap="RdBu_r", vmin=0.0, vmax=1.0, shading="auto"
    )
    ax.streamplot(
        f.x[0], f.z[:, 0], f.u, f.w, color="k", density=1.1, linewidth=0.7, arrowsize=0.9
    )
    fig.colorbar(mesh, ax=ax, fraction=0.02, pad=0.01, label="温度（底 = 1，顶 = 0）")
    x, y, zz = c_plus
    ax.set_title(
        f"三者合起来：C+ = ({x:.2f}, {y:.2f}, {zz:.0f}) 是一对定常对流卷"
        f"（σ=10, ρ=28, β=8/3，横向一个波长 = {f.x[0, -1]:.2f} H）",
        fontsize=10,
    )

    return _finish(fig, output, "fig18_convection")


# slides — data the HTML slides replay
# ---------------------------------------------------------------------------

#: Initial-condition offset on the cover, in x.
HERO_DELTA = 1e-5


def slides_data(output: Path) -> Path:
    """The start point shared by every slide demo, the cover's two orbits, and fig15's 3-D panel.

    The cover shows two reference-accuracy trajectories ``HERO_DELTA`` apart,
    up to the first time their separation exceeds the report's threshold.
    ``eigen`` is the rho=28 eigenstructure at each fixed point, for the slide
    that redraws fig15's right panel as a rotatable view; a spiral point's
    plane axes are orthonormalised exactly as in fig15.
    Written as a script that sets ``window.LORENZ_DATA`` so that ``slides.html``
    can load it from ``file://``, where ``fetch`` is blocked.
    """
    u0 = on_attractor(CLASSICAL)
    dt = 0.005
    a = reference_trajectory(CLASSICAL, u0, 30.0, dt_out=dt)
    b = reference_trajectory(CLASSICAL, u0 + np.array([HERO_DELTA, 0.0, 0.0]), 30.0, dt_out=dt)
    delta = separation(a, b)
    n = int(np.flatnonzero(delta > DEFAULT_THRESHOLD)[0]) + 1

    def series(v: NDArray) -> str:
        return "[" + ",".join(f"{x:.4f}" for x in v[:n]) + "]"

    def vec(v: NDArray) -> str:
        return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"

    eigen = []
    for es in all_eigen_structures(CLASSICAL):
        axes = es.directions if es.kind == "real" else np.array(
            [es.directions[0], *_orthonormal_pair(es.spiral_plane)]
        )
        eigen.append(
            f'{{"name":"{es.name}","kind":"{es.kind}","p":{vec(es.point)},'
            f'"re":{vec(es.eigenvalues.real)},"im":{vec(es.eigenvalues.imag)},'
            '"dirs":[' + ",".join(vec(d) for d in axes) + "]}"
        )

    output.mkdir(parents=True, exist_ok=True)
    path = output / "slides_data.js"
    path.write_text(
        "// 由 `lorenz63 figure slides` 生成。u0 = on_attractor(CLASSICAL)；hero 是封面的两条轨道："
        f"DOP853 rtol=1e-13，初值在 x 上差 {HERO_DELTA:g}，"
        f"到 |Δ| 首次超过 {DEFAULT_THRESHOLD:g} 为止；"
        "eigen 是 ρ=28 各不动点的特征值与方向（spiral 的后两个方向是不变平面的正交轴）。\n"
        "window.LORENZ_DATA = {"
        '"u0":[' + ",".join(repr(float(x)) for x in u0) + "],"
        f'"hero":{{"dt":{dt},"delta":{HERO_DELTA},'
        f'"ax":{series(a.x)},"az":{series(a.z)},"bx":{series(b.x)},"bz":{series(b.z)},'
        '"logd":[' + ",".join(f"{x:.3f}" for x in np.log10(delta[:n])) + "]},"
        '"eigen":[' + ",".join(eigen) + "]};\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------

FIGURES: dict[str, Callable[[Path], Path]] = {
    "fig01": fig01_paradox,
    "fig02": fig02_attractor,
    "fig03": fig03_bifurcation,
    "fig04": fig04_fixed_point_stability,
    "fig05": fig05_volume,
    "fig06": fig06_stability,
    "fig07": fig07_ensemble,
    "fig08": fig08_lyapunov,
    "fig09": fig09_tolerance,
    "fig10": fig10_trajectory_vs_statistics,
    "fig11": fig11_invariant_density,
    "fig12": fig12_ftle,
    "fig13": fig13_phase_portraits,
    "fig14": fig14_poincare,
    "fig15": fig15_eigen_structure,
    "fig16": fig16_orthogonalisation,
    "fig17": fig17_conservative,
    "fig18": fig18_convection,
    "fig19": fig19_section_types,
    "slides": slides_data,
}
