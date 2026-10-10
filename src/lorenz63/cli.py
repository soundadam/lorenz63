"""Console script: ``lorenz63 info`` / ``lorenz63 figure all``."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lorenz63 import __version__
from lorenz63.lyapunov import benettin_spectrum
from lorenz63.portrait import all_eigen_structures
from lorenz63.stability import RK4_REAL_AXIS_LIMIT, max_stable_step, spectral_extent
from lorenz63.system import RHO_HOMOCLINIC, LorenzParams, divergence, hopf_rho

DEFAULT_OUTPUT = Path("report/figures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lorenz63",
        description="Lorenz (1963): reproducible numerics for the course report.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    figure = sub.add_parser("figure", help="render one figure, or all of them")
    figure.add_argument("name", help="figure id such as fig09, or 'all'")
    figure.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output directory (default: {DEFAULT_OUTPUT})",
    )

    info = sub.add_parser("info", help="print the numbers the report quotes")
    info.add_argument(
        "--total-time",
        type=float,
        default=2000.0,
        help="integration time for the Lyapunov spectrum (default: 2000)",
    )
    return parser


def _print_info(total_time: float) -> None:
    p = LorenzParams()
    print("lorenz63 — Lorenz (1963) 系统关键数值")
    print(f"version {__version__}")
    print()
    print(f"  参数            sigma={p.sigma}  rho={p.rho}  beta={p.beta:.10f}")
    print()

    print("闭式量（纯代数，无积分）")
    s = np.sqrt(p.beta * (p.rho - 1.0))
    print(f"  相空间散度 tr J        {divergence(p):.10f}   （常数）")
    print(f"  不动点 C±              (±{s:.6f}, ±{s:.6f}, {p.rho - 1:.1f})")
    print("  叉式分岔               rho = 1")
    print(f"  同宿爆炸               rho ≈ {RHO_HOMOCLINIC}   （文献值，非闭式）")
    print(f"  亚临界 Hopf            rho_H = {hopf_rho(p):.10f}")
    print()

    print("不动点处的 Jacobian 特征值（线性稳定性，纯代数）")
    for es in all_eigen_structures(p):
        verdict = "稳定" if es.is_stable else f"不稳定（{es.n_unstable} 个增长方向）"
        print(f"  {es.name:<8s} {verdict}")
        for value in es.eigenvalues:
            if abs(value.imag) < 1e-12:
                print(f"      lambda = {value.real:+12.6f}")
            else:
                print(f"      lambda = {value.real:+12.6f} {value.imag:+.6f}i   （旋转）")
    print("  原点是鞍点：一个正实特征值把轨道推出去，非线性再把它折回来。")
    print()

    print(f"Lyapunov 谱（Benettin，T={total_time:g}）")
    t0 = time.perf_counter()
    res = benettin_spectrum(p, total_time=total_time)
    elapsed = time.perf_counter() - t0
    for k, value in enumerate(res.spectrum, start=1):
        print(f"  lambda_{k}               {value:+.6f}")
    print(f"  Σ lambda_i             {res.total:+.9f}")
    print(f"  tr J（精确）           {divergence(p):+.9f}")
    print(f"  偏差                   {abs(res.total - divergence(p)):.3e}   ← 自洽检验")
    print(f"  Kaplan–Yorke 维数      {res.kaplan_yorke:.6f}")
    print(f"  倍增时间 ln2/lambda_1  {res.doubling_time:.6f}")
    print(f"  双精度预报极限         {res.predictability_horizon(1e-16):.4f}  时间单位")
    print(f"  （用时 {elapsed:.1f}s）")
    print()

    print("数值稳定性（RK4）")
    extent = spectral_extent(p, t_end=150.0)
    h_max = max_stable_step(p, t_end=150.0)
    print(f"  max |lambda_J|         {extent['max_abs']:.4f}")
    print(f"  min Re lambda_J        {extent['min_real']:.4f}")
    print(f"  max Re lambda_J        {extent['max_real']:+.4f}")
    print(f"  实部为正的比例         {extent['fraction_expanding'] * 100:.1f}%   ← 物理增长方向")
    print(f"  稳定步长上限 h         {h_max:.6f}   （仅约束衰减模）")
    crude = RK4_REAL_AXIS_LIMIT / abs(extent["min_real"])
    print(f"  粗估 {RK4_REAL_AXIS_LIMIT:.3f}/|min Re|    {crude:.6f}")
    print()
    print("  结论：实用步长 h≈0.01 比稳定性上限小一个数量级——")
    print("        Lorenz 上的瓶颈是精度，不是稳定性。")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        _print_info(args.total_time)
        return 0

    if args.command == "figure":
        # Imported lazily: pulling in matplotlib costs ~1s that `info` does not need.
        from lorenz63.figures import FIGURES

        output = Path(args.output)
        if args.name == "all":
            selected = list(FIGURES.items())
        elif args.name in FIGURES:
            selected = [(args.name, FIGURES[args.name])]
        else:
            parser.error(f"unknown figure {args.name!r}; choose from {', '.join(FIGURES)} or 'all'")
            return 2

        for name, render in selected:
            t0 = time.perf_counter()
            path = render(output)
            print(f"  {name}  {time.perf_counter() - t0:6.1f}s  ->  {path}")
        print(f"wrote {len(selected)} figure(s) to {output}")
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
