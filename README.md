# lorenz63

Lorenz (1963) 系统的可复现数值工程，为课程汇报《**从对流方程到可信的数值解**》提供全部计算与图。

报告正文：[`report/report.md`](report/report.md)

汇报幻灯片：[`report/slides.html`](report/slides.html)，浏览器直接打开，按 N 显示讲稿。

在线版：<https://soundadam.github.io/lorenz63/>。`report/` 推到 main 后由 `.github/workflows/pages.yml` 原样发布，首页是幻灯片，打印稿在 `mindmap.html`、`cheatsheet.html`。

讲台上用的打印稿：[`report/mindmap.html`](report/mindmap.html)（A4 横向一页）与 [`report/cheatsheet.html`](report/cheatsheet.html)（A4 纵向，左栏板书与心算、右栏讲稿），浏览器打开后打印。

论点一句话：**判断数值结果可信的标准随系统类型而变。守恒系统靠不变量验证；混沌系统的单条轨道不可信，统计量可信。**

Lorenz 是这个论点的理想载体：它没有守恒量可查，可预报时间被最大 Lyapunov 指数锁死，但它的 Lyapunov 谱、不变密度和 Lorenz 映射对积分容差不敏感。具体数字由 `lorenz63 info` 打印，报告正文引用它们。

## 安装

需要 Python 3.11+（仓库用 `.python-version` 钉在 3.12）。依赖只有 numpy / scipy / matplotlib。

```bash
uv sync --all-groups
```

## 使用

```bash
# 打印报告引用的全部数字（Lyapunov 谱、分岔阈值、稳定步长上限……）
uv run lorenz63 info

# 重建 report/figures/ 下全部图（耗时见报告附录 A）
uv run lorenz63 figure all

# 单张
uv run lorenz63 figure fig09

# 测试与 lint
uv run pytest
uv run ruff check src tests
```

图默认写到 `report/figures/`（**已提交进 git**——它们是交付物，不是临时产物）。昂贵的中间结果缓存在 `.cache/`（已忽略）；删掉重跑应得到相同的图，所有随机量都由固定 seed 生成。

## 目录

```
src/lorenz63/
  system.py         向量场、Jacobian、不动点、分岔阈值（全闭式）
  convection.py     x、y、z 还原成对流卷的流场与温度场；Nusselt 数
  integrate.py      自实现定步长 RK4；solve_ivp 包装，暴露内部步长 h(t)
  lyapunov.py       Benettin 全谱、Kaplan–Yorke 维数、有限时间指数、正交化对照
  predictability.py 可分辨时间、容差扫描、系综误差增长      ← 核心实验
  statistics.py     Lorenz 映射、Poincaré 截面、不变密度    ← 核心证据
  bifurcation.py    ρ 扫描；C± 特征值分支与复平面轨迹
  portrait.py       相图：分区轨道、不动点本征方向与不变平面
  stability.py      RK4 稳定域与吸引子上的 Jacobian 谱
  efficiency.py     工作量对精度：非刚性时阶数的回报，刚性时步长被稳定域卡住
  hamiltonian.py    守恒对照：Kepler 问题上的 RK4 与 Störmer–Verlet
  figures.py        报告的图，每张一个函数
  cli.py
report/
  report.md         报告正文
  slides.html       汇报幻灯片：每页正文与讲稿（引用 figures/ 里的图）
  mindmap.html      讲述思维导图：六个板块与黑板右栏，A4 横向
  cheatsheet.html   板书小抄：每段写什么、心算、讲稿，A4 纵向
  slides/           幻灯片的样式与脚本
    slides.css
    dop853_coefficients.js  DOP853 系数表（照 scipy）
    lorenz.js       向量场与 DOP853，各演示共用
    hero.js         封面：回放两条相差 1e-5 的轨道
    paradox.js      开场悖论：换 rtol，看在哪个 t 与参考解脱钩
    roll.js         现场演示：拖 ρ 看对流卷
    split.js        现场演示：缩小初值差
    eigen3d.js      备用「本征结构」：可旋转的本征方向与不变平面
    deck.js         翻页、讲稿、计时，进出页时开关演示
  figures/          png 图；slides_data.js 是幻灯片用的数据（`lorenz63 figure slides`）
tests/
```

## 两个作为测试存在的论点

本仓库的断言不只写在报告里，也写成了测试：

**`test_lyapunov.py::test_spectrum_sums_to_the_divergence`** — 对任何流 `Σλᵢ = ⟨tr J⟩`，而 Lorenz 的 `tr J` 是常数 `−(σ+1+β)`。计算出的谱必须精确加和到这个常数，无需拟合。它守的是变分方程、积分精度与 QR 记账；三个指数怎么分这个和，由报告 5.2 节独立的系综斜率核对。

**`test_predictability.py::test_divergence_time_grows_linearly_in_log_inverse_tolerance`** — 可分辨时间对 `ln(1/rtol)` 呈直线，斜率应为 `1/λ₁`。这直接检验报告的核心论点：精度只换来对数的预报时间。

慢测试标了 `slow` 标记但默认运行；全套不到一分钟。

## 待补的证据

报告里下面几处断言还没有图或测试支撑：

- [ ] **统计判据可证伪**（§6.3）：同一个求解器做两次独立长跑，量出采样噪声底；再拿大步长 Euler 或 ρ 的微扰当反例，看统计量能不能把它们检出来
- [ ] **短窗口收敛阶**（§7）：RK4 在 T=1/5/15/30 上的误差–步长图，O(h⁴) 直线随 T 增大逐渐饱和
- [ ] **影随数值实验**（§6.1）：用多重打靶加 Newton 迭代，把 rtol=1e-6 的伪轨道修正成一条真轨道，测影随距离

## 关系

与同一 workspace 下的 [`compacoustics`](../compacoustics) 是独立仓库：那个包实现李太宝《计算声学》的算法，与本仓库无代码依赖。两者的交集是数值积分的结构保持问题——报告第 7 节讨论了 FDTD 的蛙跳格式为什么是一个辛积分器。

## 许可

MIT
