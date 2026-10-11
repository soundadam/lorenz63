/* ---------- 开场演示：同一起点，只换 DOP853 的 rtol，看它与参考解（rtol = 1e-13）在哪个 t 脱钩 ---------- */
const paradox = (function () {
  const top = document.getElementById("paradoxX");
  const bot = document.getElementById("paradoxLog");
  const tc = top.getContext("2d"), bc = bot.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  const col = name => css.getPropertyValue(name).trim();
  const NAVY = col("--navy"), ORBIT = col("--orbit"), INK = col("--ink"), INK2 = col("--ink-2"), RULE = col("--rule");
  const MONO = col("--mono"), SANS = col("--sans");
  const T_MAX = 45, DT = 0.01;
  const REF_RTOL = 1e-13, REF_ATOL = 1e-15;         // 与 src/lorenz63/integrate.py 的参考精度相同
  const PER_FRAME = 30;                             // 每帧揭开的样本数：45 个时间单位约 2.5 秒放完
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const pow10 = n => "10" + String(n).split("").map(c => SUP[c]).join("");

  const f = Lorenz.field(Lorenz.RHO);
  const u0 = Lorenz.start();
  function run(rtol, atol) {
    const sm = Lorenz.sampler(f, u0, { dt: DT, tEnd: T_MAX, rtol, atol });
    const u = [u0.slice()];
    let smp;
    while ((smp = sm.next())) u.push(smp[1]);
    return { u, steps: sm.solver.steps };
  }
  let ref = null;                                   // 第一次显示这一页时才算

  /* 两张图共用横轴 t ∈ [0, T_MAX] */
  const M = { l: 190, r: 40, t: 24, b: 70 };
  const X = (c, t) => M.l + t / T_MAX * (c.width - M.l - M.r);
  const XM = 22, LMIN = -14, LMAX = 2;
  const YX = x => M.t + (XM - x) / (2 * XM) * (top.height - M.t - M.b);
  const YL = l => M.t + (LMAX - l) / (LMAX - LMIN) * (bot.height - M.t - M.b);

  let runs = [], cur = null, n = 0, queue = [], raf = null, active = false;
  const slider = document.getElementById("paradoxTol");
  const label = () => { document.getElementById("paradoxTolV").textContent = "1e−" + slider.value; };

  function start(k) {
    if (!ref) ref = run(REF_RTOL, REF_ATOL).u;
    const rtol = Math.pow(10, -k), r = run(rtol, rtol * 1e-3);
    const logd = r.u.map((v, i) => Math.log10(Math.max(1e-300, Math.hypot(v[0] - ref[i][0], v[1] - ref[i][1], v[2] - ref[i][2]))));
    const j = logd.findIndex(l => l > 0);
    runs = runs.filter(q => q.k !== k);
    cur = { k, u: r.u, logd, steps: r.steps, T: j < 0 ? null : j * DT };
    runs.push(cur);
    slider.value = k; label();
    n = 0;
  }
  const finished = () => !cur || n >= cur.u.length - 1;

  function axes(c, Y, ticks, fmt, ylabel) {
    c.clearRect(0, 0, c.canvas.width, c.canvas.height);
    const W = c.canvas.width, H = c.canvas.height;
    c.font = "30px " + MONO; c.textBaseline = "middle"; c.strokeStyle = RULE; c.lineWidth = 1.5; c.fillStyle = INK2;
    ticks.forEach(v => {
      c.beginPath(); c.moveTo(M.l, Y(v)); c.lineTo(W - M.r, Y(v)); c.stroke();
      c.textAlign = "right"; c.fillText(fmt(v), M.l - 14, Y(v));
    });
    c.textBaseline = "top"; c.textAlign = "center";
    for (let tt = 0; tt <= 40; tt += 10) c.fillText(String(tt), X(c.canvas, tt), H - M.b + 12);
    c.font = "30px " + SANS; c.textAlign = "right"; c.fillText("t", W - M.r, H - M.b + 12);
    c.save(); c.translate(40, M.t + (H - M.t - M.b) / 2); c.rotate(-Math.PI / 2); c.textAlign = "center"; c.fillText(ylabel, 0, 0); c.restore();
  }
  function line(c, pts, upto, Y, color, w, alpha) {
    c.strokeStyle = color; c.lineWidth = w; c.globalAlpha = alpha; c.lineJoin = "round";
    c.beginPath();
    for (let i = 0; i <= upto; i++) { const xx = X(c.canvas, i * DT), yy = Y(pts[i]); i ? c.lineTo(xx, yy) : c.moveTo(xx, yy); }
    c.stroke(); c.globalAlpha = 1;
  }

  function draw() {
    axes(tc, YX, [-20, -10, 0, 10, 20], String, "x(t)");
    axes(bc, YL, [-12, -8, -4, 0], l => (l === 0 ? "1" : pow10(l)), "与参考解的距离");
    if (!cur) return;
    // 上图：参考解（藏青、宽）垫底，当前 rtol（朱红、细）压在上面
    line(tc, ref.map(v => v[0]), n, YX, NAVY, 6, 0.45);
    line(tc, cur.u.map(v => v[0]), n, YX, ORBIT, 2.6, 1);
    // 下图：|Δ| = 1 的虚线，旧 rtol 留淡色，当前加粗
    bc.strokeStyle = INK; bc.lineWidth = 2; bc.setLineDash([12, 10]);
    bc.beginPath(); bc.moveTo(M.l, YL(0)); bc.lineTo(bot.width - M.r, YL(0)); bc.stroke(); bc.setLineDash([]);
    runs.forEach(r => {
      const now = r === cur, upto = now ? n : r.logd.length - 1;
      line(bc, r.logd.map(l => Math.max(LMIN, Math.min(LMAX, l))), upto, YL, now ? ORBIT : INK2, now ? 4.5 : 2.5, now ? 1 : 0.4);
      if (r.T !== null && (!now || n * DT >= r.T)) {
        bc.fillStyle = now ? ORBIT : INK2;
        bc.beginPath(); bc.arc(X(bot, r.T), YL(0), now ? 10 : 7, 0, 2 * Math.PI); bc.fill();
        if (!now) { bc.font = "26px " + MONO; bc.textAlign = "center"; bc.textBaseline = "bottom"; bc.fillText("1e−" + r.k, X(bot, r.T), YL(0) - 14); }
      }
    });
    document.getElementById("paradoxSteps").textContent = cur.steps;
    document.getElementById("paradoxT").textContent = cur.T === null ? "> " + T_MAX : cur.T.toFixed(1);
  }

  function frame() {
    if (!finished()) n = Math.min(cur.u.length - 1, n + PER_FRAME);
    else if (queue.length) start(queue.shift());
    draw();
    raf = !finished() || queue.length ? requestAnimationFrame(frame) : null;
  }
  function play() {
    if (reduce) {
      for (;;) { n = cur.u.length - 1; if (!queue.length) break; start(queue.shift()); }
      draw(); return;
    }
    if (raf === null && active) raf = requestAnimationFrame(frame);
  }

  slider.addEventListener("input", () => { label(); queue = []; start(+slider.value); play(); });
  document.getElementById("paradoxSweep").addEventListener("click", () => { runs = []; queue = [6, 9, 12]; start(3); play(); });
  document.getElementById("paradoxClear").addEventListener("click", () => { queue = []; runs = cur ? [cur] : []; draw(); });
  label();
  draw();

  return {
    toggle(on) {
      active = on;
      if (on && !cur) start(+slider.value);
      if (on) play();
      else if (raf !== null) { cancelAnimationFrame(raf); raf = null; }
    }
  };
})();
