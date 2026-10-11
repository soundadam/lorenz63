/* ---------- 演示：同一起点、不同 ε，记录 |Δ| 首次超过 1 的时刻 ---------- */
const split = (function () {
  const phase = document.getElementById("splitPhase");
  const plot = document.getElementById("splitLog");
  const pc = phase.getContext("2d"), lc = plot.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  const col = name => css.getPropertyValue(name).trim();
  const NAVY = col("--navy"), ORBIT = col("--orbit"), INK = col("--ink"), INK2 = col("--ink-2"), RULE = col("--rule");
  const MONO = col("--mono"), SANS = col("--sans");
  const H = 0.005;
  const LAMBDA1 = 0.903764;          // lorenz63 info 的 Benettin 值，只用来画参考斜率
  const T_MAX = 45, AFTER = 4;       // 分开之后再跑 AFTER 个时间单位就停
  const KEEP = 3600;                 // 每个 RK4 步都画，折线才不出棱角
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const f = Lorenz.field(Lorenz.RHO);
  const rk4 = u => Lorenz.rk4(f, u, H);
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const pow10 = n => "10" + String(n).split("").map(c => SUP[c]).join("");

  const a0 = Lorenz.start();

  /* 相空间：固定的斜视角，吸引子底图只画一次 */
  const AZ = 0.62, EL = 0.32, ZC = 25, SC = phase.width / 64;
  function proj(u) {
    const X = u[0] * Math.cos(AZ) + u[1] * Math.sin(AZ);
    const D = u[1] * Math.cos(AZ) - u[0] * Math.sin(AZ);
    const Y = (u[2] - ZC) * Math.cos(EL) - D * Math.sin(EL);
    return [phase.width / 2 + X * SC, phase.height / 2 - Y * SC];
  }
  const backdrop = document.createElement("canvas");
  backdrop.width = phase.width; backdrop.height = phase.height;
  (function () {
    const g = backdrop.getContext("2d");
    g.strokeStyle = INK2; g.globalAlpha = 0.16; g.lineWidth = 1.4;
    let u = a0.slice();
    g.beginPath();
    for (let k = 0; k < 16000; k++) {
      u = rk4(u);
      const p = proj(u);
      k ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]);
    }
    g.stroke();
  })();

  /* 右图坐标：t ∈ [0, 45]，log10|Δ| ∈ [−15, 2] */
  const M = { l: 190, r: 40, t: 30, b: 86 };
  const W = plot.width - M.l - M.r, Hh = plot.height - M.t - M.b;
  const LMIN = -15, LMAX = 2;
  const X = t => M.l + t / T_MAX * W;
  const Y = l => M.t + (LMAX - l) / (LMAX - LMIN) * Hh;

  let runs = [], cur = null, queue = [], a, b, t, ta, tb, raf = null, stepsPerFrame = 16, active = false;
  const slider = document.getElementById("splitEps");
  const label = () => { document.getElementById("splitEpsV").textContent = pow10(-slider.value); };

  function start(k) {
    runs = runs.filter(r => r.k !== k);
    cur = { k, eps: Math.pow(10, -k), pts: [], T: null };
    runs.push(cur);
    slider.value = k; label();
    a = a0.slice(); b = [a0[0] + cur.eps, a0[1], a0[2]];
    t = 0; ta = [a]; tb = [b];
    cur.pts.push([0, Math.log10(dist(a, b))]);
  }
  function finished() { return !cur || t >= T_MAX || (cur.T !== null && t >= cur.T + AFTER); }
  function advance(n) {
    for (let k = 0; k < n && !finished(); k++) {
      a = rk4(a); b = rk4(b); t += H;
      if (cur.T === null && dist(a, b) > 1) cur.T = t;
      ta.push(a); tb.push(b);
    }
    cur.pts.push([t, Math.log10(dist(a, b))]);
    if (ta.length > KEEP) { ta.splice(0, ta.length - KEEP); tb.splice(0, tb.length - KEEP); }
  }

  function drawPhase() {
    pc.clearRect(0, 0, phase.width, phase.height);
    pc.drawImage(backdrop, 0, 0);
    if (!ta) return;
    function trail(pts, color, w, alpha) {
      pc.strokeStyle = color; pc.lineWidth = w; pc.globalAlpha = alpha; pc.lineJoin = "round";
      pc.beginPath();
      pts.forEach((u, k) => { const p = proj(u); k ? pc.lineTo(p[0], p[1]) : pc.moveTo(p[0], p[1]); });
      pc.stroke(); pc.globalAlpha = 1;
    }
    trail(ta, NAVY, 9, 0.35);
    trail(tb, ORBIT, 3, 1);
    const pa = proj(a), pb = proj(b);
    pc.fillStyle = NAVY; pc.beginPath(); pc.arc(pa[0], pa[1], 13, 0, 2 * Math.PI); pc.fill();
    pc.fillStyle = ORBIT; pc.beginPath(); pc.arc(pb[0], pb[1], 7, 0, 2 * Math.PI); pc.fill();
    document.getElementById("splitT").textContent = "t = " + t.toFixed(1);
    document.getElementById("splitD").textContent = "|Δ| = " + dist(a, b).toExponential(1).replace("-", "−");
  }

  function drawPlot() {
    const c = lc;
    c.clearRect(0, 0, plot.width, plot.height);
    c.font = "32px " + MONO; c.textBaseline = "middle";
    // 网格与刻度
    c.strokeStyle = RULE; c.lineWidth = 1.5; c.fillStyle = INK2;
    for (let l = -14; l <= 2; l += 2) {
      c.beginPath(); c.moveTo(M.l, Y(l)); c.lineTo(M.l + W, Y(l)); c.stroke();
      if (l % 4 === 0) { c.textAlign = "right"; c.fillText(l === 0 ? "1" : pow10(l), M.l - 14, Y(l)); }
    }
    c.textBaseline = "top";
    for (let tt = 0; tt <= 40; tt += 10) { c.textAlign = "center"; c.fillText(String(tt), X(tt), M.t + Hh + 14); }
    c.font = "32px " + SANS; c.textAlign = "right"; c.fillText("t", M.l + W, M.t + Hh + 14);
    c.save(); c.translate(44, M.t + Hh / 2); c.rotate(-Math.PI / 2); c.textAlign = "center"; c.fillText("两条轨道的距离 |Δ|", 0, 0); c.restore();
    // |Δ| = 1：分开
    c.strokeStyle = INK; c.lineWidth = 2; c.setLineDash([12, 10]);
    c.beginPath(); c.moveTo(M.l, Y(0)); c.lineTo(M.l + W, Y(0)); c.stroke(); c.setLineDash([]);
    // 图例放右下角：曲线在 t > 29 时早已升过 10⁻¹⁰，从不经过那里
    const lx = M.l + W - 430;
    c.font = "28px " + SANS; c.textAlign = "left"; c.textBaseline = "middle";
    c.strokeStyle = INK; c.lineWidth = 2; c.setLineDash([12, 10]);
    c.beginPath(); c.moveTo(lx, Y(-11.6)); c.lineTo(lx + 70, Y(-11.6)); c.stroke();
    c.fillStyle = INK; c.fillText("|Δ| = 1，记为分开", lx + 86, Y(-11.6));
    c.strokeStyle = NAVY; c.globalAlpha = 0.55; c.lineWidth = 2.5; c.setLineDash([4, 9]);
    c.beginPath(); c.moveTo(lx, Y(-13.4)); c.lineTo(lx + 70, Y(-13.4)); c.stroke();
    c.setLineDash([]); c.globalAlpha = 1;
    c.fillStyle = NAVY; c.fillText("斜率 λ₁ = 0.904", lx + 86, Y(-13.4));
    // 当前 ε 的参考斜率 λ₁
    if (cur) {
      const l0 = -cur.k, t1 = Math.min(T_MAX, (LMAX - 0.6 - l0) * Math.LN10 / LAMBDA1);
      c.strokeStyle = NAVY; c.globalAlpha = 0.55; c.lineWidth = 2.5; c.setLineDash([4, 9]);
      c.beginPath(); c.moveTo(X(0), Y(l0)); c.lineTo(X(t1), Y(l0 + LAMBDA1 * t1 / Math.LN10)); c.stroke();
      c.setLineDash([]); c.globalAlpha = 1;
    }
    // 每个 ε 一条曲线，当前的那条加粗
    runs.forEach(r => {
      const now = r === cur;
      c.strokeStyle = ORBIT; c.globalAlpha = now ? 1 : 0.38; c.lineWidth = now ? 4.5 : 3; c.lineJoin = "round";
      c.beginPath();
      r.pts.forEach((p, k) => { const yy = Y(Math.max(LMIN, Math.min(LMAX, p[1]))); k ? c.lineTo(X(p[0]), yy) : c.moveTo(X(p[0]), yy); });
      c.stroke(); c.globalAlpha = 1;
      if (r.T !== null) {
        c.fillStyle = now ? ORBIT : INK2;
        c.beginPath(); c.arc(X(r.T), Y(0), now ? 10 : 7, 0, 2 * Math.PI); c.fill();
      }
    });
    if (cur && cur.T !== null) {
      c.fillStyle = ORBIT; c.font = "500 32px " + MONO; c.textAlign = "left"; c.textBaseline = "top";
      c.fillText("T = " + cur.T.toFixed(1), X(cur.T) + 14, Y(0) + 14);
    }
  }

  function drawFit() {
    const done = runs.filter(r => r.T !== null);
    const el = document.getElementById("splitFit");
    if (done.length < 2) { el.textContent = "换一个 ε 再跑一次，比较两次分开的时刻。"; return; }
    const n = done.length;
    const mk = done.reduce((s, r) => s + r.k, 0) / n, mt = done.reduce((s, r) => s + r.T, 0) / n;
    const sxx = done.reduce((s, r) => s + (r.k - mk) ** 2, 0);
    const slope = done.reduce((s, r) => s + (r.k - mk) * (r.T - mt), 0) / sxx;
    el.innerHTML = "已跑 " + n + " 个 ε：每缩小 10 倍，分开时刻平均推后 <strong class=\"orbit\">" +
      slope.toFixed(2) + "</strong><span class=\"muted\">　理论 ln10 / λ₁ = 2.55</span>";
  }

  function draw() { drawPhase(); drawPlot(); drawFit(); }

  function frame() {
    if (cur && !finished()) advance(stepsPerFrame);
    else if (queue.length) start(queue.shift());
    draw();
    raf = (cur && !finished()) || queue.length ? requestAnimationFrame(frame) : null;
  }
  function play() {
    if (reduce) {
      while (cur) { while (!finished()) advance(stepsPerFrame); if (!queue.length) break; start(queue.shift()); }
      draw(); return;
    }
    if (raf === null && active) raf = requestAnimationFrame(frame);
  }

  slider.addEventListener("input", () => { label(); queue = []; stepsPerFrame = 16; start(+slider.value); play(); });
  document.getElementById("splitRun").addEventListener("click", () => { queue = []; stepsPerFrame = 16; start(+slider.value); play(); });
  document.getElementById("splitSweep").addEventListener("click", () => {
    runs = []; stepsPerFrame = 40;
    queue = [4, 6, 8, 10, 12, 14]; start(2); play();
  });
  document.getElementById("splitClear").addEventListener("click", () => {
    queue = []; runs = cur ? [cur] : []; draw();
  });
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
