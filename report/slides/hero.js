/* ---------- 封面：两条初值相差 1e-5 的轨道，RK4 实时积分 ---------- */
const hero = (function () {
  const cv = document.getElementById("hero");
  const ctx = cv.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  const colA = css.getPropertyValue("--navy").trim();
  const colB = css.getPropertyValue("--orbit").trim();
  const DELTA = 1e-5;
  const HS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.04];   // 滑块档位；0.005 是其余演示用的步长
  const T_FRAME = 0.04;                                  // 每帧推进的物理时间，与 h 无关，换 h 不改变播放速度
  const DT_DRAW = 0.0025;                                // 画线的时间分辨率：步与步之间用三次 Hermite 补点
  const TRAIL = 16.8;                                    // 尾迹保留的时间长度
  const KEEP = Math.round(TRAIL / DT_DRAW);
  const slider = document.getElementById("heroH");
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const f = Lorenz.field(Lorenz.RHO);
  /* 推进一步，并把这一步内部按 DT_DRAW 用三次 Hermite 插值补进尾迹。
     插值只决定线画得多细，不改变 u：轨道的精度只由 h 决定。 */
  function step(o, h) {
    const u1 = Lorenz.rk4(f, o.u, h, o.f), f1 = f(u1);
    const m = Math.max(1, Math.ceil(h / DT_DRAW));
    for (let j = 1; j <= m; j++) {
      const s = j / m, s2 = s * s, s3 = s2 * s;
      const c0 = 2 * s3 - 3 * s2 + 1, c1 = s3 - 2 * s2 + s, c2 = -2 * s3 + 3 * s2, c3 = s3 - s2;
      o.pts.push(o.u.map((v, i) => c0 * v + c1 * h * o.f[i] + c2 * u1[i] + c3 * h * f1[i]));
    }
    if (o.pts.length > KEEP) o.pts.splice(0, o.pts.length - KEEP);
    o.u = u1; o.f = f1;
  }

  let h, a, b, t, tSep, raf = null;
  const u0 = Lorenz.start();
  function orbit(u) { return { u: u.slice(), f: f(u), pts: [u.slice()] }; }
  function reset() {
    h = HS[+slider.value];
    a = orbit(u0); b = orbit([u0[0] + DELTA, u0[1], u0[2]]);
    t = 0; tSep = null;
    document.getElementById("heroHV").textContent = String(h);
  }
  const px = x => (x + 24) / 48 * cv.width;
  const pz = z => cv.height - (z - 2) / 52 * cv.height;

  function trail(pts, col) {
    ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.globalAlpha = 0.9; ctx.lineJoin = "round";
    ctx.beginPath();
    pts.forEach((p, k) => (k ? ctx.lineTo(px(p[0]), pz(p[2])) : ctx.moveTo(px(p[0]), pz(p[2]))));
    ctx.stroke();
    const e = pts[pts.length - 1];
    ctx.globalAlpha = 1; ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(px(e[0]), pz(e[2]), 7, 0, 2 * Math.PI); ctx.fill();
  }
  const dist = () => Math.hypot(a.u[0] - b.u[0], a.u[1] - b.u[1], a.u[2] - b.u[2]);
  function draw() {
    ctx.clearRect(0, 0, cv.width, cv.height);
    trail(a.pts, colA); trail(b.pts, colB);
    document.getElementById("heroT").textContent = "t = " + t.toFixed(1);
    document.getElementById("heroD").textContent = "|Δ| = " + dist().toExponential(1).replace("-", "−");
    document.getElementById("heroS").textContent = "|Δ| > 1 于 t = " + (tSep === null ? "—" : tSep.toFixed(1));
  }
  function advance() {
    const n = Math.max(1, Math.round(T_FRAME / h));
    for (let k = 0; k < n; k++) {
      step(a, h); step(b, h); t += h;
      if (tSep === null && dist() > 1) tSep = t;
    }
  }
  function frame() {
    advance(); draw();
    if (t > 60) reset();
    raf = requestAnimationFrame(frame);
  }
  function settle() { for (let k = 0; k < 900; k++) advance(); draw(); }
  slider.addEventListener("input", () => { reset(); if (reduce) settle(); else draw(); });
  reset();
  if (reduce) settle();
  else draw();

  return {
    toggle(on) {
      if (reduce) return;
      if (on && raf === null) raf = requestAnimationFrame(frame);
      if (!on && raf !== null) { cancelAnimationFrame(raf); raf = null; }
    }
  };
})();
