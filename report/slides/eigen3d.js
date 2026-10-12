/* ---------- 备用「本征结构」：fig15 右图的可旋转版本 ---------- */
/* 不动点、特征值与方向来自 figures/slides_data.js 的 eigen（Python 端 all_eigen_structures 算好）；
   灰色轨道由 Lorenz.sampler 从共用起点积 12 个时间单位，与 fig15 同一段。
   正交投影、三轴等比例，所以画面里的夹角就是真实夹角。拖动旋转，滚轮缩放，双击回到默认视角。 */
const eigen3d = (function () {
  const cv = document.getElementById("eigenCanvas");
  const ctx = cv.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  const col = name => css.getPropertyValue(name).trim();
  const NAVY = col("--navy"), ORBIT = col("--orbit"), INK = col("--ink"), INK2 = col("--ink-2"), RULE = col("--rule");
  const MONO = col("--mono"), SANS = col("--sans");
  const W = cv.width, H = cv.height;
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const EIG = window.LORENZ_DATA.eigen;
  const [O, CP] = EIG;
  const add = (a, b, k = 1) => a.map((v, i) => v + k * b[i]);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

  // 线段长度与 fig15 相同：原点不稳定方向 ±16、稳定方向 ±11，C± 收缩方向 ±8，不变平面半宽 8
  const lines = [], quads = [];
  for (const es of EIG) {
    if (es.kind === "real") {
      es.dirs.forEach((d, k) => {
        const up = es.re[k] > 0, L = up ? 16 : 11;
        lines.push({ a: add(es.p, d, -L), b: add(es.p, d, L), color: up ? ORBIT : NAVY, w: up ? 5 : 3.4 });
      });
    } else {
      lines.push({ a: add(es.p, es.dirs[0], -8), b: add(es.p, es.dirs[0], 8), color: NAVY, w: 3.6 });
      const [u, v] = [es.dirs[1], es.dirs[2]], s = 8;
      quads.push([add(add(es.p, u, -s), v, -s), add(add(es.p, u, s), v, -s), add(add(es.p, u, s), v, s), add(add(es.p, u, -s), v, s)]);
    }
  }

  const traj = [];
  {
    const smp = Lorenz.sampler(Lorenz.field(Lorenz.RHO), Lorenz.start(), { dt: 0.004, tEnd: 12, rtol: 1e-10, atol: 1e-12 });
    traj.push(Lorenz.start());
    for (let r = smp.next(); r; r = smp.next()) traj.push(r[1]);
  }

  // 数据框：x ∈ [−20, 20]，y ∈ [−25, 25]，z ∈ [−12, 48]，盒心是旋转中心
  const BOX = [[-20, 20], [-25, 25], [-12, 48]];
  const CENTER = BOX.map(([a, b]) => (a + b) / 2);

  /* 视角与 matplotlib 相同：azim 绕 z 轴，elev 是视线仰角；默认 (−60°, 30°) 与 fig15 一致 */
  const DEFAULT = { az: -60, el: 30, zoom: 1 };
  let view = { ...DEFAULT };
  const rad = Math.PI / 180;
  function basis() {
    const a = view.az * rad, e = view.el * rad;
    return {
      eye: [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)],
      right: [-Math.sin(a), Math.cos(a), 0],
      up: [-Math.sin(e) * Math.cos(a), -Math.sin(e) * Math.sin(a), Math.cos(e)]
    };
  }
  let B = basis(), SCALE = 1;
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  function proj(p) {
    const q = [p[0] - CENTER[0], p[1] - CENTER[1], p[2] - CENTER[2]];
    return [W / 2 + SCALE * dot(q, B.right), H / 2 - SCALE * dot(q, B.up), dot(q, B.eye)];
  }

  function segment(a, b, color, w, alpha = 1) {
    const p = proj(a), q = proj(b);
    ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = w;
    ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
  }

  function drawBox() {
    const corners = [];
    for (const x of BOX[0]) for (const y of BOX[1]) for (const z of BOX[2]) corners.push([x, y, z]);
    // 只画背面三块墙的棱，前面的棱会挡住吸引子
    const far = corners.reduce((m, c) => (proj(c)[2] < proj(m)[2] ? c : m));
    ctx.lineCap = "round";
    for (let k = 0; k < 3; k++) {
      for (const c of corners) {
        if (c[k] !== BOX[k][0]) continue;
        const d = c.slice(); d[k] = BOX[k][1];
        const touchesFar = [0, 1, 2].some(j => j !== k && c[j] === far[j]);
        if (touchesFar) segment(c, d, RULE, 2);
      }
    }
  }

  /* 右上角的坐标轴小标：旋转时随视角转，告诉人 x、y、z 现在朝哪 */
  function drawGizmo() {
    const cx = W - 100, cy = 110, L = 52;
    ["x", "y", "z"].forEach((name, k) => {
      const e = [0, 0, 0]; e[k] = 1;
      const dx = L * dot(e, B.right), dy = -L * dot(e, B.up);
      ctx.globalAlpha = 1; ctx.strokeStyle = INK2; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + dx, cy + dy); ctx.stroke();
      ctx.fillStyle = INK; ctx.font = `italic 600 26px ${SANS}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(name, cx + dx * 1.4, cy + dy * 1.4);
    });
  }

  /* 轨道按深度分 6 档上色：近处深、远处浅，静止画面也能看出前后 */
  function drawTraj() {
    const P = traj.map(proj);
    let dmin = Infinity, dmax = -Infinity;
    for (const p of P) { dmin = Math.min(dmin, p[2]); dmax = Math.max(dmax, p[2]); }
    const BINS = 6, paths = Array.from({ length: BINS }, () => new Path2D());
    for (let i = 1; i < P.length; i++) {
      const d = (P[i][2] + P[i - 1][2]) / 2;
      const k = Math.min(BINS - 1, Math.floor((d - dmin) / (dmax - dmin + 1e-9) * BINS));
      paths[k].moveTo(P[i - 1][0], P[i - 1][1]); paths[k].lineTo(P[i][0], P[i][1]);
    }
    ctx.strokeStyle = INK2; ctx.lineWidth = 1.6; ctx.lineJoin = "round";
    paths.forEach((p, k) => { ctx.globalAlpha = 0.18 + 0.5 * k / (BINS - 1); ctx.stroke(p); });
  }

  function drawQuads() {
    for (const q of quads) {
      const P = q.map(proj);
      ctx.beginPath(); ctx.moveTo(P[0][0], P[0][1]);
      for (let k = 1; k < 4; k++) ctx.lineTo(P[k][0], P[k][1]);
      ctx.closePath();
      ctx.globalAlpha = 0.22; ctx.fillStyle = ORBIT; ctx.fill();
      ctx.globalAlpha = 0.55; ctx.strokeStyle = ORBIT; ctx.lineWidth = 1.6; ctx.stroke();
    }
  }

  function drawPoints() {
    ctx.globalAlpha = 1;
    const label = { origin: "原点", "C+": "C₊", "C-": "C₋" };
    for (const es of EIG) {
      const p = proj(es.p);
      ctx.fillStyle = INK; ctx.beginPath(); ctx.arc(p[0], p[1], 8, 0, 2 * Math.PI); ctx.fill();
      ctx.font = `600 26px ${SANS}`; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
      ctx.fillText(label[es.name], p[0] + 12, p[1] - 8);
    }
  }

  function draw() {
    B = basis();
    SCALE = view.zoom * Math.min(W, H) / 74;
    ctx.clearRect(0, 0, W, H);
    ctx.lineCap = "round";
    drawBox();
    drawGizmo();
    drawTraj();
    drawQuads();
    // 线段按深度从远到近画，近的压在上面
    lines.map(l => ({ l, d: proj(add(l.a, l.b).map(v => v / 2))[2] }))
      .sort((p, q) => p.d - q.d)
      .forEach(({ l }) => segment(l.a, l.b, l.color, l.w));
    drawPoints();
    ctx.globalAlpha = 1;
    document.getElementById("eigenView").textContent =
      "azim " + Math.round(view.az).toString().replace("-", "−") + "° · elev " + Math.round(view.el).toString().replace("-", "−") + "°";
  }

  /* 预设视角：x–z 正视（演示页那张蝴蝶）、正对 C₊ 的不变平面（看螺旋）、贴着平面看（看它有多薄） */
  const toView = d => ({ az: Math.atan2(d[1], d[0]) / rad, el: Math.asin(Math.max(-1, Math.min(1, d[2]))) / rad });
  const normal = cross(CP.dirs[1], CP.dirs[2]);
  const n = normal[2] < 0 ? normal.map(v => -v) : normal;
  const edge = cross(n, [0, 0, 1]);
  const PRESETS = {
    home: DEFAULT,
    xz: { az: -90, el: 0 },
    face: toView(n),
    edge: toView(edge[1] < 0 ? edge : edge.map(v => -v))
  };

  let raf = null;
  function tween(target) {
    if (raf !== null) cancelAnimationFrame(raf);
    const from = { ...view }, to = { zoom: 1, ...target };
    let dAz = ((to.az - from.az) % 360 + 540) % 360 - 180;   // 走近的那一边
    if (reduce) { view = { az: from.az + dAz, el: to.el, zoom: to.zoom }; draw(); return; }
    const t0 = performance.now(), T = 450;
    const step = now => {
      const s = Math.min(1, (now - t0) / T), e = s < 0.5 ? 2 * s * s : 1 - 2 * (1 - s) * (1 - s);
      view = { az: from.az + dAz * e, el: from.el + (to.el - from.el) * e, zoom: from.zoom + (to.zoom - from.zoom) * e };
      draw();
      raf = s < 1 ? requestAnimationFrame(step) : null;
    };
    raf = requestAnimationFrame(step);
  }
  document.querySelectorAll("#eigenSlide [data-view]").forEach(b =>
    b.addEventListener("click", () => tween(PRESETS[b.dataset.view])));

  // 拖动：水平改 azim，竖直改 elev；拖动的像素按舞台缩放折回 1600×900 的坐标
  let drag = null;
  const stageScale = () => cv.getBoundingClientRect().width / cv.offsetWidth || 1;
  cv.addEventListener("pointerdown", e => {
    if (raf !== null) { cancelAnimationFrame(raf); raf = null; }
    drag = { x: e.clientX, y: e.clientY, az: view.az, el: view.el };
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointermove", e => {
    if (!drag) return;
    const k = 0.35 / stageScale();
    view.az = drag.az - (e.clientX - drag.x) * k;
    view.el = Math.max(-89, Math.min(89, drag.el + (e.clientY - drag.y) * k));
    draw();
  });
  const end = () => { drag = null; };
  cv.addEventListener("pointerup", end);
  cv.addEventListener("pointercancel", end);
  cv.addEventListener("wheel", e => {
    e.preventDefault();
    view.zoom = Math.max(0.5, Math.min(4, view.zoom * Math.exp(-e.deltaY * 0.0015)));
    draw();
  }, { passive: false });
  cv.addEventListener("dblclick", () => tween(DEFAULT));
  // 在画布上拖动不触发翻页手势
  ["touchstart", "touchend"].forEach(t => cv.addEventListener(t, e => e.stopPropagation(), { passive: true }));

  draw();
  return {
    toggle(on) {
      if (on) draw();
      else if (raf !== null) { cancelAnimationFrame(raf); raf = null; }
    }
  };
})();
