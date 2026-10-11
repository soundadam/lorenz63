/* ---------- 演示①：拖 ρ，看对流卷从无到有，再到停不下来 ---------- */
const roll = (function () {
  const cell = document.getElementById("rollCell");
  const strip = document.getElementById("rollX");
  const phase = document.getElementById("rollPhase");
  const cc = cell.getContext("2d"), sc = strip.getContext("2d"), pc = phase.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  const col = name => css.getPropertyValue(name).trim();
  const NAVY = col("--navy"), ORBIT = col("--orbit"), INK = col("--ink"), INK2 = col("--ink-2"), RULE = col("--rule"), PAPER = col("--paper");
  const MONO = col("--mono"), SANS = col("--sans");
  const S = Lorenz.SIGMA, B = Lorenz.BETA, H = 0.005, STEPS = 4;
  const RHO_H = S * (S + B + 3) / (S - B - 1);   // 470/19：C± 在这里失稳
  // 由状态 (x, y, z) 重建流场与温度场，公式同 src/lorenz63/convection.py（H = κ = ΔT = 1）
  const A2 = 4 / B - 1, A = Math.sqrt(A2);       // β = 4/(1+a²)，所以 a² = 1/2
  const L = 2 / A;                               // 一个波长：一对反向旋转的卷
  const AMP = Math.SQRT2 * (1 + A2) / A;         // ψ = AMP·x·sin(πaX)·sin(πZ)
  const TAU = Math.PI * Math.PI * (1 + A2);      // Lorenz 的 1 个时间单位 = H²/κ 的 1/TAU
  const VIS = 0.15;                              // 短线按真实流速的 0.15 倍走，只示意方向与快慢
  const NOISE = 1e-3;                            // 流体里总有的微小扰动，让卷能在几秒内从静止长出来
  const WINDOW = 30, TRAIL = 15;                 // x(t) 显示最近 30 个时间单位，相平面留 15 个
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let rho = 0.5, u, t, hist, raf = null;
  const rk4 = v => Lorenz.rk4(Lorenz.field(rho), v, H);   // rho 随滑块变，每步现取
  function reset() { u = [NOISE, 0, 0]; t = 0; hist = [[0, u.slice()]]; }

  /* 温度场先画在一张小图上，再放大到对流层：底热（朱红）顶冷（藏青） */
  const GW = 180, GH = 64;
  const field = document.createElement("canvas");
  field.width = GW; field.height = GH;
  const fc = field.getContext("2d"), img = fc.createImageData(GW, GH);
  const cosX = [], sinZ = [], sin2Z = [], Zs = [];
  for (let i = 0; i < GW; i++) cosX.push(Math.cos(Math.PI * A * (i + 0.5) / GW * L));
  for (let j = 0; j < GH; j++) {
    const Z = 1 - (j + 0.5) / GH;
    Zs.push(Z); sinZ.push(Math.sin(Math.PI * Z)); sin2Z.push(Math.sin(2 * Math.PI * Z));
  }
  const rgb = hex => [1, 3, 5].map(k => parseInt(hex.slice(k, k + 2), 16));
  const COLD = rgb(NAVY), MID = [244, 241, 236], HOT = rgb(ORBIT);
  function paint(T) {
    const s = Math.max(0, Math.min(1, T));
    const [p, q, w] = s < 0.5 ? [COLD, MID, s * 2] : [MID, HOT, s * 2 - 1];
    return [0, 1, 2].map(k => p[k] + (q[k] - p[k]) * w);
  }
  function drawField() {
    const d = img.data;
    for (let j = 0; j < GH; j++) for (let i = 0; i < GW; i++) {
      const theta = (Math.SQRT2 * u[1] * cosX[i] * sinZ[j] - u[2] * sin2Z[j]) / (Math.PI * rho);
      const c = paint(1 - Zs[j] + theta), o = 4 * (j * GW + i);
      d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255;
    }
    fc.putImageData(img, 0, 0);
  }

  /* 示踪粒子：随当前速度场走，偶尔换一个随机位置，免得都挤到卷心 */
  const NP = 240, px = new Float64Array(NP), pz = new Float64Array(NP);
  const seed = k => { px[k] = Math.random() * L; pz[k] = 0.04 + 0.92 * Math.random(); };
  for (let k = 0; k < NP; k++) seed(k);
  function vel(X, Z) {
    const s = Math.sin(Math.PI * A * X), c = Math.cos(Math.PI * A * X);
    return [-AMP * u[0] * Math.PI * s * Math.cos(Math.PI * Z), AMP * u[0] * Math.PI * A * c * Math.sin(Math.PI * Z)];
  }
  function moveParticles(dtau) {
    for (let k = 0; k < NP; k++) {
      if (Math.random() < 0.006) { seed(k); continue; }
      const v = vel(px[k], pz[k]), g = VIS * dtau / TAU;
      px[k] = Math.max(0, Math.min(L, px[k] + v[0] * g));
      pz[k] = Math.max(0.01, Math.min(0.99, pz[k] + v[1] * g));
    }
  }

  const BAR = 52, CH = cell.height - 2 * BAR;
  const cx = X => X / L * cell.width, cz = Z => BAR + (1 - Z) * CH;
  function drawCell() {
    drawField();
    cc.imageSmoothingEnabled = true;
    cc.drawImage(field, 0, BAR, cell.width, CH);
    cc.fillStyle = NAVY; cc.fillRect(0, 0, cell.width, BAR);
    cc.fillStyle = ORBIT; cc.fillRect(0, cell.height - BAR, cell.width, BAR);
    cc.fillStyle = PAPER; cc.font = "500 32px " + SANS; cc.textAlign = "center"; cc.textBaseline = "middle";
    cc.fillText("顶板 · 冷", cell.width / 2, BAR / 2);
    cc.fillText("底板 · 热，温差是起对流所需的 " + rho.toFixed(1) + " 倍", cell.width / 2, cell.height - BAR / 2);
    cc.strokeStyle = INK; cc.globalAlpha = 0.6; cc.lineWidth = 4; cc.lineCap = "round";
    cc.beginPath();
    for (let k = 0; k < NP; k++) {
      const v = vel(px[k], pz[k]);
      const len = Math.min(0.1, 0.0009 * Math.hypot(v[0], v[1])) / (Math.hypot(v[0], v[1]) || 1);
      cc.moveTo(cx(px[k]), cz(pz[k]));
      cc.lineTo(cx(px[k] - v[0] * len), cz(pz[k] - v[1] * len));
    }
    cc.stroke(); cc.globalAlpha = 1;
  }

  /* x(t)：最近 WINDOW 个时间单位 */
  const XM = 24;
  function drawStrip() {
    const c = sc, W = strip.width, Hh = strip.height, l = 120, r = 20;
    const sx = tt => l + (tt - (t - WINDOW)) / WINDOW * (W - l - r);
    const sy = x => Hh / 2 - x / XM * (Hh / 2 - 16);
    c.clearRect(0, 0, W, Hh);
    c.fillStyle = col("--wash-navy"); c.fillRect(0, 0, W, Hh);
    c.strokeStyle = RULE; c.lineWidth = 2; c.beginPath(); c.moveTo(l, sy(0)); c.lineTo(W - r, sy(0)); c.stroke();
    c.fillStyle = INK2; c.font = "28px " + MONO; c.textAlign = "right"; c.textBaseline = "middle";
    c.fillText("x(t)", l - 18, sy(0));
    c.font = "28px " + SANS; c.textAlign = "left";
    c.fillText("x > 0：两侧升、中间降", l + 12, 32); c.fillText("x < 0：反过来", l + 12, Hh - 32);
    c.strokeStyle = NAVY; c.lineWidth = 4; c.lineJoin = "round"; c.beginPath();
    let first = true;
    hist.forEach(h => { if (h[0] < t - WINDOW) return; const X = sx(h[0]), Y = sy(Math.max(-XM, Math.min(XM, h[1][0]))); first ? c.moveTo(X, Y) : c.lineTo(X, Y); first = false; });
    c.stroke();
  }

  /* x–z 相平面：轨道尾迹与三个不动点，实心稳定、空心不稳定 */
  function drawPhase() {
    const c = pc, W = phase.width, Hh = phase.height, m = 36;
    const qx = x => W / 2 + x / 26 * (W / 2 - m);
    const qz = z => Hh - m - z / 54 * (Hh - 2 * m);
    c.clearRect(0, 0, W, Hh);
    c.fillStyle = col("--wash-navy"); c.fillRect(0, 0, W, Hh);
    c.fillStyle = INK2; c.font = "26px " + MONO; c.textBaseline = "middle";
    c.textAlign = "right"; c.fillText("x", W - 14, qz(0));
    c.textAlign = "left"; c.fillText("z", W / 2 + 12, m - 6);
    c.strokeStyle = RULE; c.lineWidth = 2; c.beginPath();
    c.moveTo(m, qz(0)); c.lineTo(W - m, qz(0)); c.moveTo(W / 2, Hh - m); c.lineTo(W / 2, m); c.stroke();
    c.strokeStyle = NAVY; c.lineWidth = 3; c.globalAlpha = 0.75; c.lineJoin = "round"; c.beginPath();
    let first = true;
    hist.forEach(h => { if (h[0] < t - TRAIL) return; const v = h[1]; first ? c.moveTo(qx(v[0]), qz(v[2])) : c.lineTo(qx(v[0]), qz(v[2])); first = false; });
    c.stroke(); c.globalAlpha = 1;
    const pts = [[0, 0, rho < 1]];
    if (rho > 1) { const e = Math.sqrt(B * (rho - 1)); pts.push([e, rho - 1, rho < RHO_H], [-e, rho - 1, rho < RHO_H]); }
    pts.forEach(([x, z, stable]) => {
      c.beginPath(); c.arc(qx(x), qz(z), 11, 0, 2 * Math.PI);
      c.lineWidth = 4; c.strokeStyle = INK; c.fillStyle = stable ? INK : PAPER; c.fill(); c.stroke();
    });
    c.fillStyle = ORBIT; c.beginPath(); c.arc(qx(u[0]), qz(u[2]), 12, 0, 2 * Math.PI); c.fill();
  }

  const regime = document.getElementById("rollRegime");
  const slider = document.getElementById("rollRho");
  function label() {
    document.getElementById("rollRhoV").textContent = rho.toFixed(1);
    regime.innerHTML = rho < 1 ? "<strong>静止</strong>：浮力敌不过粘性和热扩散，热量只靠传导"
      : rho < RHO_H ? "<strong>一对定常的对流卷</strong>：轨道落到不动点 C<sub>+</sub> 或 C<sub>−</sub>，转向取决于初值"
      : "<strong class=\"orbit\">卷不停地翻转</strong>：C± 也失稳，轨道哪里都停不下来";
  }
  function readout() {
    document.getElementById("rollNu").textContent = "Nu = 1 + 2z/ρ = " + (1 + 2 * u[2] / rho).toFixed(2);
  }
  function draw() { drawCell(); drawStrip(); drawPhase(); readout(); }

  function advance(n) {
    for (let k = 0; k < n; k++) { u = rk4(u); t += H; hist.push([t, u.slice()]); }
    u[0] += (Math.random() - 0.5) * 2 * NOISE;
    while (hist.length && hist[0][0] < t - WINDOW) hist.shift();
    moveParticles(n * H);
  }
  function frame() {
    advance(STEPS); draw();
    raf = requestAnimationFrame(frame);
  }
  function settle() { if (reduce) { for (let k = 0; k < 600; k++) advance(STEPS); draw(); } }
  function setRho(r) { rho = r; slider.value = r; label(); settle(); }

  slider.addEventListener("input", () => setRho(+slider.value));
  document.querySelectorAll("#rollSlide [data-rho]").forEach(b => b.addEventListener("click", () => setRho(+b.dataset.rho)));
  // 推一下：z 突然加 15。ρ = 24.5 时从定常卷或它的衰减螺旋出发，这样推都会落到混沌集上
  document.getElementById("rollKick").addEventListener("click", () => { u[2] += 15; settle(); });
  document.getElementById("rollReset").addEventListener("click", () => { reset(); settle(); });
  reset(); label(); draw();

  return {
    toggle(on) {
      if (on && reduce) settle();
      if (on && !reduce && raf === null) raf = requestAnimationFrame(frame);
      if (!on && raf !== null) { cancelAnimationFrame(raf); raf = null; }
    }
  };
})();
