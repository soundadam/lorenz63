/* ---------- 封面：两条初值相差 1e-5 的轨道，回放 figures/hero_data.js ---------- */
/* 数据由 `lorenz63 figure hero` 用 DOP853（rtol=1e-13）算好，浏览器只回放，到 |Δ| 首次超过 5 停住。 */
const hero = (function () {
  const D = window.LORENZ_HERO;
  const cv = document.getElementById("hero");
  const ctx = cv.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  const colA = css.getPropertyValue("--navy").trim();
  const colB = css.getPropertyValue("--orbit").trim();
  const T_FRAME = 0.04;                                   // 每帧推进的物理时间
  const PER_FRAME = Math.max(1, Math.round(T_FRAME / D.dt));
  const N = D.ax.length;
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const px = x => (x + 24) / 48 * cv.width;
  const pz = z => cv.height - (z - 2) / 52 * cv.height;

  function trail(xs, zs, n, col, w) {
    ctx.strokeStyle = col; ctx.lineWidth = w; ctx.globalAlpha = 0.9; ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(px(xs[0]), pz(zs[0]));
    for (let k = 1; k <= n; k++) ctx.lineTo(px(xs[k]), pz(zs[k]));
    ctx.stroke();
    ctx.globalAlpha = 1; ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(px(xs[n]), pz(zs[n]), 7, 0, 2 * Math.PI); ctx.fill();
  }
  let n = 0, raf = null;
  function draw() {
    ctx.clearRect(0, 0, cv.width, cv.height);
    trail(D.ax, D.az, n, colA, 6); trail(D.bx, D.bz, n, colB, 2.4);   // 蓝线垫在下面加宽，重合时两色都看得见
    document.getElementById("heroT").textContent = "t = " + (n * D.dt).toFixed(1);
    document.getElementById("heroD").textContent = "|Δ| = " + Math.pow(10, D.logd[n]).toExponential(1).replace("-", "−");
  }
  function frame() {
    n = Math.min(N - 1, n + PER_FRAME);
    draw();
    raf = n < N - 1 ? requestAnimationFrame(frame) : null;
  }
  function play() {
    if (raf !== null) cancelAnimationFrame(raf);
    n = 0;
    if (reduce) { n = N - 1; draw(); raf = null; return; }
    draw();
    raf = requestAnimationFrame(frame);
  }
  document.getElementById("heroReplay").addEventListener("click", play);
  n = N - 1; draw();

  return {
    toggle(on) {
      if (on) play();
      else if (raf !== null) { cancelAnimationFrame(raf); raf = null; }
    }
  };
})();
