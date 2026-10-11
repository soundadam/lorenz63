/* ---------- 演示共用的向量场与 DOP853；经典参数同 src/lorenz63/system.py ---------- */
/* DOP853 照 scipy.integrate._ivp.rk.DOP853 逐行移植：同一张系数表、同一个初始步长选择与步长控制、
   同一个 7 次稠密输出。幻灯片里的轨道因此和 Python 端 solve_ivp(method="DOP853") 是同一个算法。 */
const Lorenz = (function () {
  const SIGMA = 10, BETA = 8 / 3, RHO = 28;
  const { N_STAGES, A, B, C, E3, E5, D } = DOP853_COEF;
  const SAFETY = 0.9, MIN_FACTOR = 0.2, MAX_FACTOR = 10, ERR_EXP = -1 / 8;

  function field(rho) {
    return u => [SIGMA * (u[1] - u[0]), u[0] * (rho - u[2]) - u[1], u[0] * u[1] - BETA * u[2]];
  }

  const rms = v => Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length);

  // scipy.integrate._ivp.common.select_initial_step，误差估计阶 7
  function initialStep(f, y0, f0, span, rtol, atol) {
    const scale = y0.map(v => atol + Math.abs(v) * rtol);
    const d0 = rms(y0.map((v, i) => v / scale[i])), d1 = rms(f0.map((v, i) => v / scale[i]));
    let h0 = d0 < 1e-5 || d1 < 1e-5 ? 1e-6 : 0.01 * d0 / d1;
    h0 = Math.min(h0, span);
    const f1 = f(y0.map((v, i) => v + h0 * f0[i]));
    const d2 = rms(f1.map((v, i) => (v - f0[i]) / scale[i])) / h0;
    const h1 = d1 <= 1e-15 && d2 <= 1e-15 ? Math.max(1e-6, h0 * 1e-3) : Math.pow(0.01 / Math.max(d1, d2), 1 / 8);
    return Math.min(100 * h0, h1, span);
  }

  /* 一个求解器实例：从 (t0, y0) 积到 tEnd。每次 step() 前进一个被接受的步，
     之后可以用 dense(t) 在刚走完的这一步 [tOld, t] 内取值。 */
  function solver(f, y0, { t0 = 0, tEnd, rtol, atol, h0 }) {
    const n = y0.length;
    const s = { t: t0, y: y0.slice(), f: f(y0), tOld: t0, steps: 0, done: false };
    let hAbs = h0 ?? initialStep(f, s.y, s.f, tEnd - t0, rtol, atol);
    let K = [], yOld = null, hPrev = 0, F = null;

    const comb = (rows, coef, upto) => {
      const out = new Array(n).fill(0);
      for (let j = 0; j < upto; j++) if (coef[j] !== 0) for (let i = 0; i < n; i++) out[i] += coef[j] * rows[j][i];
      return out;
    };

    function trial(h) {
      K = [s.f];
      for (let st = 1; st < N_STAGES; st++) {
        const dy = comb(K, A[st], st);
        K.push(f(s.y.map((v, i) => v + h * dy[i])));
      }
      const yNew = comb(K, B, N_STAGES).map((v, i) => s.y[i] + h * v);
      K.push(f(yNew));
      return yNew;
    }
    function errorNorm(h, yNew) {
      const scale = s.y.map((v, i) => atol + Math.max(Math.abs(v), Math.abs(yNew[i])) * rtol);
      const e5 = comb(K, E5, N_STAGES + 1).map((v, i) => v / scale[i]);
      const e3 = comb(K, E3, N_STAGES + 1).map((v, i) => v / scale[i]);
      const n5 = e5.reduce((a, x) => a + x * x, 0), n3 = e3.reduce((a, x) => a + x * x, 0);
      if (n5 === 0 && n3 === 0) return 0;
      return Math.abs(h) * n5 / Math.sqrt((n5 + 0.01 * n3) * n);
    }

    s.step = function () {
      if (s.done) return false;
      let accepted = false, rejected = false, h, tNew, yNew;
      while (!accepted) {
        tNew = Math.min(s.t + hAbs, tEnd);
        h = tNew - s.t;
        yNew = trial(h);
        const err = errorNorm(h, yNew);
        if (err < 1) {
          let factor = err === 0 ? MAX_FACTOR : Math.min(MAX_FACTOR, SAFETY * Math.pow(err, ERR_EXP));
          if (rejected) factor = Math.min(1, factor);
          hAbs = Math.abs(h) * factor;
          accepted = true;
        } else {
          hAbs = Math.abs(h) * Math.max(MIN_FACTOR, SAFETY * Math.pow(err, ERR_EXP));
          rejected = true;
        }
      }
      yOld = s.y; hPrev = h; F = null;
      s.tOld = s.t; s.t = tNew; s.y = yNew; s.f = K[N_STAGES];
      s.steps += 1; s.hNext = hAbs;
      if (s.t >= tEnd) s.done = true;
      return true;
    };

    // scipy 的 DOP853._dense_output_impl 与 Dop853DenseOutput：多算 3 个级，得到 7 次插值多项式
    s.dense = function (t) {
      if (F === null) {
        const KE = K.slice();
        for (let st = N_STAGES + 1; st < A.length; st++) {
          const dy = comb(KE, A[st], st);
          KE.push(f(yOld.map((v, i) => v + hPrev * dy[i])));
        }
        const dY = s.y.map((v, i) => v - yOld[i]);
        F = [dY, KE[0].map((v, i) => hPrev * v - dY[i]), dY.map((v, i) => 2 * v - hPrev * (s.f[i] + KE[0][i]))];
        for (const d of D) F.push(comb(KE, d, KE.length).map(v => hPrev * v));
      }
      const x = (t - s.tOld) / hPrev;
      const y = new Array(n).fill(0);
      for (let k = F.length - 1, m = 0; k >= 0; k--, m++) {
        for (let i = 0; i < n; i++) { y[i] += F[k][i]; y[i] *= m % 2 === 0 ? x : 1 - x; }
      }
      return y.map((v, i) => v + yOld[i]);
    };
    return s;
  }

  /* 按 dt 等距取样的 DOP853（相当于 solve_ivp(dense_output=True) 再在网格上取值）。
     next() 每次返回下一个样本 [t, y]，到 tEnd 之后返回 null；求解器自己选步，取样不影响步长。 */
  function sampler(f, y0, { t0 = 0, dt, tEnd, rtol, atol, h0 }) {
    const s = solver(f, y0, { t0, tEnd, rtol, atol, h0 });
    let k = 0;
    return {
      solver: s,
      next() {
        const tk = t0 + (k + 1) * dt;
        if (tk > tEnd + 1e-9) return null;
        while (s.t < tk - 1e-12) if (!s.step()) return null;
        k += 1;
        return [tk, s.dense(Math.min(tk, s.t))];
      }
    };
  }

  // 所有演示共用 Python 端的起点 on_attractor(CLASSICAL)：(1,1,1) 出发、参考精度丢掉 50 个时间单位的瞬态
  const START = window.LORENZ_DATA.u0;

  return { SIGMA, BETA, RHO, field, solver, sampler, start: () => START.slice() };
})();
