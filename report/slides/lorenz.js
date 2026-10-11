/* ---------- 三个演示共用的向量场与定步长 RK4；经典参数同 src/lorenz63/system.py ---------- */
const Lorenz = (function () {
  const SIGMA = 10, BETA = 8 / 3, RHO = 28;

  function field(rho) {
    return u => [SIGMA * (u[1] - u[0]), u[0] * (rho - u[2]) - u[1], u[0] * u[1] - BETA * u[2]];
  }
  // k1 可由调用方传入：封面做 Hermite 插值时已经算过这一点的 f
  function rk4(f, u, h, k1 = f(u)) {
    const k2 = f(u.map((v, i) => v + 0.5 * h * k1[i]));
    const k3 = f(u.map((v, i) => v + 0.5 * h * k2[i]));
    const k4 = f(u.map((v, i) => v + h * k3[i]));
    return u.map((v, i) => v + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  }

  // 封面与缩小初值差的演示用同一个起点：ρ = 28、h = 0.005，从 (1,1,1) 丢掉 20 个时间单位的瞬态
  const START = (function () {
    const f = field(RHO);
    let u = [1, 1, 1];
    for (let k = 0; k < 4000; k++) u = rk4(f, u, 0.005);
    return u;
  })();

  return { SIGMA, BETA, RHO, field, rk4, start: () => START.slice() };
})();
