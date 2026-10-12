/* ---------- 翻页、讲稿、计时；进出页时开关各页的演示 ---------- */
(function () {
  const stage = document.getElementById("stage");
  const area = document.getElementById("stageArea");
  const slides = Array.from(document.querySelectorAll(".slide"));
  const chips = Array.from(document.querySelectorAll("#chain span"));
  const panel = document.getElementById("notesPanel");
  const notesBtn = document.getElementById("notesBtn");
  const backBtn = document.getElementById("backBtn");
  const N = slides.length;

  // 目录和各段分隔页共用一张思维导图；点一行跳到那一段的分隔页
  const firstOf = p => slides.findIndex(s => s.dataset.part === p);
  const tpl = document.getElementById("mapTpl");
  document.querySelectorAll(".map-host").forEach(host => {
    host.appendChild(tpl.content.cloneNode(true));
    const focus = host.dataset.focus;
    const map = host.querySelector(".mmap");
    if (focus) {
      map.dataset.focus = focus;
      map.querySelectorAll("[data-p]").forEach(el => el.classList.toggle("on", el.dataset.p === focus));
    }
    map.querySelectorAll(".mm-row").forEach(row =>
      row.addEventListener("click", () => go(firstOf(row.dataset.p))));
  });

  // 正文里的 data-goto 链接：跳到那个 id 的页
  document.querySelectorAll("[data-goto]").forEach(a =>
    a.addEventListener("click", () => go(slides.findIndex(s => s.id === a.dataset.goto))));

  // 备用目录：列出它后面的每一页
  const backupAt = slides.findIndex(s => s.id === "backupIndex");
  const backupList = document.getElementById("backupList");
  slides.slice(backupAt + 1).forEach((s, k) => {
    const li = document.createElement("li");
    li.textContent = s.dataset.title;
    const eyebrow = s.querySelector(".eyebrow");
    if (eyebrow && eyebrow.textContent !== "备用") {
      const tag = document.createElement("span");
      tag.textContent = eyebrow.textContent.replace(/^备用 · /, "").split(" · ")[0];
      li.appendChild(tag);
    }
    li.addEventListener("click", () => go(backupAt + 1 + k));
    backupList.appendChild(li);
  });

  const minutes = slides.map(s => parseFloat(s.dataset.min) || 0);
  const cumulative = minutes.reduce((acc, m, k) => (acc.push((acc[k - 1] || 0) + m), acc), []);
  const total = cumulative[N - 1];

  let cur = 0;
  let ret = null;   // 从正文进入备用页时记下离开的那一页，R 或“返回”回去
  const fromHash = /^#s(\d+)$/.exec(location.hash || "");
  if (fromHash) cur = Math.min(N - 1, Math.max(0, parseInt(fromHash[1], 10) - 1));

  function fit() {
    const r = area.getBoundingClientRect();
    const s = Math.max(0.05, Math.min(r.width / 1600, r.height / 900));
    stage.style.setProperty("--s", s);
  }

  function fmt(m) { return (Math.round(m * 2) / 2).toFixed(1).replace(/\.0$/, ""); }

  function go(n) {
    n = Math.min(N - 1, Math.max(0, n));
    if (n < backupAt) ret = null;
    else if (cur < backupAt && ret === null && n !== cur) ret = cur;
    cur = n;
    slides.forEach((s, k) => s.classList.toggle("active", k === cur));
    const sl = slides[cur];
    const part = sl.dataset.part;
    chips.forEach(c => c.classList.toggle("on", c.dataset.p === part));
    stage.classList.toggle("is-cover", sl.classList.contains("cover"));
    const label = cur > backupAt ? "备用 " + (cur - backupAt) : cur === backupAt ? "备用" : (cur + 1) + " / " + backupAt;
    document.getElementById("pg").textContent = label;
    document.getElementById("count").textContent = label;
    backBtn.hidden = ret === null;
    if (ret !== null) backBtn.textContent = "↩ 返回第 " + (ret + 1) + " 页 R";
    const notes = sl.querySelector(".notes");
    document.getElementById("nBody").innerHTML = notes ? notes.innerHTML : "";
    document.getElementById("nTime").textContent =
      "本页约 " + fmt(minutes[cur]) + " 分钟 · 讲到这里累计 " + fmt(cumulative[cur]) + " / " + fmt(total) + " 分钟";
    document.getElementById("nNext").textContent =
      cur < N - 1 ? "下一页：" + slides[cur + 1].dataset.title : "最后一页";
    try { history.replaceState(null, "", "#s" + (cur + 1)); } catch (e) {}
    hero.toggle(sl.classList.contains("cover"));
    paradox.toggle(sl.id === "paradoxSlide");
    split.toggle(sl.id === "splitSlide");
    roll.toggle(sl.id === "rollSlide");
    eigen3d.toggle(sl.id === "eigenSlide");
  }

  function toggleNotes(force) {
    const show = typeof force === "boolean" ? force : panel.hidden;
    panel.hidden = !show;
    notesBtn.setAttribute("aria-pressed", String(show));
    try { localStorage.setItem("lorenz-slides-notes", show ? "1" : "0"); } catch (e) {}
    fit();
  }

  function toggleFullscreen() {
    const d = document;
    try {
      if (d.fullscreenElement) d.exitFullscreen().catch(() => {});
      else if (d.documentElement.requestFullscreen) d.documentElement.requestFullscreen().catch(() => {});
    } catch (e) {}
  }

  document.getElementById("prev").addEventListener("click", () => go(cur - 1));
  document.getElementById("next").addEventListener("click", () => go(cur + 1));
  notesBtn.addEventListener("click", () => toggleNotes());
  document.getElementById("fsBtn").addEventListener("click", toggleFullscreen);
  backBtn.addEventListener("click", () => { if (ret !== null) go(ret); });

  document.addEventListener("keydown", e => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (k === "ArrowRight" || k === "ArrowDown" || k === "PageDown" || k === " ") { e.preventDefault(); go(cur + 1); }
    else if (k === "ArrowLeft" || k === "ArrowUp" || k === "PageUp") { e.preventDefault(); go(cur - 1); }
    else if (k === "Home") go(0);
    else if (k === "b" || k === "B") go(backupAt);
    else if ((k === "r" || k === "R") && ret !== null) go(ret);
    else if (k === "End") go(N - 1);
    else if (k === "n" || k === "N") toggleNotes();
    else if (k === "f" || k === "F") toggleFullscreen();
  });

  let tx = null, ty = null;
  area.addEventListener("touchstart", e => { tx = e.touches[0].clientX; ty = e.touches[0].clientY; }, { passive: true });
  area.addEventListener("touchend", e => {
    if (tx === null) return;
    const dx = e.changedTouches[0].clientX - tx, dy = e.changedTouches[0].clientY - ty;
    if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy)) go(cur + (dx < 0 ? 1 : -1));
    tx = ty = null;
  });

  if (window.ResizeObserver) new ResizeObserver(fit).observe(area);
  window.addEventListener("resize", fit);

  try { if (localStorage.getItem("lorenz-slides-notes") === "1") toggleNotes(true); } catch (e) {}
  fit();
  go(cur);
})();
