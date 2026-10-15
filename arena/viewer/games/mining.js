/* ActionMan Arena, mining: pixel art, lantern light, follow camera. Registers ARENA.games.mining. */
(() => {
"use strict";
const A = window.ARENA, RM = A.RM, rng = A.rng;
const CELL = 16, SURFACE = 3, VW = 15, VH = 11;            // follow camera: 15 x 11 blocks

/* ---------- block textures (procedural, 16x16) ---------- */
const TEX = {};
function tex(draw) { const c = document.createElement("canvas"); c.width = c.height = CELL; draw(c.getContext("2d")); return c; }
function speckle(x, r, base, spots) {
  x.fillStyle = base; x.fillRect(0, 0, 16, 16);
  for (const [col, n, w, h] of spots) { x.fillStyle = col; for (let i = 0; i < n; i++) x.fillRect(Math.floor(r() * 16), Math.floor(r() * 16), w, h); }
}
for (let v = 0; v < 3; v++) {
  const r = rng(101 + v * 17);
  TEX["d" + v] = tex(x => speckle(x, r, "#6b4a2f", [["#5a3c25", 14, 2, 1], ["#7d5a3a", 10, 1, 1], ["#4d321f", 5, 1, 1]]));
  TEX["g" + v] = tex(x => {
    speckle(x, r, "#6b4a2f", [["#5a3c25", 12, 2, 1], ["#7d5a3a", 8, 1, 1]]);
    x.fillStyle = "#5fa548"; x.fillRect(0, 0, 16, 3);
    x.fillStyle = "#4c8a3a"; for (let i = 0; i < 16; i++) x.fillRect(i, 3, 1, Math.floor(r() * 3));
    x.fillStyle = "#8fd06f"; for (let i = 0; i < 5; i++) x.fillRect(Math.floor(r() * 16), 0, 1, 1);
  });
  TEX["s" + v] = tex(x => speckle(x, r, "#6f7280", [["#5d606d", 8, 3, 2], ["#878a97", 10, 1, 1], ["#555866", 8, 2, 1]]));
  TEX["D" + v] = tex(x => {
    speckle(x, r, "#6f7280", [["#5d606d", 6, 3, 2], ["#878a97", 6, 1, 1]]);
    for (const [gx, gy] of [[2, 3], [9, 2], [4, 9], [11, 10], [7, 6]]) {
      x.fillStyle = "#1f8f99"; x.fillRect(gx, gy, 3, 3); x.fillStyle = "#62e3ea"; x.fillRect(gx, gy, 2, 2);
      x.fillStyle = "#e9feff"; x.fillRect(gx, gy, 1, 1);
    }
  });
  TEX["#" + v] = tex(x => speckle(x, r, "#2b2734", [["#1a1720", 16, 2, 2], ["#3a3546", 8, 1, 1]]));
  TEX["c" + v] = tex(x => speckle(x, r, "#231c2b", [["#1b1621", 10, 2, 1], ["#2c2436", 8, 1, 1]]));
}
const CRUMB = { d: "#7d5a3a", s: "#8a8d9a", D: "#62e3ea", "#": "#3a3546" };

/* ---------- sprites ---------- */
const PAL = { Y: "#f2b35b", L: "#fff4c8", h: "#4a3222", s: "#e0a77c", e: "#1b1520", B: "#3d6ea8", b: "#2f5786", P: "#3a3346", K: "#221d29",
              G: "#5e9d4f", g: "#4a7f3e", r: "#2a0f0f", d: "#2d4a27", T: "#3a8f8a", V: "#45386e", k: "#1e1a26" };
const MINER = ["................", "....YYYYYYY.....", "...YYYYYYYYYL...", "....hhhhhhhh....", "....ssssssss....", "....sssesses....",
  "....ssssssss....", ".....ssssss.....", "...BBBBBBBBBB...", "..sBBBbBBBBBBs..", "..sBBBbBBBBBBs..", "....BBBBBBBB....",
  "....PPPPPPPP....", "....PPP..PPP....", "....PPP..PPP....", "....KKK..KKK...."];
const ZOMBIE = ["................", "....GGGGGGGG....", "....GgGGGGgG....", "....GGrGGGrG....", "....GGGGGGGG....", "....GGGddGGG....",
  "....gGGGGGGg....", ".....GGGGGG.....", "...TTTTTTTTGGGG.", "...TTTTTTTTGGGG.", "...TTTTTTTT.....", "....TTTTTTT.....",
  "....VVVVVVVV....", "....VVV..VVV....", "....VVV..VVV....", "....kkk..kkk...."];
function sprite(map) { return tex(x => map.forEach((row, y) => [...row].forEach((ch, i) => { if (ch !== ".") { x.fillStyle = PAL[ch]; x.fillRect(i, y, 1, 1); } }))); }
function flip(c) { return tex(x => { x.translate(16, 0); x.scale(-1, 1); x.drawImage(c, 0, 0); }); }
const SPR = { miner: sprite(MINER), zombie: sprite(ZOMBIE) };
SPR.minerL = flip(SPR.miner); SPR.zombieL = flip(SPR.zombie);

/* ---------- pixel hearts / apples ---------- */
const HEART = [".XX.XX.", "XXXXXXX", "XXXXXXX", ".XXXXX.", "..XXX..", "...X..."];
function heartSvg(state) {
  let s = '<svg viewBox="0 0 7 6" aria-hidden="true">';
  HEART.forEach((row, y) => [...row].forEach((c, x) => {
    if (c === "X") s += `<rect x="${x}" y="${y}" width="1" height="1" fill="${state === "full" || (state === "half" && x <= 3) ? "#e8634f" : "#3a3549"}"/>`;
  }));
  return s + "</svg>";
}
const HEARTS = { full: heartSvg("full"), half: heartSvg("half"), empty: heartSvg("empty") };
const APPLE = '<svg viewBox="0 0 7 6" aria-hidden="true"><rect x="3" y="0" width="1" height="1" fill="#6b4a2f"/><rect x="4" y="0" width="1" height="1" fill="#5fa548"/>' +
  '<rect x="1" y="1" width="5" height="4" fill="#c9473a"/><rect x="2" y="5" width="3" height="1" fill="#c9473a"/><rect x="2" y="2" width="1" height="1" fill="#f2a493"/></svg>';


/* ---------- world drawing ---------- */

function paintWorld(L, g) {
  const x = L.wctx, W = g[0].length, H = g.length;
  const sky = x.createLinearGradient(0, 0, 0, SURFACE * CELL);
  sky.addColorStop(0, "#5aa6dc"); sky.addColorStop(1, "#c4e8f5");
  x.fillStyle = sky; x.fillRect(0, 0, W * CELL, SURFACE * CELL);
  x.fillStyle = "rgba(255,255,255,.9)";
  for (const [cx, cy, w] of L.clouds) { x.fillRect(cx, cy, w, 5); x.fillRect(cx + 5, cy - 4, w - 12, 4); }
  for (let y = 0; y < H; y++) for (let i = 0; i < W; i++) {
    const b = g[y][i], v = (i * 7 + y * 13) % 3;
    if (b === ".") { if (y >= SURFACE) x.drawImage(TEX["c" + v], i * CELL, y * CELL); continue; }
    x.drawImage(TEX[(b === "d" && y === SURFACE ? "g" : b) + v], i * CELL, y * CELL);
  }
  x.fillStyle = "rgba(0,0,0,.32)";                        // inner shadow where open space meets rock
  for (let y = SURFACE; y < H; y++) for (let i = 0; i < W; i++) {
    if (g[y][i] !== ".") continue;
    if (y > 0 && g[y - 1][i] !== ".") x.fillRect(i * CELL, y * CELL, CELL, 3);
    if (i > 0 && g[y][i - 1] !== ".") x.fillRect(i * CELL, y * CELL, 2, CELL);
  }
}

function lighting(x, g, mx, my, t) {
  const W = g[0].length, H = g.length, MAXA = .74, STEPS = 6;
  const flick = RM ? 0 : Math.sin(t * 1.7) * .15 + Math.sin(t * 4.3) * .07;
  for (let y = SURFACE; y < H; y++) for (let i = 0; i < W; i++) {
    const d = Math.hypot(i - mx, y - my);
    let l = Math.max(0, 1 - (d - 1.3 - flick) / 4.6);
    l = Math.max(l, y === SURFACE ? .78 : y === SURFACE + 1 ? .5 : y === SURFACE + 2 ? .22 : 0);
    if (g[y][i] === "D") l = Math.max(l, .5);
    const a = Math.round(MAXA * (1 - l) * STEPS / MAXA) * MAXA / STEPS;
    if (a > 0) { x.fillStyle = `rgba(10,8,18,${a.toFixed(3)})`; x.fillRect(i * CELL, y * CELL, CELL, CELL); }
  }
  if (my >= SURFACE - 1) {                                  // warm lantern bloom
    const cx = (mx + .5) * CELL, cy = (my + .3) * CELL, gr = x.createRadialGradient(cx, cy, 2, cx, cy, CELL * 3.2);
    gr.addColorStop(0, "rgba(242,179,91,.22)"); gr.addColorStop(1, "rgba(242,179,91,0)");
    x.fillStyle = gr; x.fillRect(cx - CELL * 4, cy - CELL * 4, CELL * 8, CELL * 8);
  }
}

function glints(x, g, t) {
  for (let y = SURFACE; y < g.length; y++) for (let i = 0; i < g[0].length; i++) {
    if (g[y][i] !== "D") continue;
    const ph = RM ? .5 : ((t * .9 + i * .37 + y * .61) % 3) / 3;
    const a = RM ? .35 : Math.max(0, 1 - Math.abs(ph - .5) * 6);
    if (a <= 0) continue;
    const cx = i * CELL + 10, cy = y * CELL + 5;
    x.fillStyle = `rgba(233,254,255,${a.toFixed(2)})`;
    x.fillRect(cx - 2, cy, 5, 1); x.fillRect(cx, cy - 2, 1, 5);
  }
}

function crumbs(x, r, t) {
  if (RM) return;
  for (const b of r.world.blocks) {
    const age = (t - b.t) / 10; if (age < 0) break; if (age > 7) continue;
    const rr = rng(b.x * 131 + b.y * 17 + b.t), n = b.prev === "D" ? 22 : 9;
    for (let i = 0; i < n; i++) {
      const vx = (rr() - .5) * 2.6, vy = -rr() * 2.2 - (b.prev === "D" ? .8 : 0);
      const px = (b.x + .5) * CELL + vx * age * CELL * .35, py = (b.y + .5) * CELL + (vy * age + .25 * age * age) * CELL * .35;
      x.globalAlpha = Math.max(0, 1 - age / 7);
      x.fillStyle = b.prev === "D" && i % 3 === 0 ? "#ffffff" : CRUMB[b.prev] || "#888";
      x.fillRect(Math.round(px), Math.round(py), 2, 2);
    }
  }
  x.globalAlpha = 1;
}

function pickaxe(x, px, py, face, prog) {
  const ang = (-1.25 + Math.sin(prog * Math.PI * 2) * .9) * face;
  x.save(); x.translate(px + (face > 0 ? 12 : 4), py + 9); x.rotate(ang);
  x.fillStyle = "#7a5230"; x.fillRect(-1, -9, 2, 9);
  x.fillStyle = "#a9b1c2"; x.fillRect(-5, -11, 10, 2); x.fillRect(-6, -10, 2, 2); x.fillRect(4, -10, 2, 2);
  x.restore();
}

function bubble(x, px, py, prog, t) {
  const bx = Math.round(px - 2), by = Math.round(py - 15);
  x.fillStyle = "#14121c"; x.fillRect(bx, by, 20, 11); x.fillRect(bx + 4, by + 11, 3, 2);
  x.fillStyle = "#f2b35b"; x.fillRect(bx + 1, by + 9, Math.max(1, Math.round(18 * prog)), 1);
  for (let k = 0; k < 3; k++) {
    const on = RM ? true : Math.floor(t * 4 + k) % 3 !== 0;
    x.fillStyle = on ? "#f2b35b" : "#5a4630"; x.fillRect(bx + 4 + k * 5, by + 4, 2, 2);
  }
}

function blocksUpTo(r, t) {
  const bl = r.world.blocks; let lo = 0, hi = bl.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (bl[m].t <= t) lo = m + 1; else hi = m; }
  return lo;
}

function draw(L, t, amb, cur) {
  const r = L.r, x = L.ctx, ti = Math.floor(t);
  if (!L.world) {
    L.world = document.createElement("canvas"); L.world.width = r.world.grid0[0].length * CELL; L.world.height = r.world.grid0.length * CELL;
    L.wctx = L.world.getContext("2d"); L.nb = -1;
    const cr = rng(r.seed * 7 + 3);
    L.clouds = [0, 1, 2].map(() => [Math.floor(cr() * (L.world.width - 60)) + 10, 8 + Math.floor(cr() * 22), 26 + Math.floor(cr() * 30)]);
  }
  const nb = blocksUpTo(r, ti);
  if (nb !== L.nb) {
    L.grid = r.world.grid0.map(s => s.split(""));
    for (let k = 0; k < nb; k++) { const b = r.world.blocks[k]; L.grid[b.y][b.x] = b.b; }
    paintWorld(L, L.grid); L.nb = nb;
  }
  const i0 = A.frameIdx(r, t), f0 = r.frames[i0], fp = r.frames[Math.max(0, i0 - 1)];
  const k = RM ? 1 : Math.min(1, (t - f0.t) / 8);            // glide 80 ms into each new position
  const lerp = (a, b) => a + (b - a) * k;
  const mx = lerp(fp.p[0], f0.p[0]), my = lerp(fp.p[1], f0.p[1]), face = r.face[i0];
  const sameZ = fp.z.length === f0.z.length;
  const zpos = f0.z.map((z, j) => sameZ ? [lerp(fp.z[j][0], z[0]), lerp(fp.z[j][1], z[1])] : [z[0], z[1]]);
  const GW = L.grid[0].length, GH = L.grid.length;
  const camX = Math.round(Math.min(Math.max(mx - (VW - 1) / 2, 0), GW - VW) * CELL);
  const camY = Math.round(Math.min(Math.max(my - (VH - 1) / 2 + 1, 0), GH - VH) * CELL);
  x.save(); x.translate(-camX, -camY);
  x.drawImage(L.world, 0, 0);
  zpos.forEach(([ax, ay], j) => x.drawImage(ax > mx ? SPR.zombieL : SPR.zombie, Math.round(ax * CELL), Math.round(ay * CELL + (RM ? 0 : Math.sin(amb * 3 + j) * .8))));
  const px = Math.round(mx * CELL), py = Math.round(my * CELL);
  x.drawImage(face > 0 ? SPR.miner : SPR.minerL, px, py);
  if (cur && t >= cur.t_land && t < cur.t_land + 8 && /broke|cracked|hit|killed|DIAMOND|dig/.test(cur.result))
    pickaxe(x, px, py, face, (t - cur.t_land) / 8);
  lighting(x, L.grid, mx, my, amb);
  glints(x, L.grid, amb);
  zpos.forEach(([zx, zy]) => {                               // zombie eyes glow through the dark
    const ax = Math.round(zx * CELL), ay = Math.round(zy * CELL), left = zx > mx;
    x.fillStyle = "rgba(232,99,79,.35)"; x.fillRect(ax + (left ? 4 : 5), ay + 2, 8, 3);
    x.fillStyle = "#ff8a6b"; x.fillRect(ax + (left ? 5 : 6), ay + 3, 1, 1); x.fillRect(ax + (left ? 9 : 10), ay + 3, 1, 1);
  });
  crumbs(x, r, t);
  for (let j = i0; j > 0 && r.frames[j].t > t - 50; j--) {   // hit flash + floating -1 for half a second
    if (r.frames[j].hp < r.frames[j - 1].hp) {
      const age = (t - r.frames[j].t) / 10;
      x.fillStyle = `rgba(232,99,79,${(.55 * (1 - age / 5)).toFixed(2)})`; x.fillRect(px + 3, py + 1, 10, 15);
      x.font = "600 9px 'JetBrains Mono', monospace"; x.fillStyle = `rgba(255,170,150,${(1 - age / 5).toFixed(2)})`;
      x.fillText("-1", px + 12, py - 2 - age * 2);
      break;
    }
  }
  if (cur && t < cur.t_land && cur.t_land - cur.t_obs >= 10) bubble(x, px, py, (t - cur.t_obs) / (cur.t_land - cur.t_obs), amb);
  x.restore();
  hud(x, L.grid, mx, my, camX / CELL, camY / CELL, amb);
  return f0;
}

/* ---------- camera HUD: depth meter + arrow to the nearest diamond ---------- */
function hud(x, g, mx, my, cx, cy, t) {
  let best = null;
  for (let y = 0; y < g.length; y++) for (let i = 0; i < g[0].length; i++)
    if (g[y][i] === "D") { const d = Math.abs(i - mx) + Math.abs(y - my); if (!best || d < best.d) best = { i, y, d }; }
  const Hc = VH * CELL, Wc = VW * CELL, mH = Hc - 16, top = 8, mxp = Wc - 7;
  x.fillStyle = "rgba(12,10,18,.7)"; x.fillRect(mxp - 2, top - 2, 6, mH + 4);
  x.fillStyle = "#3a3549"; x.fillRect(mxp, top, 2, mH);
  const Y = yy => top + Math.round(yy / (g.length - 1) * mH);
  x.fillStyle = "#5fa548"; x.fillRect(mxp - 1, Y(SURFACE), 4, 1);
  if (best) { const dy = Y(best.y); x.fillStyle = "#62e3ea"; x.fillRect(mxp - 2, dy - 1, 6, 3); x.fillStyle = "#e9feff"; x.fillRect(mxp, dy, 1, 1); }
  x.fillStyle = "#f2b35b"; x.fillRect(mxp - 2, Y(my) - 1, 6, 3);
  if (!best) return;
  const bx = best.i - cx, by = best.y - cy;                 // diamond position in view cells
  if (bx >= 0 && bx < VW && by >= 0 && by < VH) return;
  const ang = Math.atan2(by + .5 - (my - cy + .5), bx + .5 - (mx - cx + .5));
  const ex = Math.min(Math.max(Wc / 2 + Math.cos(ang) * Wc, 10), Wc - 18), ey = Math.min(Math.max(Hc / 2 + Math.sin(ang) * Hc, 10), Hc - 10);
  const bob = RM ? 0 : Math.sin(t * 2) * 1.5;
  x.save(); x.translate(Math.round(ex + Math.cos(ang) * bob), Math.round(ey + Math.sin(ang) * bob)); x.rotate(ang);
  x.fillStyle = "rgba(12,10,18,.75)"; x.fillRect(-7, -6, 13, 12);
  x.fillStyle = "#62e3ea"; x.beginPath(); x.moveTo(5, 0); x.lineTo(-3, -4); x.lineTo(-3, 4); x.fill();
  x.restore();
}


/* ---------- replay prep and panel bits ---------- */
function prep(r) {
  const g = r.world.grid0.map(s => s.split(""));
  for (const b of r.world.blocks) { b.prev = g[b.y][b.x]; g[b.y][b.x] = b.b; }
  let face = 1;
  r.face = r.frames.map((f, i) => { if (i) { const dx = f.p[0] - r.frames[i - 1].p[0]; if (dx) face = Math.sign(dx); } return face; });
}
function vitals(r, f) {
  let h = ""; for (let k = 0; k < 5; k++) h += HEARTS[f.hp >= 2 * k + 2 ? "full" : f.hp === 2 * k + 1 ? "half" : "empty"];
  return `<div class="hearts" aria-label="Health ${f.hp} of 10">${h}</div><div class="food" aria-label="Food ${f.food}">${APPLE.repeat(f.food)}</div>`;
}

A.games.mining = {
  title: "2D Minecraft", tagline: "Dig to the diamond before the zombies get you.",
  levels: [
    { name: "Warm-up", desc: "No zombies. The diamond sits a few blocks down." },
    { name: "Survive", desc: "Zombies hunt the miner. Every slow turn costs health." },
    { name: "Renamed tools", desc: "Same game, new controls: the tools and their parameters have different names." },
    { name: "Renamed values", desc: "Directions are renamed too. The game still says left, right, up and down." },
  ],
  words: { win: "DIAMOND", dead: "DOWN", timeout: "OUT OF TIME" },
  size: [VW * CELL, VH * CELL], pixelated: true,
  prep, draw, vitals,
  meter: (r, f) => ["depth", String(Math.max(0, f.p[1] - SURFACE + 1))],
};
})();
