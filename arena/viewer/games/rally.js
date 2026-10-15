/* ActionMan Arena, rally: an inked roadbook on survey paper, follow camera, racing-line trail. Registers ARENA.games.rally. */
(() => {
"use strict";
const A = window.ARENA, RM = A.RM, rng = A.rng;
const W = 480, H = 330, CRASH = 150, MAXCR = 5;
const PAPER = "#ece2cb", CARD = "#f6efde", INK = "#2a221c", SEPIA = "120,86,50", RED = "#b8352a", LINE = "#2348a3";
const STRIPE = { actionman: "#14939c", qwen: "#d4891c", openai: "#6f55c4", scripted: "#b8352a", random: "#6b6560" };
const ROAD = 4.6, A_LINE = 2.5;                 // road half-width and racing-line swing, metres

/* ---------- noise ---------- */
function hash(i, j, s) {
  let h = Math.imul(i, 374761393) + Math.imul(j, 668265263) + Math.imul(s, 1013904223) | 0;
  h = Math.imul(h ^ h >>> 13, 1274126177); return ((h ^ h >>> 16) >>> 0) / 4294967296;
}
function vnoise(x, y, s) {
  const i = Math.floor(x), j = Math.floor(y), fx = x - i, fy = y - j, u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
  const a = hash(i, j, s), b = hash(i + 1, j, s), c = hash(i, j + 1, s), d = hash(i + 1, j + 1, s);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
function fbm(x, y, s, f = 1 / 240) {
  let a = 0, amp = .55;
  for (let k = 0; k < 4; k++) { a += amp * vnoise(x * f, y * f, s + k * 31); amp *= .5; f *= 2.1; }
  return a;
}
const ease = u => u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u);
const lerp = (a, b, k) => a + (b - a) * k;
const angLerp = (a, b, k) => a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * k;

/* ---------- stage geometry (screen axes: y down) ---------- */
function at(g, s) {
  s = Math.max(0, Math.min(g.S[g.n - 1], s));
  let lo = 0, hi = g.n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (g.S[m] <= s) lo = m; else hi = m; }
  const k = (s - g.S[lo]) / Math.max(1e-6, g.S[hi] - g.S[lo]), dx = g.X[hi] - g.X[lo], dy = g.Y[hi] - g.Y[lo];
  return { x: g.X[lo] + dx * k, y: g.Y[lo] + dy * k, h: Math.atan2(dy, dx) };
}
function offset(cs, s) {                          // racing line: outside, apex, outside; late calls run wide and wobble
  let o = 0;
  for (const c of cs) {
    const u = s - c.s; if (u < -45 || u > c.len + 45) continue;
    const sg = c.sig, k = c.k, out = -sg * A_LINE * (1 - k), apex = sg * A_LINE * (1 - .55 * k), exit = -sg * A_LINE * (.55 + .95 * k);
    if (u < 0) o += out * ease((u + 45) / 45);
    else if (u < c.len * .55) o += lerp(out, apex, ease(u / (c.len * .55)));
    else if (u < c.len) o += lerp(apex, exit, ease((u - c.len * .55) / (c.len * .45)));
    else { const v = (u - c.len) / 45; o += exit * (1 - ease(v)) + k * sg * 1.6 * Math.sin(v * Math.PI * 2.2) * (1 - v); }
  }
  return Math.max(-ROAD - 1.2, Math.min(ROAD + 1.2, o));
}

function prep(r) {
  const w = r.world, n = w.pts.length, X = new Float64Array(n), Y = new Float64Array(n), S = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    X[i] = w.pts[i][0]; Y[i] = -w.pts[i][1];
    if (i) S[i] = S[i - 1] + Math.hypot(X[i] - X[i - 1], Y[i] - Y[i - 1]);
  }
  const fix = w.len / S[n - 1]; for (let i = 0; i < n; i++) S[i] *= fix;
  const g = r.geo = { X, Y, S, n };
  const cs = r.cs = w.corners.map(c => ({ ...c, sig: c.dir === "left" ? 1 : -1, k: 0 }));
  r.crashes = [];
  for (const e of w.events) {
    if (e.crash) { cs[e.c].k = 1; r.crashes.push({ t: e.t, c: e.c, why: e.crash }); }
    else cs[e.c].k = Math.max(0, Math.min(1, 1 - e.m / 22));
  }
  // car pose every tick: x, y, heading, mode (0 drive, 1 slide, 2 rest, 3 tow)
  const N = r.end + 1, P = r.pose = new Float32Array(N * 4), fr = r.frames, sAt = new Float32Array(N), vAt = new Float32Array(N);
  for (let t = 0, i = 0; t < N; t++) {
    while (i + 1 < fr.length && fr[i + 1].t <= t) i++;
    const a = fr[i], b = fr[Math.min(i + 1, fr.length - 1)], k = b.t > a.t ? Math.min(1, (t - a.t) / (b.t - a.t)) : 0;
    sAt[t] = b.cr > a.cr ? a.s : lerp(a.s, b.s, k); vAt[t] = lerp(a.v, b.v, k);
  }
  const crashAt = new Int32Array(N).fill(-1);
  r.crashes.forEach((c, j) => {
    for (let t = c.t; t < Math.min(N, c.t + CRASH); t++) crashAt[t] = j;
    const cn = cs[c.c], p = at(g, cn.s), o = offset(cs, cn.s - .5), v0 = vAt[Math.max(0, c.t - 1)];
    c.x0 = p.x + Math.sin(p.h) * o; c.y0 = p.y - Math.cos(p.h) * o; c.h0 = p.h;
    const D = Math.max(14, Math.min(40, 12 + v0 * .28)), drift = (c.why === "speed" ? .5 : .38) * cn.sig;
    c.x1 = c.x0 + Math.cos(p.h - drift) * D; c.y1 = c.y0 + Math.sin(p.h - drift) * D;
    c.h1 = p.h - cn.sig * (c.why === "speed" ? 2.6 : 1.9); c.D = D; c.drift = drift;
  });
  for (let t = 0; t < N; t++) {
    let x, y, h, m = 0;
    const j = crashAt[t];
    if (j >= 0) {
      const c = r.crashes[j], u = t - c.t;
      if (u < 55) { const e = 1 - (1 - u / 55) ** 2.2; x = lerp(c.x0, c.x1, e); y = lerp(c.y0, c.y1, e); h = angLerp(c.h0, c.h1, e); m = 1; }
      else if (u < 115) { x = c.x1; y = c.y1; h = c.h1; m = 2; }
      else { const e = ease((u - 115) / 35); x = lerp(c.x1, c.x0, e); y = lerp(c.y1, c.y0, e); h = angLerp(c.h1, c.h0, e); m = 3; }
    } else {
      const s = sAt[t], p = at(g, s), o = offset(cs, s), d = (offset(cs, s + 1) - offset(cs, s - 1)) / 2;
      x = p.x + Math.sin(p.h) * o; y = p.y - Math.cos(p.h) * o; h = p.h - Math.atan(d);
    }
    P[t * 4] = x; P[t * 4 + 1] = y; P[t * 4 + 2] = h; P[t * 4 + 3] = m;
  }
  // camera: frame the car and the next corner, smoothed over about a second of replay
  const cx = new Float32Array(N), cy = new Float32Array(N), cz = new Float32Array(N);
  for (let t = 0, ci = 0; t < N; t++) {
    const s = sAt[t];
    while (ci < cs.length && cs[ci].s + cs[ci].len * .5 < s) ci++;
    const q = at(g, ci < cs.length ? Math.min(cs[ci].s + cs[ci].len * .5, s + 190) : Math.min(w.len + 30, s + 120));
    const x = P[t * 4], y = P[t * 4 + 1], dx = Math.abs(q.x - x) + 70, dy = Math.abs(q.y - y) + 60;
    cx[t] = (x + q.x) / 2; cy[t] = (y + q.y) / 2;
    cz[t] = Math.max(1.3, Math.min(W / dx, H / dy, lerp(3.1, 1.9, Math.min(1, vAt[t] / 100))));
  }
  const R = 55, sm = a => {
    const c = new Float64Array(N + 1); for (let t = 0; t < N; t++) c[t + 1] = c[t] + a[t];
    return a.map((_, t) => { const lo = Math.max(0, t - R), hi = Math.min(N, t + R + 1); return (c[hi] - c[lo]) / (hi - lo); });
  };
  r.cam = [sm(cx), sm(cy), sm(cz)];
  for (let t = 0; t < N; t++) {                    // keep the car inside the frame whatever the smoothing did
    const z = r.cam[2][t], mx = W / 2 / z - 34 / z, my = H / 2 / z - 34 / z;
    r.cam[0][t] = Math.max(P[t * 4] - mx, Math.min(P[t * 4] + mx, r.cam[0][t]));
    r.cam[1][t] = Math.max(P[t * 4 + 1] - my, Math.min(P[t * 4 + 1] + my, r.cam[1][t]));
  }
  r.sAt = sAt; r.vAt = vAt;
  r.stripe = STRIPE[Object.keys(STRIPE).find(k => r.agent.startsWith(k))] || "#2f6f8f";
}

/* ---------- the map: paper, contours, forest, road, roadbook ink (once per stage) ---------- */
const MAPS = new Map();
let GRAIN = null;
function grain() {
  if (GRAIN) return GRAIN;
  const c = document.createElement("canvas"); c.width = c.height = 256;
  const x = c.getContext("2d"), im = x.createImageData(256, 256), r = rng(7);
  for (let i = 0; i < im.data.length; i += 4) {
    const v = (r() - .5) * 26; im.data[i] = 236 + v; im.data[i + 1] = 226 + v; im.data[i + 2] = 203 + v * .9; im.data[i + 3] = 255;
  }
  x.putImageData(im, 0, 0);
  x.lineWidth = .6;
  for (let i = 0; i < 70; i++) {                   // paper fibres
    const a = r() * 6.28, l = 4 + r() * 10, px = r() * 256, py = r() * 256;
    x.strokeStyle = r() < .5 ? "rgba(255,250,236,.55)" : "rgba(150,120,80,.14)";
    x.beginPath(); x.moveTo(px, py); x.quadraticCurveTo(px + Math.cos(a + .6) * l / 2, py + Math.sin(a + .6) * l / 2, px + Math.cos(a) * l, py + Math.sin(a) * l); x.stroke();
  }
  return GRAIN = c;
}

function roadPath(g, jit, seed, from = 0, to = 1e9) {
  const p = new Path2D(); let first = true;
  for (let i = 0; i < g.n; i++) {
    if (g.S[i] < from || g.S[i] > to) continue;
    const j = jit ? (vnoise(g.S[i] / 9, 3.3, seed) - .5) * jit : 0, i1 = Math.min(g.n - 1, i + 1), i0 = Math.max(0, i - 1);
    const h = Math.atan2(g.Y[i1] - g.Y[i0], g.X[i1] - g.X[i0]), x = g.X[i] + Math.sin(h) * j, y = g.Y[i] - Math.cos(h) * j;
    if (first) { p.moveTo(x, y); first = false; } else p.lineTo(x, y);
  }
  return p;
}

function chequer(x, g, s, label, seed) {
  const p = at(g, s), c = Math.cos(p.h), sn = Math.sin(p.h);
  x.save(); x.translate(p.x, p.y); x.rotate(p.h);
  const q = ROAD * 2 / 6;
  for (let i = 0; i < 6; i++) for (let k = 0; k < 2; k++) { x.fillStyle = (i + k) % 2 ? INK : "#f8f2e2"; x.fillRect(k * q - q, -ROAD + i * q, q, q); }
  x.strokeStyle = INK; x.lineWidth = .5; x.strokeRect(-q, -ROAD, 2 * q, 2 * ROAD);
  x.restore();
  const lx = p.x + sn * 15 - c * 6, ly = p.y - c * 15 - sn * 6;
  x.font = "700 6px 'JetBrains Mono', monospace"; const tw = x.measureText(label).width;
  x.fillStyle = CARD; x.fillRect(lx - tw / 2 - 2.5, ly - 5, tw + 5, 7.6);
  x.strokeStyle = INK; x.lineWidth = .55; x.strokeRect(lx - tw / 2 - 2.5, ly - 5, tw + 5, 7.6);
  x.fillStyle = INK; x.textAlign = "center"; x.fillText(label, lx, ly + .9); x.textAlign = "left";
  x.beginPath(); x.moveTo(p.x + sn * (ROAD + 1), p.y - c * (ROAD + 1)); x.lineTo(lx, ly + 2.6); x.lineWidth = .4; x.stroke();
}

function makeMap(r) {
  const g = r.geo, cs = r.cs, w = r.world, seed = r.level * 977 + r.seed * 131 + 11;
  let minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
  for (let i = 0; i < g.n; i++) { minx = Math.min(minx, g.X[i]); maxx = Math.max(maxx, g.X[i]); miny = Math.min(miny, g.Y[i]); maxy = Math.max(maxy, g.Y[i]); }
  const pad = 200, x0 = minx - pad, y0 = miny - pad, wm = maxx - minx + 2 * pad, hm = maxy - miny + 2 * pad;
  const BS = Math.min(2.6, Math.sqrt(12e6 / (wm * hm)));
  const c = document.createElement("canvas"); c.width = Math.ceil(wm * BS); c.height = Math.ceil(hm * BS);
  const x = c.getContext("2d");
  x.fillStyle = x.createPattern(grain(), "repeat"); x.fillRect(0, 0, c.width, c.height);
  x.setTransform(BS, 0, 0, BS, -x0 * BS, -y0 * BS);
  x.lineCap = x.lineJoin = "round";
  const R = rng(seed);
  for (let i = 0; i < 7; i++) {                   // age stains
    const sx = x0 + R() * wm, sy = y0 + R() * hm, rad = 60 + R() * 140, gr = x.createRadialGradient(sx, sy, 0, sx, sy, rad);
    gr.addColorStop(0, "rgba(170,125,60,.075)"); gr.addColorStop(1, "rgba(170,125,60,0)"); x.fillStyle = gr; x.fillRect(sx - rad, sy - rad, rad * 2, rad * 2);
  }
  x.strokeStyle = "rgba(60,100,140,.13)"; x.lineWidth = .5;      // survey grid, 200 m
  x.beginPath();
  for (let gx = Math.ceil(x0 / 200) * 200; gx < x0 + wm; gx += 200) { x.moveTo(gx, y0); x.lineTo(gx, y0 + hm); }
  for (let gy = Math.ceil(y0 / 200) * 200; gy < y0 + hm; gy += 200) { x.moveTo(x0, gy); x.lineTo(x0 + wm, gy); }
  x.stroke();
  // contours by marching squares on fbm terrain
  const G = 7, gw = Math.ceil(wm / G) + 1, gh = Math.ceil(hm / G) + 1, ht = new Float32Array(gw * gh);
  for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) ht[j * gw + i] = fbm(x0 + i * G, y0 + j * G, seed);
  const STEP = .032, SEG = [[], [3, 2], [2, 1], [3, 1], [0, 1], [0, 1, 3, 2], [0, 2], [3, 0], [3, 0], [0, 2], [0, 3, 2, 1], [0, 1], [3, 1], [2, 1], [3, 2], []];
  for (let lv = 1; lv < 1 / STEP; lv++) {
    const L = lv * STEP, idx = lv % 5 === 0, p = new Path2D();
    for (let j = 0; j < gh - 1; j++) for (let i = 0; i < gw - 1; i++) {
      const a = ht[j * gw + i], b = ht[j * gw + i + 1], cc = ht[(j + 1) * gw + i + 1], d = ht[(j + 1) * gw + i];
      const id = (a > L) << 3 | (b > L) << 2 | (cc > L) << 1 | (d > L);
      if (id === 0 || id === 15) continue;
      const px = x0 + i * G, py = y0 + j * G, e = [
        [px + G * (L - a) / (b - a), py], [px + G, py + G * (L - b) / (cc - b)], [px + G * (L - d) / (cc - d), py + G], [px, py + G * (L - a) / (d - a)]];
      const sg = SEG[id];
      for (let k = 0; k < sg.length; k += 2) { p.moveTo(e[sg[k]][0], e[sg[k]][1]); p.lineTo(e[sg[k + 1]][0], e[sg[k + 1]][1]); }
    }
    x.strokeStyle = `rgba(${SEPIA},${idx ? .42 : .22})`; x.lineWidth = idx ? .9 : .5; x.stroke(p);
  }
  // forest stipple where a second noise field is high
  const tp = new Path2D();
  for (let ty = y0; ty < y0 + hm; ty += 6.5) for (let tx = x0; tx < x0 + wm; tx += 6.5) {
    if (fbm(tx, ty, seed + 5, 1 / 170) < .56) continue;
    const jx = tx + (hash(tx | 0, ty | 0, seed) - .5) * 5, jy = ty + (hash(ty | 0, tx | 0, seed) - .5) * 5, rr = .9 + hash(tx | 0, 7, seed) * .7;
    tp.moveTo(jx + rr, jy); tp.arc(jx, jy, rr, 0, 6.283);
  }
  x.strokeStyle = "rgba(74,98,58,.42)"; x.lineWidth = .4; x.stroke(tp);
  // spot heights away from the road
  x.font = "500 4.2px 'JetBrains Mono', monospace";
  for (let k = 0, tries = 0; k < 9 && tries < 80; tries++) {
    const sx = x0 + 40 + R() * (wm - 80), sy = y0 + 40 + R() * (hm - 80);
    let near = false; for (let i = 0; i < g.n; i += 6) if (Math.hypot(g.X[i] - sx, g.Y[i] - sy) < 45) { near = true; break; }
    if (near) continue; k++;
    x.fillStyle = `rgba(${SEPIA},.75)`; x.beginPath(); x.arc(sx, sy, .9, 0, 6.283); x.fill();
    x.fillText(String(Math.round(300 + fbm(sx, sy, seed) * 400)), sx + 2, sy + 1.5);
  }
  // the road: cleared verge, inked edges (a slightly unsteady hand), pale surface, centre dashes
  const clean = roadPath(g, 0, seed), wob = roadPath(g, .9, seed);
  x.strokeStyle = "rgba(236,226,203,.86)"; x.lineWidth = 24; x.stroke(clean);
  x.strokeStyle = INK; x.lineWidth = ROAD * 2 + 1.3; x.stroke(wob);
  x.strokeStyle = "rgba(42,34,28,.35)"; x.lineWidth = ROAD * 2 + 2.2; x.stroke(roadPath(g, 1.4, seed + 9));
  x.strokeStyle = INK; x.lineWidth = ROAD * 2 + 1.3; x.stroke(wob);
  x.strokeStyle = "#f7f0de"; x.lineWidth = ROAD * 2 - .6; x.stroke(clean);
  x.setLineDash([3.2, 4.2]); x.strokeStyle = "rgba(42,34,28,.32)"; x.lineWidth = .4; x.stroke(clean); x.setLineDash([]);
  // roadbook ink per corner: hatching outside tight corners, a numbered roundel on the outside
  for (const cn of cs) {
    if (cn.sev <= 3) {
      x.strokeStyle = cn.sev <= 2 ? "rgba(184,53,42,.8)" : "rgba(42,34,28,.5)"; x.lineWidth = .55; x.beginPath();
      for (let s = cn.s - 4; s <= cn.s + cn.len + 4; s += 2.4) {
        const p = at(g, s), nx = -Math.sin(p.h) * cn.sig, ny = Math.cos(p.h) * cn.sig;   // outward normal
        x.moveTo(p.x + nx * (ROAD + 1.3), p.y + ny * (ROAD + 1.3)); x.lineTo(p.x + nx * (ROAD + 3.6), p.y + ny * (ROAD + 3.6));
      }
      x.stroke();
    }
    const p = at(g, cn.s + cn.len / 2), nx = -Math.sin(p.h) * cn.sig, ny = Math.cos(p.h) * cn.sig;
    const lx = p.x + nx * 15, ly = p.y + ny * 15, col = cn.sev <= 2 ? RED : INK;
    x.strokeStyle = col; x.lineWidth = .45; x.beginPath(); x.moveTo(p.x + nx * (ROAD + 4), p.y + ny * (ROAD + 4)); x.lineTo(lx - nx * 4.6, ly - ny * 4.6); x.stroke();
    x.fillStyle = CARD; x.beginPath(); x.arc(lx, ly, 4.4, 0, 6.283); x.fill();
    x.lineWidth = .7; x.stroke();
    if (cn.sev === 1) { x.lineWidth = .4; x.beginPath(); x.arc(lx, ly, 5.6, 0, 6.283); x.stroke(); }
    x.fillStyle = col; x.font = "700 6px 'JetBrains Mono', monospace"; x.textAlign = "center";
    x.fillText(String(cn.sev), lx, ly + 2.1);
    x.font = "600 2.7px 'Instrument Sans', sans-serif"; x.fillText(cn.dir === "left" ? "L" : "R", lx + 4.6, ly - 3.6); x.textAlign = "left";
  }
  // kilometre posts
  x.font = "600 4.6px 'JetBrains Mono', monospace";
  for (let s = 250; s < w.len - 60; s += 250) {
    const p = at(g, s), nx = Math.sin(p.h), ny = -Math.cos(p.h);
    x.strokeStyle = INK; x.lineWidth = .7; x.beginPath(); x.moveTo(p.x - nx * (ROAD + 2), p.y - ny * (ROAD + 2)); x.lineTo(p.x + nx * (ROAD + 6), p.y + ny * (ROAD + 6)); x.stroke();
    x.fillStyle = INK; x.fillText((s / 1000).toFixed(2), p.x + nx * (ROAD + 8) - 3, p.y + ny * (ROAD + 8) + 1.5);
  }
  chequer(x, g, 0, "START", seed);
  chequer(x, g, w.len, "FINISH", seed);
  const p0 = at(g, 0);
  x.fillStyle = INK; x.font = "700 7px 'Instrument Sans', sans-serif";
  x.fillText(`SS${r.seed + 1}`, p0.x - 30, p0.y + 22);
  x.font = "500 4.2px 'JetBrains Mono', monospace"; x.fillStyle = `rgba(${SEPIA},.95)`;
  x.fillText(`${(w.len / 1000).toFixed(2)} km · L${r.level}`, p0.x - 30, p0.y + 28);
  // the inset: whole stage, fitted into a small card
  const IW = 104, IH = 64, k = Math.min((IW - 14) / (maxx - minx), (IH - 14) / (maxy - miny));
  const ox = (IW - (maxx - minx) * k) / 2 - minx * k, oy = (IH - (maxy - miny) * k) / 2 - miny * k, ip = new Path2D();
  for (let i = 0; i < g.n; i += 2) (i ? ip.lineTo : ip.moveTo).call(ip, g.X[i] * k + ox, g.Y[i] * k + oy);
  return { c, x0, y0, wm, hm, inset: { ip, k, ox, oy, IW, IH } };
}
function mapFor(r) {
  const key = `${r.level}-${r.seed}`;
  if (!MAPS.has(key)) { if (MAPS.size > 2) MAPS.delete(MAPS.keys().next().value); MAPS.set(key, makeMap(r)); }
  return MAPS.get(key);
}

/* ---------- live layers ---------- */
function trail(r, x, t, z) {
  const P = r.pose, C = 100;
  r.chunks ||= [];
  const full = Math.floor(t / C);
  for (let k = r.chunks.length; k < full; k++) r.chunks.push(trailPath(P, k * C, (k + 1) * C));
  x.strokeStyle = LINE; x.lineCap = x.lineJoin = "round";
  x.globalAlpha = .2; x.lineWidth = 4 / z;
  for (let k = 0; k < full; k++) x.stroke(r.chunks[k]);
  const tail = trailPath(P, full * C, t); x.stroke(tail);
  x.globalAlpha = .9; x.lineWidth = 1.5 / z;
  for (let k = 0; k < full; k++) x.stroke(r.chunks[k]);
  x.stroke(tail); x.globalAlpha = 1;
}
function trailPath(P, a, b) {
  const p = new Path2D(); let pen = false;
  for (let t = a; t <= b && t * 4 < P.length; t++) {
    if (P[t * 4 + 3] !== 0) { pen = false; continue; }
    if (pen) p.lineTo(P[t * 4], P[t * 4 + 1]); else { p.moveTo(P[t * 4], P[t * 4 + 1]); pen = true; }
  }
  return p;
}

function crashInk(r, x, t, z, amb) {
  for (const c of r.crashes) {
    const u = t - c.t; if (u < 0) break;
    const e = 1 - (1 - Math.min(1, u / 55)) ** 2.2, ex = lerp(c.x0, c.x1, e), ey = lerp(c.y0, c.y1, e);
    x.strokeStyle = "rgba(34,26,20,.8)"; x.lineWidth = 1.4 / z;            // skid marks: two tyre tracks
    for (const sd of [-1, 1]) {
      x.beginPath();
      for (let q = 0; q <= 12; q++) {
        const f = q / 12 * e, px = lerp(c.x0, c.x1, f), py = lerp(c.y0, c.y1, f), hh = angLerp(c.h0, c.h1, f);
        const ox = -Math.sin(hh) * sd * 1.1, oy = Math.cos(hh) * sd * 1.1;
        q ? x.lineTo(px + ox, py + oy) : x.moveTo(px + ox, py + oy);
      }
      x.stroke();
    }
    if (u >= 30) {                                                          // red-ink cross where it stopped
      const s = 11 / z * Math.min(1, (u - 30) / 12);
      x.strokeStyle = RED; x.lineWidth = 2.6 / z; x.beginPath();
      x.moveTo(c.x1 - s, c.y1 - s * .9); x.lineTo(c.x1 + s, c.y1 + s * 1.1); x.moveTo(c.x1 + s * 1.05, c.y1 - s); x.lineTo(c.x1 - s * .95, c.y1 + s); x.stroke();
    }
    if (!RM && u < 170) {                                                   // dust
      const R = rng(c.t * 13 + 5), a = u / 170;
      for (let q = 0; q < 16; q++) {
        const f = R(), px = lerp(c.x0, ex, f) + (R() - .5) * 6 * (1 + a * 2), py = lerp(c.y0, ey, f) + (R() - .5) * 6 * (1 + a * 2);
        if (f > e) continue;
        x.fillStyle = `rgba(150,118,78,${(.3 * (1 - a) * (.5 + R() * .5)).toFixed(3)})`;
        x.beginPath(); x.arc(px, py - a * 4, 1.6 + a * 6 * (.5 + R()), 0, 6.283); x.fill();
      }
    }
  }
}

function car(x, sx, sy, h, col, v, mode) {
  x.save(); x.translate(sx, sy); x.rotate(h); x.scale(1.15, 1.15);
  if (!RM && v > 55 && mode === 0) {                                        // speed streaks
    const k = Math.min(1, (v - 55) / 45);
    x.strokeStyle = `rgba(42,34,28,${(.35 * k).toFixed(3)})`; x.lineWidth = .8;
    for (const [oy, l] of [[-3, 12], [0, 18], [3, 10]]) { x.beginPath(); x.moveTo(-11, oy); x.lineTo(-11 - l * k, oy); x.stroke(); }
  }
  x.fillStyle = "rgba(20,14,8,.28)"; rr(x, -8.2, -3.2, 18, 9.2, 3); x.fill();               // shadow
  x.fillStyle = "#fbf8f1"; x.strokeStyle = INK; x.lineWidth = 1; rr(x, -9, -4.3, 18, 8.6, 2.6); x.fill(); x.stroke();
  x.fillStyle = col; x.fillRect(-9, -1.4, 18, 1.1); x.fillRect(-9, .3, 18, 1.1);             // twin stripes
  x.fillStyle = "#28313b"; x.beginPath(); x.moveTo(2.2, -3.4); x.lineTo(4.6, -2.8); x.lineTo(4.6, 2.8); x.lineTo(2.2, 3.4); x.closePath(); x.fill();
  x.fillStyle = "#3a434d"; x.fillRect(-5.6, -3.2, 1.6, 6.4);                                 // rear window
  x.fillStyle = "#f2b35b"; x.fillRect(8, -3.6, 1, 1.6); x.fillRect(8, 2, 1, 1.6);            // lamps
  x.fillStyle = INK; x.fillRect(-9.6, -4.3, 1.2, 8.6);                                       // spoiler
  x.restore();
}
function rr(x, a, b, w, h, r) { x.beginPath(); x.roundRect ? x.roundRect(a, b, w, h, r) : x.rect(a, b, w, h); }

function wrap(x, text, max) {
  const out = []; let line = "";
  for (const wd of text.split(" ")) { const tr = line ? line + " " + wd : wd; if (x.measureText(tr).width > max && line) { out.push(line); line = wd; } else line = tr; }
  if (line) out.push(line);
  return out;
}

function tulip(x, cx, cy, cn, sz) {             // roadbook tulip: dot in, arrow out along the corner
  x.strokeStyle = INK; x.fillStyle = INK; x.lineWidth = 2; x.lineCap = "round";
  if (!cn) {                                    // finish: a chequered flag
    const q = sz * .11; x.fillRect(cx - sz * .3, cy - sz * .36, 1.6, sz * .78);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) if ((i + j) % 2 === 0) x.fillRect(cx - sz * .3 + 1.6 + i * q, cy - sz * .36 + j * q, q, q);
    x.lineWidth = 1; x.strokeRect(cx - sz * .3 + 1.6, cy - sz * .36, 4 * q, 3 * q); return;
  }
  const ang = { 1: 175, 2: 110, 3: 82, 4: 58, 5: 38, 6: 20 }[cn.sev] * Math.PI / 180, sg = cn.sig;
  let px = cx, py = cy + sz * .42, h = -Math.PI / 2;
  x.beginPath(); x.arc(px, py, 2.4, 0, 6.283); x.fill();
  x.beginPath(); x.moveTo(px, py); py -= sz * .3; x.lineTo(px, py);
  const n = 14, seg = (cn.sev === 1 ? sz * .62 : sz * .5) / n;
  for (let i = 0; i < n; i++) { h -= sg * ang / n; px += Math.cos(h) * seg; py += Math.sin(h) * seg; x.lineTo(px, py); }
  px += Math.cos(h) * sz * .12; py += Math.sin(h) * sz * .12; x.lineTo(px, py); x.stroke();
  arrowHead(x, px, py, h);
}
function arrowHead(x, px, py, h) {
  x.beginPath(); x.moveTo(px + Math.cos(h) * 5, py + Math.sin(h) * 5);
  x.lineTo(px + Math.cos(h + 2.5) * 5, py + Math.sin(h + 2.5) * 5); x.lineTo(px + Math.cos(h - 2.5) * 5, py + Math.sin(h - 2.5) * 5); x.fill();
}

function card(x, r, s, cur, t, amb) {
  const cs = r.cs, ci = cs.findIndex(c => c.s + 2 > s), cn = ci < 0 ? null : cs[ci];
  const d = (cn ? cn.s : r.world.len) - s, bw = 196, bh = 62, X0 = W - bw - 10, Y0 = 10;
  x.save(); x.translate(X0, Y0); x.rotate(-.012);
  x.fillStyle = "rgba(10,6,2,.28)"; x.fillRect(3, 4, bw, bh);
  x.fillStyle = CARD; x.fillRect(0, 0, bw, bh); x.strokeStyle = INK; x.lineWidth = 1; x.strokeRect(.5, .5, bw - 1, bh - 1);
  x.strokeRect(6.5, 6.5, 49, 49);                                            // tulip box
  tulip(x, 31, 31, cn, 46);
  if (cn) { x.font = "700 13px 'JetBrains Mono', monospace"; x.fillStyle = cn.sev <= 2 ? RED : INK; x.fillText(String(cn.sev), cn.sig > 0 ? 44 : 10, 20); }
  x.font = "600 9px 'JetBrains Mono', monospace"; x.fillStyle = "rgba(42,34,28,.6)";
  x.fillText("NOTE", 64, 15);
  x.textAlign = "right"; x.fillText(cn ? `${Math.max(0, Math.round(d))} m` : "FINISH", bw - 8, 15); x.textAlign = "left";
  x.beginPath(); x.moveTo(64, 19.5); x.lineTo(bw - 8, 19.5); x.strokeStyle = "rgba(42,34,28,.25)"; x.lineWidth = .8; x.stroke();
  x.font = "600 12.5px 'Instrument Sans', system-ui, sans-serif"; x.fillStyle = INK;
  const lines = wrap(x, cur ? cur.obs : "Stage start.", bw - 72).slice(0, 2);
  lines.forEach((ln, i) => x.fillText(ln, 64, 34 + i * 15));
  if (cur && t < cur.t_land && cur.t_land - cur.t_obs >= 4) {                // the driver is still thinking about this note
    const k = (t - cur.t_obs) / (cur.t_land - cur.t_obs);
    x.fillStyle = "rgba(217,146,46,.25)"; x.fillRect(1, bh - 4, bw - 2, 3);
    x.fillStyle = "#d9922e"; x.fillRect(1, bh - 4, (bw - 2) * k, 3);
  }
  x.restore();
}

function inset(x, M, r, P, t) {
  const I = M.inset, X0 = W - I.IW - 10, Y0 = H - I.IH - 10;
  x.save(); x.translate(X0, Y0);
  x.fillStyle = "rgba(10,6,2,.22)"; x.fillRect(3, 3, I.IW, I.IH);
  x.fillStyle = CARD; x.fillRect(0, 0, I.IW, I.IH); x.strokeStyle = INK; x.lineWidth = 1; x.strokeRect(.5, .5, I.IW - 1, I.IH - 1);
  x.strokeStyle = "rgba(42,34,28,.45)"; x.lineWidth = 1.6; x.lineJoin = x.lineCap = "round"; x.stroke(I.ip);
  const g = r.geo, s = r.sAt[Math.min(r.end, Math.floor(t))], dp = new Path2D();
  for (let i = 0; i < g.n && g.S[i] <= s; i += 2) (i ? dp.lineTo : dp.moveTo).call(dp, g.X[i] * I.k + I.ox, g.Y[i] * I.k + I.oy);
  x.strokeStyle = LINE; x.lineWidth = 1.8; x.stroke(dp);
  for (const c of r.crashes) if (c.t <= t) { const px = c.x0 * I.k + I.ox, py = c.y0 * I.k + I.oy; x.strokeStyle = RED; x.lineWidth = 1.3; x.beginPath(); x.moveTo(px - 2.5, py - 2.5); x.lineTo(px + 2.5, py + 2.5); x.moveTo(px + 2.5, py - 2.5); x.lineTo(px - 2.5, py + 2.5); x.stroke(); }
  const px = P[0] * I.k + I.ox, py = P[1] * I.k + I.oy;
  x.fillStyle = r.stripe; x.strokeStyle = INK; x.lineWidth = 1; x.beginPath(); x.arc(px, py, 2.8, 0, 6.283); x.fill(); x.stroke();
  x.restore();
}

let VIG = null;
function draw(L, t, amb, cur) {
  const r = L.r, x = L.ctx, M = mapFor(r), P = r.pose;
  const tt = Math.max(0, Math.min(r.end, t)), i0 = Math.floor(tt), i1 = Math.min(r.end, i0 + 1), k = tt - i0;
  const px = lerp(P[i0 * 4], P[i1 * 4], k), py = lerp(P[i0 * 4 + 1], P[i1 * 4 + 1], k), ph = angLerp(P[i0 * 4 + 2], P[i1 * 4 + 2], k), mode = P[i0 * 4 + 3];
  const cx = lerp(r.cam[0][i0], r.cam[0][i1], k), cy = lerp(r.cam[1][i0], r.cam[1][i1], k), z = lerp(r.cam[2][i0], r.cam[2][i1], k);
  x.setTransform(1, 0, 0, 1, 0, 0); x.fillStyle = PAPER; x.fillRect(0, 0, W, H);
  x.setTransform(z, 0, 0, z, W / 2 - cx * z, H / 2 - cy * z);
  x.imageSmoothingQuality = "high";
  x.drawImage(M.c, M.x0, M.y0, M.wm, M.hm);
  trail(r, x, tt, z);
  crashInk(r, x, tt, z, amb);
  x.setTransform(1, 0, 0, 1, 0, 0);
  const sx = (px - cx) * z + W / 2, sy = (py - cy) * z + H / 2;
  if (cur && tt < cur.t_land && cur.t_land - cur.t_obs >= 10) {           // thinking: an amber ring fills around the car
    const kk = (tt - cur.t_obs) / (cur.t_land - cur.t_obs);
    x.lineWidth = 2; x.strokeStyle = "rgba(217,146,46,.28)"; x.beginPath(); x.arc(sx, sy, 15, 0, 6.283); x.stroke();
    x.strokeStyle = "#d9922e"; x.beginPath(); x.arc(sx, sy, 15, -Math.PI / 2, -Math.PI / 2 + kk * 6.283); x.stroke();
  }
  car(x, sx, sy, ph, r.stripe, r.vAt[i0], mode);
  if (!VIG) { VIG = x.createRadialGradient(W / 2, H / 2, H * .45, W / 2, H / 2, W * .72); VIG.addColorStop(0, "rgba(40,24,10,0)"); VIG.addColorStop(1, "rgba(40,24,10,.32)"); }
  x.fillStyle = VIG; x.fillRect(0, 0, W, H);
  card(x, r, r.sAt[i0], cur, tt, amb);
  inset(x, M, r, [px, py], tt);
  return r.frames[A.frameIdx(r, t)];
}

/* ---------- panel bits ---------- */
const XS = on => `<svg viewBox="0 0 10 10" width="11" height="11" aria-hidden="true"><path d="M2 2L8 8M8 2L2 8" stroke="${on ? "#e8634f" : "#3a3549"}" stroke-width="2.2" stroke-linecap="round"/></svg>`;
function vitals(r, f) {
  const cr = f.cr || 0;
  return `<span>speed <b>${Math.round(f.v * 3.6)}</b> km/h</span><span>line <b>${f.ln || "none"}</b></span>` +
    `<span style="margin-left:auto;display:flex;gap:3px;align-items:center" aria-label="Crashes ${cr} of ${MAXCR}">${Array.from({ length: MAXCR }, (_, i) => XS(i < cr)).join("")}</span>`;
}

A.games.rally = {
  title: "Rally", tagline: "Drive a special stage from pace notes, flat out.",
  levels: [
    { name: "Warm-up", desc: "A short stage at 250 km/h: a handful of corners, wide gaps and one hairpin." },
    { name: "Full stage", desc: "360 km/h, flat-out complexes and a hairpin. A late call is a crash." },
    { name: "Renamed tools", desc: "Same stage style, new controls: the tools and their parameters have different names." },
    { name: "Renamed values", desc: "Left and right are renamed too. The notes still say left and right." },
  ],
  words: { win: "FINISH", dead: "RETIRED", timeout: "OUT OF TIME" },
  size: [W, H], pixelated: false,
  prep, draw, vitals,
  meter: (r, f) => ["km", `${(f.s / 1000).toFixed(2)} / ${(r.world.len / 1000).toFixed(2)}`],
};
})();
