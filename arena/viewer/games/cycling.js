/* ActionMan Arena, cycling: flat-vector side view at golden hour. Parallax hills, a road that follows the course
   profile, and a rider whose cranks turn half a revolution per pedal() call. Registers ARENA.games.cycling. */
(() => {
"use strict";
const A = window.ARENA, RM = A.RM, rng = A.rng;
const W = 480, H = 300, S = 2;                 // logical canvas, drawn at 2x for crisp vectors
const PPM = 9, EX = 4, RX = 150, RY = 214, RS = 1.3;   // ... and rider scale     // px per metre, elevation exaggeration, rider anchor on screen
const TAU = Math.PI * 2;
const C = {
  sky: [[0, "#211a3a"], [.34, "#4d3360"], [.58, "#a1526a"], [.8, "#e88a5c"], [1, "#f8c77a"]],
  far: ["#b0687c", "#86557a"], mid: ["#6d4466", "#523657"], near: ["#3b2a48", "#2d223b"],
  ground: ["#251c33", "#120e1a"], rim: "#f5b46f", asphalt: "#3b3348", kerb: "#161220",
  ink: "#17121f", jersey: "#f2b35b", jersey2: "#c9803a", helmet: "#62e3ea", frame: "#ece6da",
  cream: "#f4ead8", ember: "#e8634f", diamond: "#62e3ea", lantern: "#f2b35b", dust: "#c9a27a",
};

/* ---------- course geometry ---------- */
function prep(r) {
  const w = r.world, n = Math.ceil(w.len) + 200, el = new Float32Array(n + 1);
  for (let i = 1; i <= n; i++) {                   // elevation per metre, index i = course metre i - 60
    const x = i - 60, s = w.segs.find(s => s[0] <= x - .5 && x - .5 < s[1]);
    el[i] = el[i - 1] + (s ? s[2] / 100 : 0);
  }
  w.el = el; w.lo = Math.min(...el); w.hi = Math.max(...el);
  const F = r.frames;
  r.st = []; r.cad = [];                          // tick each stroke count was reached; strokes over the last second
  F.forEach((f, i) => {
    r.st.push(i && f.n === F[i - 1].n ? r.st[i - 1] : f.t);
    const j = A.frameIdx(r, f.t - 100);
    r.cad.push(f.t >= 100 ? f.n - F[j].n : f.n);
  });
}
function elev(w, x) {
  const p = Math.min(Math.max(x + 60, 0), w.el.length - 2), i = Math.floor(p);
  return w.el[i] + (w.el[i + 1] - w.el[i]) * (p - i);
}
function terrainAt(w, x) { const s = w.segs.find(s => s[0] <= x && x < s[1]); return s ? s[2] : 0; }

/* ---------- interpolated state at tick t ---------- */
function state(r, t) {
  const F = r.frames, i = A.frameIdx(r, t), f0 = F[i], f1 = F[i + 1];
  const k = f1 && f1.t > f0.t ? Math.min(1, (t - f0.t) / (f1.t - f0.t)) : 0;
  const lerp = key => f0[key] + (f1 ? (f1[key] - f0[key]) * k : 0);
  const det = f0.d > 0 || (f1 && f1.d > 0 && k > 0);
  return { i, f: f0, x: det ? f0.x : lerp("x"), o: lerp("o"), v: lerp("v"), det, d: lerp("d") };
}
function cadence(r, i) { return r.cad[i] || 0; }

/* ---------- scenery ---------- */
function ridge(seed, u, amp) {                     // smooth hill line from three seeded sines
  const p = rng(seed), a = [p() * TAU, p() * TAU, p() * TAU];
  return amp * (.55 * Math.sin(u * .011 + a[0]) + .3 * Math.sin(u * .027 + a[1]) + .15 * Math.sin(u * .061 + a[2]));
}
function layer(x, seed, off, base, amp, cols, trees) {
  const g = x.createLinearGradient(0, base - amp, 0, H);
  g.addColorStop(0, cols[0]); g.addColorStop(1, cols[1]);
  x.fillStyle = g; x.beginPath(); x.moveTo(-4, H);
  const pts = [];
  for (let sx = -8; sx <= W + 8; sx += 6) { const y = base + ridge(seed, sx + off, amp); pts.push([sx, y]); x.lineTo(sx, y); }
  x.lineTo(W + 8, H); x.closePath(); x.fill();
  if (!trees) return;
  x.fillStyle = cols[1];
  const gap = 37, first = Math.floor(off / gap);
  for (let k = first; k < first + W / gap + 2; k++) {
    const rr = rng(seed * 31 + k * 7 + 1000);
    if (rr() < .45) continue;
    const sx = k * gap - off + rr() * 20, y = base + ridge(seed, sx + off, amp) + 2, h = 9 + rr() * 11;
    if (rr() < .6) { x.beginPath(); x.ellipse(sx, y - h / 2, 2.6, h / 2 + 1, 0, 0, TAU); x.fill(); }   // cypress
    else { x.beginPath(); x.arc(sx, y - h * .55, h * .42, 0, TAU); x.fill(); x.fillRect(sx - .8, y - h * .3, 1.6, h * .3 + 2); }
  }
}
function sky(x, L, amb, o) {
  const g = x.createLinearGradient(0, 0, 0, 178);
  for (const [s, c] of C.sky) g.addColorStop(s, c);
  x.fillStyle = g; x.fillRect(0, 0, W, H);
  const sx = 352, sy = 136;
  const glow = x.createRadialGradient(sx, sy, 8, sx, sy, 190);
  glow.addColorStop(0, "rgba(255,214,150,.75)"); glow.addColorStop(.25, "rgba(250,170,110,.28)"); glow.addColorStop(1, "rgba(250,150,100,0)");
  x.fillStyle = glow; x.fillRect(0, 0, W, H);
  x.fillStyle = "#ffe6b0"; x.beginPath(); x.arc(sx, sy, 27, 0, TAU); x.fill();
  x.fillStyle = "rgba(255,246,220,.9)"; x.beginPath(); x.arc(sx, sy, 22, 0, TAU); x.fill();
  for (const [cy, w0, sp, al, cw] of L.clouds) {    // soft lenticular wisps, lit from below by the low sun
    const cx = ((w0 - o * sp) % (W + 300) + W + 300) % (W + 300) - 150;
    const g = x.createLinearGradient(0, cy - 6, 0, cy + 4);
    g.addColorStop(0, `rgba(170,110,150,${al * .5})`); g.addColorStop(1, `rgba(255,190,150,${al})`);
    x.fillStyle = g;
    x.beginPath(); x.ellipse(cx, cy, cw, 3.2, 0, 0, TAU); x.fill();
    x.beginPath(); x.ellipse(cx + cw * .25, cy - 3, cw * .55, 3, 0, 0, TAU); x.fill();
    x.fillStyle = `rgba(255,214,170,${al})`; x.beginPath(); x.ellipse(cx + cw * .1, cy + 2.2, cw * .8, .9, 0, 0, TAU); x.fill();
  }
  if (!RM) {                                        // a few birds drifting home
    x.strokeStyle = "rgba(40,24,48,.6)"; x.lineWidth = 1;
    for (let b = 0; b < 3; b++) {
      const bx = (amb * 9 + b * 23 + L.seed * 50) % (W + 60) - 30, by = 70 + b * 7 + Math.sin(amb * 1.3 + b) * 3, fl = Math.sin(amb * 8 + b * 2) * 1.6;
      x.beginPath(); x.moveTo(bx - 4, by - fl); x.lineTo(bx, by); x.lineTo(bx + 4, by - fl); x.stroke();
    }
  }
}

/* ---------- road ---------- */
function roadY(w, cx, ce, wx, det) { return det ? RY : RY - (elev(w, wx) - ce) * PPM * EX; }
const SX = (cx, wx) => RX + (wx - cx) * PPM;

function road(x, w, cx, ce, o, det) {
  const pts = [];
  for (let sx = -10; sx <= W + 10; sx += 4) { const wx = cx + (sx - RX) / PPM; pts.push([sx, roadY(w, cx, ce, wx, det)]); }
  x.strokeStyle = "#2a2034"; x.lineWidth = .6;       // fence on the far verge
  const fy = [];
  for (let k = Math.floor((o - 20) / 4); k < (o + 40) / 4; k++) {
    const sx = RX + (k * 4 - o) * PPM, wx = cx + (sx - RX) / PPM, y = roadY(w, cx, ce, wx, det);
    x.fillStyle = "#2a2034"; x.fillRect(sx - .6, y - 9, 1.2, 9); fy.push([sx, y]);
  }
  for (const dy of [7.5, 4]) { x.beginPath(); fy.forEach(([sx, y], k) => k ? x.lineTo(sx, y - dy) : x.moveTo(sx, y - dy)); x.stroke(); }
  const g = x.createLinearGradient(0, RY - 30, 0, H);
  g.addColorStop(0, C.ground[0]); g.addColorStop(1, C.ground[1]);
  x.fillStyle = g; x.beginPath(); x.moveTo(-10, H);
  for (const [sx, y] of pts) x.lineTo(sx, y);
  x.lineTo(W + 10, H); x.closePath(); x.fill();
  x.lineJoin = "round";
  const line = (dy, col, lw) => { x.strokeStyle = col; x.lineWidth = lw; x.beginPath(); pts.forEach(([sx, y], k) => k ? x.lineTo(sx, y + dy) : x.moveTo(sx, y + dy)); x.stroke(); };
  line(4, det ? "#4a3a3c" : C.asphalt, 7);           // asphalt band (a dusty track on a detour)
  line(.5, C.rim, 1.4);                             // sunlit edge
  line(8, C.kerb, 1.2);
  const tuft = rng(5);                              // grass tufts on the verge, scrolling with the road
  x.fillStyle = "#1a1424";
  for (let k = Math.floor((o - 30) / 2.2); k < (o + 40) / 2.2; k++) {
    const tr = rng(k * 7919 + 3), sx = RX + (k * 2.2 + tr() * 1.5 - o) * PPM, wx = cx + (sx - RX) / PPM, y = roadY(w, cx, ce, wx, det) + 9;
    if (tr() < .5) continue;
    const h = 3 + tr() * 6;
    x.beginPath(); x.moveTo(sx - 2.5, y); x.lineTo(sx - 1.5, y + .5 - h * .6); x.lineTo(sx - .6, y); x.lineTo(sx + .2, y - h); x.lineTo(sx + 1, y); x.lineTo(sx + 2.4, y - h * .7); x.lineTo(sx + 3, y); x.fill();
  }
  x.fillStyle = "rgba(245,180,111,.45)";            // edge marks scroll by distance actually ridden
  const step = 3, first = Math.floor((o - RX / PPM) / step);
  for (let k = first; k < first + W / PPM / step + 2; k++) {
    const sx = RX + (k * step - o) * PPM, wx = cx + (sx - RX) / PPM;
    x.fillRect(sx, roadY(w, cx, ce, wx, det) + 4, 6, 1);
  }
}

function gravel(x, w, cx, ce, h, seed) {
  const r = rng(seed * 97 + Math.round(h.x));
  for (let i = 0; i < h.l * 9; i++) {
    const wx = h.x + r() * h.l, sx = SX(cx, wx);
    if (sx < -4 || sx > W + 4) { r(); r(); continue; }
    const y = roadY(w, cx, ce, wx) - .5 + r() * 6;
    x.fillStyle = r() < .5 ? C.dust : "#8d8094";
    x.fillRect(sx, y, 1.6, 1.2);
  }
}

function warning(x, sx, y, kind) {                  // triangle sign on a post
  x.fillStyle = "#2a2232"; x.fillRect(sx - .6, y - 22, 1.2, 22);
  x.fillStyle = C.ember; x.beginPath(); x.moveTo(sx, y - 34); x.lineTo(sx + 8, y - 21); x.lineTo(sx - 8, y - 21); x.closePath(); x.fill();
  x.fillStyle = C.cream; x.beginPath(); x.moveTo(sx, y - 31); x.lineTo(sx + 5.4, y - 22.6); x.lineTo(sx - 5.4, y - 22.6); x.closePath(); x.fill();
  x.fillStyle = x.strokeStyle = C.ink;
  if (kind === "gravel") { for (const [dx, dy] of [[-2, -24.5], [0, -26.5], [2, -24.5], [-.6, -24.4], [1, -27.6]]) x.fillRect(sx + dx, y + dy, 1.2, 1.2); }
  else { x.lineWidth = 1.1; x.beginPath(); x.moveTo(sx - 1.5, y - 23.4); x.lineTo(sx - 1.5, y - 26); x.quadraticCurveTo(sx - 1.5, y - 28.6, sx + 1.6, y - 28.6); x.stroke();
         x.beginPath(); x.moveTo(sx + 1.4, y - 30); x.lineTo(sx + 3, y - 28.6); x.lineTo(sx + 1.4, y - 27.2); x.fill(); }
}

function chevrons(x, sx, y) {
  x.fillStyle = "#2a2232"; x.fillRect(sx - 9, y - 11, 1.2, 11); x.fillRect(sx + 8, y - 11, 1.2, 11);
  x.fillStyle = C.ink; x.fillRect(sx - 11, y - 18, 22, 8);
  x.fillStyle = C.lantern;
  for (let k = 0; k < 3; k++) { const bx = sx - 8 + k * 6; x.beginPath(); x.moveTo(bx, y - 17); x.lineTo(bx + 3, y - 14); x.lineTo(bx, y - 11); x.lineTo(bx + 2, y - 11); x.lineTo(bx + 5, y - 14); x.lineTo(bx + 2, y - 17); x.fill(); }
}

function branch(x, sx, y, side) {                  // the other branch: into the scene (left) or towards us (right)
  if (side === "left") {
    const len = 150, g = x.createLinearGradient(sx, y, sx + len, y - 26);
    g.addColorStop(0, C.asphalt); g.addColorStop(1, "rgba(59,51,72,0)");
    x.fillStyle = g; x.beginPath(); x.moveTo(sx - 8, y + 1); x.lineTo(sx + len, y - 26); x.lineTo(sx + len, y - 23); x.lineTo(sx + 14, y + 1); x.closePath(); x.fill();
    x.strokeStyle = "rgba(245,180,111,.35)"; x.lineWidth = 1; x.beginPath(); x.moveTo(sx - 8, y + .6); x.lineTo(sx + len, y - 26.4); x.stroke();
    return;
  }
  const g = x.createLinearGradient(0, y + 7, 0, y + 46);
  g.addColorStop(0, C.asphalt); g.addColorStop(1, "rgba(59,51,72,0)");
  x.fillStyle = g; x.beginPath(); x.moveTo(sx - 4, y + 8); x.lineTo(sx + 22, y + 8); x.lineTo(sx + 84, y + 46); x.lineTo(sx + 34, y + 46); x.closePath(); x.fill();
  const e = x.createLinearGradient(0, y + 8, 0, y + 46);
  e.addColorStop(0, "rgba(245,180,111,.4)"); e.addColorStop(1, "rgba(245,180,111,0)");
  x.strokeStyle = e; x.lineWidth = 1; x.beginPath(); x.moveTo(sx + 22, y + 8); x.lineTo(sx + 84, y + 46); x.stroke();
}

function signpost(x, sx, y, f) {
  const other = f.c === "left" ? "right" : "left";
  x.fillStyle = "#2a2232"; x.fillRect(sx - .7, y - 40, 1.4, 40);
  board(x, sx, y - 40, `FINISH ${f.c.toUpperCase()}`, C.cream, C.ink, f.c === "left" ? -1 : 1);
  board(x, sx, y - 28, `+${f.d} m ${other.toUpperCase()}`, "#4a3a52", "#d8cbe0", other === "left" ? -1 : 1);
}
function board(x, sx, y, text, bg, fg, dir) {
  x.font = "600 7.5px 'JetBrains Mono', monospace";
  const tw = x.measureText(text).width + 10, bx = sx - tw / 2;
  x.fillStyle = bg; x.beginPath();
  if (dir > 0) { x.moveTo(bx, y); x.lineTo(bx + tw, y); x.lineTo(bx + tw + 5, y + 5); x.lineTo(bx + tw, y + 10); x.lineTo(bx, y + 10); }
  else { x.moveTo(bx + tw, y); x.lineTo(bx, y); x.lineTo(bx - 5, y + 5); x.lineTo(bx, y + 10); x.lineTo(bx + tw, y + 10); }
  x.closePath(); x.fill();
  x.fillStyle = fg; x.textAlign = "center"; x.fillText(text, sx, y + 7.6); x.textAlign = "left";
}

function marker(x, sx, y, m) {
  x.fillStyle = "#2a2232"; x.fillRect(sx - .5, y - 13, 1, 13);
  x.fillStyle = C.cream; x.beginPath(); x.roundRect(sx - 8, y - 20, 16, 8, 1.5); x.fill();
  x.fillStyle = C.ink; x.font = "600 6.5px 'JetBrains Mono', monospace"; x.textAlign = "center"; x.fillText(String(m), sx, y - 13.8); x.textAlign = "left";
}

function finishArch(x, sx, y) {
  x.fillStyle = "rgba(42,34,50,.8)"; x.fillRect(sx + 7, y - 80, 2, 78);       // far post, a step behind the road
  x.fillStyle = "#2a2232"; x.fillRect(sx - 1.2, y - 80, 2.4, 80);
  const bx = sx - 18, by = y - 92, bw = 44;
  x.fillStyle = C.ink; x.beginPath(); x.roundRect(bx - 2, by - 2, bw + 4, 18, 2); x.fill();
  for (let k = 0; k < 11; k++) for (let j = 0; j < 2; j++) { x.fillStyle = (k + j) % 2 ? C.ink : C.cream; x.fillRect(bx + k * 4, by + 8 + j * 3.5, 4, 3.5); }
  x.fillStyle = C.lantern; x.font = "700 8px 'Pixelify Sans', monospace"; x.textAlign = "center";
  x.fillText("FINISH", bx + bw / 2, by + 6.5); x.textAlign = "left";
  x.fillStyle = "rgba(255,255,255,.85)"; x.fillRect(sx, y + 1, 1.6, 7);     // line across the road
}

function features(x, w, cx, ce, seed, amb, back) {
  const vis = wx => SX(cx, wx) > -60 && SX(cx, wx) < W + 160;
  for (const f of w.forks) if (vis(f.x)) {
    const sx = SX(cx, f.x), y = roadY(w, cx, ce, f.x);
    const other = f.c === "left" ? "right" : "left";
    if (back === (other === "left")) branch(x, sx, y, other);
    if (back) signpost(x, SX(cx, f.x - 4), roadY(w, cx, ce, f.x - 4), f);
  }
  if (!back) return;
  for (const h of w.haz) {
    if (vis(h.x)) { if (h.k === "gravel") gravel(x, w, cx, ce, h, seed); else chevrons(x, SX(cx, h.x + h.l / 2), roadY(w, cx, ce, h.x + h.l / 2)); }
    if (vis(h.x - 14)) warning(x, SX(cx, h.x - 14), roadY(w, cx, ce, h.x - 14), h.k);
  }
  for (let m = 25; m < w.len; m += 25) {
    const wx = w.len - m; if (vis(wx) && !w.forks.some(f => Math.abs(f.x - wx) < 6) && !w.haz.some(h => Math.abs(h.x - 14 - wx) < 5)) marker(x, SX(cx, wx), roadY(w, cx, ce, wx), m);
  }
  if (vis(w.len)) finishArch(x, SX(cx, w.len), roadY(w, cx, ce, w.len));
}

/* ---------- rider ---------- */
function ik(hx, hy, fx, fy, l1, l2) {               // knee, bending forward
  let dx = fx - hx, dy = fy - hy, d = Math.hypot(dx, dy);
  d = Math.min(d, l1 + l2 - .01);
  const a = Math.atan2(dy, dx), b = Math.acos((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d));
  return [hx + Math.cos(a - b) * l1, hy + Math.sin(a - b) * l1];
}
function wheel(x, cx, cy, ang, blur) {
  x.strokeStyle = C.ink; x.lineWidth = 2.6; x.beginPath(); x.arc(cx, cy, 10, 0, TAU); x.stroke();
  x.strokeStyle = "rgba(236,230,218,.55)"; x.lineWidth = .7; x.beginPath(); x.arc(cx, cy, 8.4, 0, TAU); x.stroke();
  if (blur) { x.fillStyle = "rgba(236,230,218,.07)"; x.beginPath(); x.arc(cx, cy, 8.2, 0, TAU); x.fill(); }
  else {
    x.strokeStyle = "rgba(236,230,218,.4)"; x.lineWidth = .5; x.beginPath();
    for (let k = 0; k < 4; k++) { const a = ang + k * Math.PI / 4; x.moveTo(cx - Math.cos(a) * 8.4, cy - Math.sin(a) * 8.4); x.lineTo(cx + Math.cos(a) * 8.4, cy + Math.sin(a) * 8.4); }
    x.stroke();
  }
  x.fillStyle = C.frame; x.beginPath(); x.arc(cx, cy, 1.1, 0, TAU); x.fill();
}
function rider(x, P) {                              // origin: ground under the bottom bracket; facing right
  const { crank, wang, lean, blur, stand } = P;
  const RW = [-15, -10], FW = [15, -10], BB = [-2, -9], SEAT = [-6.5, -26], HEAD = [11, -25], BAR = [13.5, -28];
  const CR = 4.6, ped = s => [BB[0] + Math.cos(crank + s) * CR, BB[1] + Math.sin(crank + s) * CR];
  const hip = [SEAT[0] + 1 + stand * 3, SEAT[1] - 2.5 - stand * 4];
  const ta = -(.62 + lean) , TL = 15.5;              // torso angle from horizontal (negative = up)
  const sh = [hip[0] + Math.cos(ta) * TL, hip[1] + Math.sin(ta) * TL];
  const leg = (s, col, w) => {
    const p = ped(s), k = ik(hip[0], hip[1], p[0], p[1] - 1, 12.4, 12.6);
    x.strokeStyle = col; x.lineWidth = w; x.lineCap = "round";
    x.beginPath(); x.moveTo(hip[0], hip[1]); x.lineTo(k[0], k[1]); x.lineTo(p[0], p[1] - 1); x.stroke();
    x.fillStyle = C.ink; x.beginPath(); x.ellipse(p[0] + 1, p[1], 2.4, 1.2, 0, 0, TAU); x.fill();
  };
  leg(Math.PI, "#3a2c46", 3.6);                      // far leg
  wheel(x, RW[0], RW[1], wang, blur); wheel(x, FW[0], FW[1], wang, blur);
  x.strokeStyle = C.frame; x.lineWidth = 1.7; x.lineJoin = "round"; x.lineCap = "round";
  x.beginPath(); x.moveTo(RW[0], RW[1]); x.lineTo(BB[0], BB[1]); x.lineTo(SEAT[0], SEAT[1]); x.lineTo(RW[0], RW[1]);
  x.moveTo(SEAT[0], SEAT[1]); x.lineTo(HEAD[0], HEAD[1] + 1); x.lineTo(BB[0], BB[1]);
  x.moveTo(HEAD[0], HEAD[1]); x.lineTo(FW[0], FW[1]); x.moveTo(HEAD[0], HEAD[1]); x.lineTo(BAR[0], BAR[1]); x.stroke();
  x.strokeStyle = C.ink; x.lineWidth = 1.6;
  x.beginPath(); x.moveTo(BAR[0], BAR[1]); x.quadraticCurveTo(BAR[0] + 4, BAR[1] + .5, BAR[0] + 2.5, BAR[1] + 4); x.stroke();
  x.beginPath(); x.moveTo(SEAT[0] - 3, SEAT[1] - .5); x.lineTo(SEAT[0] + 3, SEAT[1] - .5); x.stroke();
  if (blur) { x.strokeStyle = "rgba(236,230,218,.35)"; x.lineWidth = 1; x.beginPath(); x.arc(BB[0], BB[1], CR, 0, TAU); x.stroke(); }
  x.strokeStyle = "#2a2232"; x.lineWidth = 1.2; x.beginPath(); const p1 = ped(0), p2 = ped(Math.PI);
  x.moveTo(p1[0], p1[1]); x.lineTo(p2[0], p2[1]); x.stroke();
  x.fillStyle = C.ink; x.beginPath(); x.arc(BB[0], BB[1], 1.8, 0, TAU); x.fill();
  // torso: jersey with a darker side panel, then near leg over the frame
  x.strokeStyle = C.jersey; x.lineWidth = 6.2; x.lineCap = "round";
  x.beginPath(); x.moveTo(hip[0], hip[1]); x.lineTo(sh[0], sh[1]); x.stroke();
  x.strokeStyle = C.jersey2; x.lineWidth = 2; x.beginPath(); x.moveTo(hip[0] + .5, hip[1] + 1.6); x.lineTo(sh[0] + .8, sh[1] + 1.8); x.stroke();
  x.fillStyle = C.ink; x.beginPath(); x.arc(hip[0], hip[1] + .5, 3.4, 0, TAU); x.fill();     // shorts
  leg(0, C.ink, 3.8);
  const hand = [BAR[0] + 2.5, BAR[1] + 2.5], el = [(sh[0] + hand[0]) / 2 - .5, (sh[1] + hand[1]) / 2 + 2.6];
  x.strokeStyle = C.jersey; x.lineWidth = 2.6; x.beginPath(); x.moveTo(sh[0], sh[1]); x.lineTo(el[0], el[1]); x.stroke();
  x.strokeStyle = "#3a2c46"; x.lineWidth = 2.1; x.beginPath(); x.moveTo(el[0], el[1]); x.lineTo(hand[0], hand[1]); x.stroke();
  const hd = [sh[0] + Math.cos(ta + .55) * 6.2, sh[1] + Math.sin(ta + .55) * 6.2 - 1.2];
  x.fillStyle = "#2b2033"; x.beginPath(); x.arc(hd[0], hd[1], 3.6, 0, TAU); x.fill();
  x.fillStyle = C.helmet; x.beginPath(); x.moveTo(hd[0] - 5.5, hd[1] + .2); x.quadraticCurveTo(hd[0] - 2, hd[1] - 5.6, hd[0] + 3.6, hd[1] - 1.6);
  x.quadraticCurveTo(hd[0] + 4.2, hd[1] - .2, hd[0] + 3, hd[1] + .2); x.closePath(); x.fill();
  x.fillStyle = "rgba(255,255,255,.55)"; x.fillRect(hd[0] - 1.6, hd[1] - 3.4, 2.4, .8);
}

function fallen(x, ang) {                          // bike on its side, rider sitting up beside it
  x.save(); x.translate(8, 0); x.scale(1, .3);
  for (const cx of [-15, 15]) { x.strokeStyle = C.ink; x.lineWidth = 3; x.beginPath(); x.arc(cx, -10, 10, 0, TAU); x.stroke(); }
  x.strokeStyle = C.frame; x.lineWidth = 2.2; x.beginPath(); x.moveTo(-15, -10); x.lineTo(-2, -9); x.lineTo(-6.5, -26); x.lineTo(-15, -10);
  x.moveTo(-6.5, -26); x.lineTo(11, -24); x.lineTo(-2, -9); x.moveTo(11, -25); x.lineTo(15, -10); x.stroke();
  x.restore();
  const hip = [-14, -3.5], sh = [-17, -17], knee = [-6, -9], foot = [0, -1.5];
  x.lineCap = "round";
  x.strokeStyle = C.ink; x.lineWidth = 3.8; x.beginPath(); x.moveTo(hip[0], hip[1]); x.lineTo(knee[0], knee[1]); x.lineTo(foot[0], foot[1]); x.stroke();
  x.strokeStyle = C.jersey; x.lineWidth = 6.2; x.beginPath(); x.moveTo(hip[0], hip[1]); x.lineTo(sh[0], sh[1]); x.stroke();
  x.lineWidth = 2.6; x.beginPath(); x.moveTo(sh[0], sh[1] + 1); x.lineTo(-22, -6); x.stroke();
  x.fillStyle = "#2b2033"; x.beginPath(); x.arc(-16.5, -22.5, 3.6, 0, TAU); x.fill();
  x.fillStyle = C.helmet; x.beginPath(); x.arc(-16.5, -23.2, 3.9, Math.PI * 1.05, Math.PI * 1.95); x.closePath(); x.fill();
}

/* ---------- HUD ---------- */
function ring(x, cad, amb) {
  const cx = W - 34, cy = 36, R = 21, frac = Math.min(1, cad / 40);
  x.fillStyle = "rgba(20,15,30,.55)"; x.beginPath(); x.arc(cx, cy, R + 7, 0, TAU); x.fill();
  x.strokeStyle = "rgba(255,255,255,.12)"; x.lineWidth = 3.5; x.beginPath(); x.arc(cx, cy, R, 0, TAU); x.stroke();
  for (let k = 0; k <= 8; k++) { const a = -Math.PI / 2 + k / 8 * TAU * .999; x.fillStyle = "rgba(255,255,255,.25)"; x.fillRect(cx + Math.cos(a) * (R + 4.5) - .5, cy + Math.sin(a) * (R + 4.5) - .5, 1, 1); }
  if (frac > 0) {
    const g = x.createLinearGradient(cx - R, cy + R, cx + R, cy - R);
    g.addColorStop(0, "#e88a5c"); g.addColorStop(1, "#ffd28a");
    x.strokeStyle = g; x.lineWidth = 3.5; x.lineCap = "round";
    x.beginPath(); x.arc(cx, cy, R, -Math.PI / 2, -Math.PI / 2 + frac * TAU); x.stroke();
  }
  x.fillStyle = C.cream; x.textAlign = "center";
  x.font = "600 15px 'JetBrains Mono', monospace"; x.fillText(String(cad), cx, cy + 3);
  x.font = "600 5px 'Instrument Sans', sans-serif"; x.fillStyle = "rgba(236,230,218,.75)"; x.fillText("STROKES/S", cx, cy + 10.5);
  x.textAlign = "left";
}
function speedo(x, kmh, gear) {
  x.fillStyle = C.cream; x.font = "600 22px 'JetBrains Mono', monospace"; x.fillText(String(kmh), 14, 32);
  const w = x.measureText(String(kmh)).width;
  x.font = "500 9.5px 'Instrument Sans', sans-serif"; x.fillStyle = "rgba(236,230,218,.75)"; x.fillText("km/h", 17 + w, 32);
  x.font = "600 7.5px 'JetBrains Mono', monospace";
  const lab = gear ? "HIGH GEAR" : "LOW GEAR", gw = x.measureText(lab).width + 10;
  x.fillStyle = gear ? "rgba(98,227,234,.18)" : "rgba(242,179,91,.2)"; x.beginPath(); x.roundRect(14, 39, gw, 13, 6.5); x.fill();
  x.fillStyle = gear ? C.diamond : C.lantern; x.fillText(lab, 19, 48.2);
}
function strip(x, w, xr, det, amb) {
  const x0 = 14, x1 = W - 14, y0 = H - 30, y1 = H - 9, span = Math.max(.5, w.hi - w.lo);
  x.fillStyle = "rgba(12,9,20,.6)"; x.beginPath(); x.roundRect(x0 - 6, y0 - 7, x1 - x0 + 12, y1 - y0 + 12, 6); x.fill();
  const X = m => x0 + m / w.len * (x1 - x0), Y = m => y1 - 3 - (elev(w, m) - w.lo) / span * (y1 - y0 - 7);
  const path = upto => { x.beginPath(); x.moveTo(X(0), y1); for (let m = 0; m <= upto; m += 1) x.lineTo(X(m), Y(m)); x.lineTo(X(upto), Y(upto)); x.lineTo(X(upto), y1); x.closePath(); };
  x.fillStyle = "rgba(236,230,218,.12)"; path(w.len); x.fill();
  const g = x.createLinearGradient(0, y0, 0, y1); g.addColorStop(0, "rgba(242,179,91,.75)"); g.addColorStop(1, "rgba(242,179,91,.15)");
  x.fillStyle = g; path(Math.max(0, Math.min(w.len, xr))); x.fill();
  x.strokeStyle = "rgba(236,230,218,.35)"; x.lineWidth = .8; x.beginPath();
  for (let m = 0; m <= w.len; m++) m ? x.lineTo(X(m), Y(m)) : x.moveTo(X(m), Y(m));
  x.stroke();
  for (const h of w.haz) { x.fillStyle = C.ember; x.beginPath(); const hx = X(h.x), hy = Y(h.x) - 4; x.moveTo(hx, hy - 4); x.lineTo(hx + 3, hy); x.lineTo(hx - 3, hy); x.fill(); }
  for (const f of w.forks) { x.fillStyle = C.diamond; const fx = X(f.x), fy = Y(f.x) - 5; x.beginPath(); x.moveTo(fx, fy - 3); x.lineTo(fx + 3, fy); x.lineTo(fx, fy + 3); x.lineTo(fx - 3, fy); x.fill(); }
  const fx = X(w.len) - 1, fy = Y(w.len);
  x.fillStyle = C.cream; x.fillRect(fx, fy - 13, 1, 13); x.fillRect(fx - 9, fy - 13, 9, 6);
  for (let k = 0; k < 6; k++) { x.fillStyle = C.ink; x.fillRect(fx - 9 + (k % 3) * 3 + (Math.floor(k / 3) % 2 ? 1.5 : 0), fy - 13 + Math.floor(k / 3) * 3, 1.5, 3); }
  const px = X(Math.min(w.len, xr)), py = Y(Math.min(w.len, xr)), pulse = det && !RM ? 1 + .4 * Math.sin(amb * 8) : 1;
  x.fillStyle = det ? C.ember : C.cream; x.beginPath(); x.arc(px, py, 3.2 * pulse, 0, TAU); x.fill();
  x.strokeStyle = C.lantern; x.lineWidth = 1.2; x.beginPath(); x.arc(px, py, 4.6 * pulse, 0, TAU); x.stroke();
}
function banner(x, text, col, a = 1) {
  x.font = "600 9px 'JetBrains Mono', monospace";
  const tw = x.measureText(text).width + 18, bx = (W - tw) / 2;
  x.globalAlpha = a; x.fillStyle = "rgba(20,15,30,.78)"; x.beginPath(); x.roundRect(bx, 64, tw, 17, 8.5); x.fill();
  x.fillStyle = col; x.textAlign = "center"; x.fillText(text, W / 2, 75.5); x.textAlign = "left"; x.globalAlpha = 1;
}
function bubble(x, bx, by, prog, amb) {
  x.fillStyle = "rgba(20,15,30,.82)"; x.beginPath(); x.roundRect(bx - 14, by - 13, 28, 14, 7); x.fill();
  x.beginPath(); x.moveTo(bx - 3, by + 1); x.lineTo(bx + 1, by + 5); x.lineTo(bx + 3, by + 1); x.fill();
  for (let k = 0; k < 3; k++) { x.fillStyle = (RM || Math.floor(amb * 4 + k) % 3) ? C.lantern : "#5a4630"; x.beginPath(); x.arc(bx - 6 + k * 6, by - 7, 1.5, 0, TAU); x.fill(); }
  x.fillStyle = C.lantern; x.fillRect(bx - 10, by - 2.5, 20 * prog, 1);
}

/* ---------- frame ---------- */
function draw(L, t, amb, cur) {
  const r = L.r, w = r.world, x = L.ctx;
  if (!L.clouds) {
    const cr = rng(r.seed * 13 + r.level * 5 + 1);
    L.seed = r.seed; L.clouds = [0, 1, 2, 3].map(k => [56 + k * 15 + cr() * 6, cr() * 700, .04 + cr() * .05, .3 + cr() * .25, 40 + cr() * 50]);
    L.ang = 0; L.tgt = null; L.wang = 0; L.wo = null;
  }
  const s = state(r, t), f = s.f, cx = s.x, ce = elev(w, cx), det = s.det;
  x.setTransform(S, 0, 0, S, 0, 0);
  sky(x, L, amb, s.o);
  const vy = det ? 0 : ce * PPM * EX;                // hills drift with the rider's height, less the further away
  layer(x, 11 + r.seed, s.o * PPM * .05, 146 + vy * .06, 30, C.far, false);
  layer(x, 23 + r.seed, s.o * PPM * .15, 172 + vy * .15, 26, C.mid, true);
  layer(x, 37 + r.seed, s.o * PPM * .36, 196 + vy * .3, 18, C.near, true);
  const haze = x.createLinearGradient(0, 130, 0, 215); haze.addColorStop(0, "rgba(248,199,122,0)"); haze.addColorStop(1, "rgba(248,170,110,.14)");
  x.fillStyle = haze; x.fillRect(0, 130, W, 85);
  if (!det) features(x, w, cx, ce, r.seed, amb, true);
  road(x, w, cx, ce, s.o, det);
  if (!det) features(x, w, cx, ce, r.seed, amb, false);

  // cranks: half a turn per stroke, in sync; past one stroke per screen frame they blur instead of aliasing
  const st = r.st[s.i], n = f.n - 1 + Math.min(1, (t - st) / Math.max(1, Math.min(14, (r.frames[s.i + 1]?.t ?? t + 14) - st)));
  const tgt = Math.PI * Math.max(0, n);
  let blur = false;
  if (L.tgt === null || tgt < L.tgt || tgt - L.tgt > 40) L.ang = tgt;
  else { const d = tgt - L.tgt; if (d > Math.PI * .45) blur = true; L.ang += Math.min(d, Math.PI * .45); }
  L.tgt = tgt;
  if (L.wo === null || s.o < L.wo) L.wang = s.o / .34; else L.wang += Math.min((s.o - L.wo) / .34, .7);
  L.wo = s.o;
  const grade = det ? 0 : terrainAt(w, cx), slope = det ? 0 : -Math.atan((roadY(w, cx, ce, cx + 1) - roadY(w, cx, ce, cx - 1)) / (2 * PPM));
  const kmh = Math.round(s.v * 3.6), crashed = f.c === 1;
  if (!RM && s.v > 8 && !crashed) {                  // speed lines
    const a = Math.min(.5, (s.v - 8) / 14), sr = rng(7);
    x.strokeStyle = `rgba(255,230,190,${a.toFixed(2)})`; x.lineWidth = .8;
    for (let k = 0; k < 7; k++) { const yy = RY - 46 + sr() * 40, len = 14 + sr() * 26, ph = ((amb * (90 + s.v * 9) + sr() * 400) % 220);
      x.beginPath(); x.moveTo(RX - 26 - ph, yy); x.lineTo(RX - 26 - ph - len, yy); x.stroke(); }
  }
  x.save(); x.translate(RX, RY); x.scale(RS, RS);
  x.rotate(-slope);
  if (crashed) fallen(x, L.ang);
  else { x.rotate(slope * .55); rider(x, { crank: L.ang, wang: L.wang, lean: grade > 0 ? -.02 : grade < 0 ? .34 : .14, blur, stand: grade > 0 ? 1 : 0 }); }
  x.restore();
  const evs = w.ev;                                  // crash dust, brake glow, fork calls
  for (const e of evs) {
    const age = t - e.t; if (age < 0) break; if (age > 110) continue;
    if (e.k === "crash" && !RM) {
      const er = rng(e.t);
      for (let k = 0; k < 26; k++) {
        const vx = (er() - .3) * 1.4, vy = -er() * 1.1, a2 = age / 10;
        x.fillStyle = `rgba(201,162,122,${Math.max(0, .75 - age / 110).toFixed(2)})`;
        x.beginPath(); x.arc(RX + vx * a2 * 6, RY - 2 + vy * a2 * 6 + a2 * a2 * .35, 1 + er() * 2.2 + a2 * .25, 0, TAU); x.fill();
      }
    }
    if (age < 60) {
      const lab = { slow: ["BRAKED", C.cream], fork: ["ON COURSE", C.diamond], wrong: ["WRONG BRANCH", C.ember] }[e.k];
      if (lab) { x.font = "700 8px 'JetBrains Mono', monospace"; x.fillStyle = lab[1]; x.globalAlpha = 1 - age / 60; x.textAlign = "center";
        x.fillText(lab[0], RX, RY - 68 - age * .15); x.globalAlpha = 1; x.textAlign = "left"; }
    }
  }
  const vg = x.createRadialGradient(W / 2, H * .45, H * .35, W / 2, H * .45, W * .62);   // poster vignette
  vg.addColorStop(0, "rgba(10,6,16,0)"); vg.addColorStop(1, "rgba(10,6,16,.38)");
  x.fillStyle = vg; x.fillRect(0, 0, W, H);
  if (cur && t < cur.t_land && cur.t_land - cur.t_obs >= 10) bubble(x, RX + 26, RY - 76, (t - cur.t_obs) / (cur.t_land - cur.t_obs), amb);
  speedo(x, kmh, f.g);
  ring(x, cadence(r, s.i), amb);
  if (det) banner(x, `DETOUR  ${Math.ceil(s.d)} m TO REJOIN`, C.ember);
  else if (crashed) banner(x, "CRASHED  BACK ON THE BIKE", C.ember);
  strip(x, w, cx, det, amb);
  x.setTransform(1, 0, 0, 1, 0, 0);
  return f;
}

/* ---------- panel bits ---------- */
function vitals(r, f, t) {
  const i = A.frameIdx(r, t);
  return `<span>speed <b>${Math.round(f.v * 3.6)}</b> km/h</span><span>cadence <b>${cadence(r, i)}</b>/s</span>` +
    `<span>gear <b>${f.g ? "high" : "low"}</b></span>` + (f.c ? `<span style="color:var(--ember)">crashed</span>` : "");
}

A.games.cycling = {
  title: "Cycling", tagline: "Every call is a pedal stroke. Faster thinking, faster bike.",
  levels: [
    { name: "Warm-up", desc: "A flat road with one fork. Pedal, and pick the branch the finish is on." },
    { name: "Full course", desc: "Climbs want low gear, descents high gear. Brake before gravel and bends, choose the right branch." },
    { name: "Renamed tools", desc: "Same course, new controls: the tools and their parameters have different names." },
    { name: "Renamed values", desc: "Gears and sides are renamed too. The road still says left, right, up and down." },
  ],
  words: { win: "FINISH", dead: "CRASHED OUT", timeout: "OUT OF TIME" },
  size: [W * S, H * S], pixelated: false,
  prep, draw, vitals,
  meter: (r, f) => ["distance", `${Math.round(f.x)} / ${r.world.len} m`],
};
})();
