"""Original soundtrack and sound design for the ActionMan research film, synthesised from scratch in numpy (no samples,
no scipy, nothing to license). Adapted from the Piccaso film's make_music.py.

120 bpm in A minor, bars every 2 s starting at 1 s, so the big cuts (title at 11 s, arena at 83 s, end card at 121 s) land on
a downbeat. Pixel-game colour: pulse-wave arps and lead over a warm pad. Sound effects are placed from cues.json, which the
page itself exports (typewriter tokens, the one-shot probability snap, slot pops, copies, every tool call landing in the
arena races, finishes, block wipes), so the audio follows the animation even after data.json or the replays change.

  node render.cjs --cues cues.json     # export the cue list from index.html first
  python make_music.py                 # -> music.wav (44.1 kHz stereo)
"""
import json
import sys
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# optional: python make_music.py <cues.json> <out.wav> (defaults: cues.json, music.wav)
CUES_PATH = HERE / (sys.argv[1] if len(sys.argv) > 1 else "cues.json")
OUT_PATH = HERE / (sys.argv[2] if len(sys.argv) > 2 else "music.wav")
CUES = json.loads(CUES_PATH.read_text()) if CUES_PATH.exists() else {}
SR = 44100
DUR = float(CUES.get("duration", 128))
N = int(SR * DUR)
BEAT, BAR0 = 0.5, 1.0
rng = np.random.default_rng(11)
SC = {k: v[0] for k, v in CUES.get("scenes", {}).items()}
S = lambda k, d: SC.get(k, d)  # noqa: E731  scene start time with a fallback
T_TITLE, T_MECH, T_JOURNEY, T_FAIL, T_GAP, T_BENCH, T_ARENA, T_LIMITS, T_END = (
    S("s2", 11), S("s3", 17), S("s5", 47), S("s6", 57), S("s7", 65), S("s8", 73), S("s9", 83), S("s13", 114), S("s14", 121))


def T(sec):
    return np.arange(int(sec * SR)) / SR


# ---------------------------------------------------------------- numpy-only DSP
def filt(x, kind, f, order=2):
    """Zero-phase Butterworth-magnitude filter applied in the frequency domain (works on 1-D or [2, n])."""
    n = x.shape[-1]
    m = 1 << int(np.ceil(np.log2(max(n, 2))))
    fr = np.fft.rfftfreq(m, 1 / SR)
    if kind == "lowpass":
        h = 1 / np.sqrt(1 + (fr / f) ** (2 * order))
    elif kind == "highpass":
        h = 1 / np.sqrt(1 + (f / np.maximum(fr, 1e-3)) ** (2 * order))
    else:
        lo, hi = f
        h = (1 / np.sqrt(1 + (lo / np.maximum(fr, 1e-3)) ** (2 * order))) * (1 / np.sqrt(1 + (fr / hi) ** (2 * order)))
    return np.fft.irfft(np.fft.rfft(x, m, axis=-1) * h, m, axis=-1)[..., :n]


def conv(a, b):
    n = a.shape[-1] + b.shape[-1] - 1
    m = 1 << int(np.ceil(np.log2(n)))
    return np.fft.irfft(np.fft.rfft(a, m) * np.fft.rfft(b, m), m)[:n]


def fade(x, a=0.003, r=0.02):
    n = x.shape[-1]
    e = np.ones(n)
    na, nr = min(n, int(a * SR)), min(n, int(r * SR))
    if na:
        e[:na] = np.linspace(0, 1, na)
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return x * e


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def noise(sec):
    return rng.standard_normal(int(sec * SR))


class Bus:
    def __init__(self):
        self.x = np.zeros((2, N))

    def add(self, sig, t, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if sig.ndim == 1:
            p = np.asarray(pan, dtype=float)
            sig = np.stack([sig * np.cos((p + 1) * np.pi / 4), sig * np.sin((p + 1) * np.pi / 4)]) * np.sqrt(2)
        if i < 0:
            sig, i = sig[:, -i:], 0
        n = min(sig.shape[1], N - i)
        if n > 0:
            self.x[:, i:i + n] += gain * sig[:, :n]


# ---------------------------------------------------------------- instruments
def kick():
    t = T(0.5)
    f = 46 + 110 * np.exp(-t / 0.032)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.24)
    click = filt(noise(0.5), "highpass", 2500) * np.exp(-t / 0.004) * 0.22
    return fade(np.tanh(1.6 * (body + click)))


def clap():
    t = T(0.45)
    env = sum(np.where(t >= o, np.exp(-(t - o) / d), 0) for o, d in ((0, 0.006), (0.011, 0.006), (0.022, 0.13)))
    return fade(filt(noise(0.45), "bandpass", [900, 3400]) * env)


def hat(d=0.04):
    t = T(d * 6)
    return fade(filt(noise(d * 6), "highpass", 7500, 4) * np.exp(-t / d))


@lru_cache(maxsize=None)
def pulse(m, dur, duty=0.25, bright=1.0, decay=None):
    """Band-limited pulse wave (the pixel-game voice), additive so it never aliases."""
    t = T(dur)
    f = hz(m)
    s = np.zeros_like(t)
    for h in range(1, 40):
        if f * h > min(SR / 2.4, 9000 * bright):
            break
        s += np.sin(np.pi * h * duty) / h * np.cos(2 * np.pi * f * h * t)
    env = np.exp(-t / decay) if decay else np.ones_like(t)
    return fade(s * env * 0.9, 0.002, 0.01)


@lru_cache(maxsize=None)
def lead(m, dur):
    t = T(dur + 0.12)
    vib = 1 + 0.005 * np.sin(2 * np.pi * 5.2 * t) * np.minimum(1, t / 0.3)
    ph = 2 * np.pi * np.cumsum(hz(m) * vib) / SR
    s = sum(np.sin(h * ph) * (np.sin(np.pi * h * 0.3) / h) for h in range(1, 14))
    env = np.minimum(1, t / 0.008) * (0.6 + 0.4 * np.exp(-t / 0.15)) * np.where(t < dur, 1, np.exp(-(t - dur) / 0.04))
    return fade(s * env)


@lru_cache(maxsize=None)
def saw_note(m, dur, voice=0):
    """Three detuned saws; `voice` picks an independent set of start phases (left / right channel)."""
    t = T(dur + 0.8)
    ph = np.random.default_rng(int(m) * 7 + voice).random(3)
    s = sum(2 * ((hz(m) * d * t + p) % 1) - 1 for d, p in zip((0.993, 1.0, 1.007), ph)) / 3
    env = np.minimum(1, t / 0.4) * np.where(t < dur, 1, np.exp(-(t - dur) / 0.25))
    return fade(s * env)


@lru_cache(maxsize=None)
def bass_note(m, dur):
    t = T(dur + 0.05)
    f = hz(m)
    tri = 2 / np.pi * np.arcsin(np.sin(2 * np.pi * f * t))
    s = 0.8 * tri + 0.5 * np.sin(np.pi * f * t)
    env = np.minimum(1, t / 0.006) * np.where(t < dur, np.exp(-t / 0.5), np.exp(-dur / 0.5) * np.exp(-(t - dur) / 0.03))
    return fade(np.tanh(1.3 * s * env))


# ---------------------------------------------------------------- sound effects
@lru_cache(maxsize=None)
def pop(m, dur=0.13):
    t = T(dur)
    f = hz(m) * (1 + 0.6 * np.exp(-t / 0.01))
    return fade(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.04))


def blip(m, dur=0.06):
    return pulse(m, dur, 0.5, 0.7, decay=0.025)


def key_click():
    t = T(0.035)
    return fade(filt(noise(0.035), "bandpass", [1800, 6000]) * np.exp(-t / 0.004) + 0.3 * np.sin(2 * np.pi * 1300 * t) * np.exp(-t / 0.006), 0.0005, 0.003)


def tick(f0=3000):
    t = T(0.02)
    return fade(filt(noise(0.02), "highpass", f0) * np.exp(-t / 0.003), 0.0005, 0.002)


def thud(f0=130, f1=45, d=0.3):
    t = T(d * 3)
    f = f1 + (f0 - f1) * np.exp(-t / 0.03)
    return fade(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / d))


def crash(d=1.4):
    t = T(d * 2.5)
    return fade(filt(noise(d * 2.5), "highpass", 4000) * np.exp(-t / d))


@lru_cache(maxsize=None)
def bell(m, d=1.6):
    t = T(d * 2)
    s = sum(a * np.sin(2 * np.pi * hz(m) * r * t) * np.exp(-t / (d / r ** 0.7)) for r, a in ((1, 1), (2.0, 0.45), (3.01, 0.25), (4.2, 0.15), (5.43, 0.08)))
    return fade(s * np.minimum(1, t / 0.002))


def whoosh(dur, f0, f1, f2, peak=0.5, width=0.45):
    """Band of noise whose centre glides f0 -> f1 (at `peak`) -> f2; a hand-rolled STFT overlap-add."""
    nper, hop = 1024, 256
    n = int(dur * SR)
    x = noise(dur + nper / SR)
    out = np.zeros(n + nper)
    win = np.hanning(nper)
    fr = np.fft.rfftfreq(nper, 1 / SR)
    for st in range(0, n, hop):
        u = st / n
        fc = f0 * (f1 / f0) ** (u / peak) if u < peak else f1 * (f2 / f1) ** ((u - peak) / (1 - peak))
        env = (u / peak) ** 2 if u < peak else ((1 - u) / (1 - peak)) ** 1.5
        mask = np.exp(-0.5 * (np.log(fr + 1) - np.log(fc)) ** 2 / width ** 2) * env
        out[st:st + nper] += np.fft.irfft(np.fft.rfft(x[st:st + nper] * win) * mask, nper) * win
    out = out[:n]
    return fade(out / (np.abs(out).max() + 1e-9))


def riser(dur):
    return crash(dur * 0.6)[: int(dur * SR)][::-1] * np.linspace(0.2, 1, int(dur * SR)) ** 2


# ---------------------------------------------------------------- harmony and form
CH = {"Am": [57, 60, 64, 67, 71], "F": [53, 57, 60, 64, 69], "C": [55, 60, 62, 64, 67], "Em": [52, 55, 59, 62, 67], "G": [55, 59, 62, 67, 69]}
ROOT = {"Am": 45, "F": 41, "C": 48, "Em": 40, "G": 43}
PROG = ["Am", "F", "C", "G"]
MEL = {"Am": [(0, .5, 76), (.5, .5, 79), (1, 1, 81), (2.5, .5, 79), (3, 1, 76)],
       "F": [(0, .5, 77), (.5, .5, 76), (1, 1.5, 72), (3, .5, 74), (3.5, .5, 76)],
       "C": [(0, .5, 79), (.5, .5, 76), (1, 1, 74), (2.5, .5, 72), (3, 1, 74)],
       "G": [(0, 1.5, 74), (1.5, .5, 71), (2, 1, 72), (3, 1, 74)]}


def chord(b):
    t = BAR0 + 2 * b
    if t >= T_END - 1e-6:
        return "Am"
    return PROG[b % 4]


def section(t):
    for end, name in ((T_TITLE, "intro"), (T_JOURNEY, "A"), (T_FAIL, "B"), (T_GAP, "dip"), (T_BENCH, "B"), (T_ARENA, "build"),
                      (T_LIMITS, "arena"), (T_END, "break"), (DUR + 1, "end")):
        if t < end - 1e-9:
            return name
    return "end"


GROOVE = {"A", "B", "arena"}
drums, bass, pad, arp, ld, sfx = Bus(), Bus(), Bus(), Bus(), Bus(), Bus()
duck = np.ones(N)
K, CL = kick(), clap()

for k in range(-2, int(DUR / BEAT) + 2):
    t = BAR0 + k * BEAT
    s, bb = section(t), k % 4
    if t < 0 or t >= DUR:
        continue
    if s in GROOVE or (s == "dip" and bb == 0):
        drums.add(K, t, 1.0 if s == "arena" else 0.8)
        i = int(t * SR)
        dk = 1 - 0.5 * np.exp(-T(0.4) / 0.11)
        duck[i:i + len(dk)] = np.minimum(duck[i:i + len(dk)], dk[: max(0, min(len(dk), N - i))])
    if s in ("B", "arena") and bb in (1, 3):
        drums.add(CL, t, 0.4, 0.05)
    if s in GROOVE | {"build"}:
        drums.add(hat(), t + 0.25, 0.2, 0.25)
    if s == "arena" or (s == "build" and t > T_ARENA - 4):
        for o in (0.125, 0.375):
            drums.add(hat(0.025), t + o, 0.07 * rng.uniform(0.7, 1.1), -0.3)
    if s in GROOVE:
        root = ROOT[chord(int(np.floor((t - BAR0) / 2)))]
        bass.add(bass_note(root + (12 if s == "arena" and bb == 3 else 0), 0.22), t + 0.25, 0.55)
    if s == "intro" and t >= 2.0:
        bass.add(bass_note(ROOT[chord(int(np.floor((t - BAR0) / 2)))], 0.35), t, 0.28)   # a slow pulse under the cold open

for t0 in (T_ARENA - 2.0,):                          # snare roll into the arena drop
    for j, t in enumerate(np.concatenate([np.arange(t0, t0 + 1, 0.25), np.arange(t0 + 1, t0 + 2, 0.125)])):
        drums.add(CL, t, 0.1 + 0.28 * j / 12, 0.05)

for b in range(-1, int((DUR - BAR0) / 2) + 1):
    t, c = BAR0 + 2 * b, chord(b)
    if t >= DUR:
        break
    last = t >= T_END - 1e-6
    s = section(t)
    dur = 6.5 if last else 2.0
    for m in CH[c]:
        pad.add(np.stack([saw_note(m, dur, 0), saw_note(m, dur, 1)]), t, 0.1)
    if last:
        bass.add(bass_note(ROOT[c] - 12, 4.0), t, 0.5)
        break
    if s in ("A", "B", "dip", "build", "arena", "break"):
        tones = [m + 12 for m in CH[c]]
        g = {"A": 0.12, "B": 0.13, "dip": 0.08, "build": 0.14, "arena": 0.15, "break": 0.09}[s]
        for j in range(16):
            ts = t + j * 0.125
            arp.add(pulse(tones[(0, 2, 4, 1, 3, 4, 2, 3)[j % 8]], 0.12, 0.25, 0.8, decay=0.06), ts, g * (1.0 if j % 4 == 0 else 0.7), 0.4 if j % 2 else -0.4)
    if s in ("B", "arena") and not (T_FAIL <= t < T_GAP):
        for o, d, m in MEL[c]:
            ld.add(lead(m, d * BEAT * 0.9), t + o * BEAT, 0.13 if s == "B" else 0.16)

# brightness automation: dark cold open, open groove, darker "what failed", bright arena, soft ending
w = np.interp(np.arange(N) / SR, [0, T_TITLE - 0.5, T_TITLE, T_FAIL, T_FAIL + 0.5, T_GAP, T_GAP + .5, T_BENCH, T_ARENA, T_LIMITS, T_END, DUR],
              [0.0, 0.1, 0.8, 0.8, 0.3, 0.3, 0.8, 0.5, 1.0, 1.0, 0.4, 0.6])
pad_mix = filt(pad.x, "lowpass", 450) * (1 - w) + filt(pad.x, "lowpass", 2200) * w
arp_mix = filt(arp.x, "lowpass", 1400) * (1 - w) + filt(arp.x, "lowpass", 5000) * w
bass_mix = filt(bass.x, "lowpass", 800)

# ---------------------------------------------------------------- sound effects, from the page's own cue list
for b in CUES.get("wipes", []):                      # pixel-block wipes: blips fill the screen, then clear
    for j in range(16):
        sfx.add(blip(int(rng.choice([72, 76, 79, 84, 88])), 0.05), b - 0.55 + j * 0.028, 0.05, -0.8 + j * 0.1)
    sfx.add(whoosh(1.0, 300, 2400, 500, peak=0.5), b - 0.55, 0.3, np.linspace(-0.7, 0.7, int(1.0 * SR)))
for t in (T_TITLE, T_ARENA, T_END):                  # drops
    drums.add(riser(2.0), t - 2.0, 0.3)
    drums.add(crash(1.6), t, 0.28)
    drums.add(thud(110, 38, 0.6), t, 0.55)
for t in CUES.get("type", []):                       # the LLM typing its call, one token per step
    sfx.add(key_click(), t + rng.uniform(-0.006, 0.006), 0.22 * rng.uniform(0.7, 1), rng.uniform(0.2, 0.6))
for t in CUES.get("snap", []):                       # one-shot probabilities: everything lands at once
    for m in (69, 72, 76, 81):
        sfx.add(pop(m + 12, 0.25), t, 0.09, -0.5)
    sfx.add(bell(88, 1.0), t, 0.06, -0.5)
for t, i in CUES.get("pops", []):
    sfx.add(pop([72, 76, 79, 81, 84, 88][int(i) % 6]), t, 0.17, 0.2)
for t in CUES.get("copy", []):                       # a span copied out of the request
    sfx.add(whoosh(0.4, 900, 3200, 1800, peak=0.7), t - 0.2, 0.14, 0.3)
    sfx.add(bell(91, 0.6), t + 0.05, 0.05, 0.3)
for t in CUES.get("bars", []):
    sfx.add(whoosh(0.9, 200, 1600, 600, peak=0.6), t - 0.1, 0.22, -0.2)
for t in CUES.get("thud", []):
    sfx.add(thud(150, 52, 0.25), t, 0.3)
last = {}
for t, lane, ok in sorted(CUES.get("race", [])):     # every tool call landing in the arena, thinned to <= 25 per second per lane
    if t - last.get(lane, -9) < 0.04:
        continue
    last[lane] = t
    if ok:
        sfx.add(blip([96, 88, 81][int(lane) % 3], 0.04), t, 0.05, [-0.5, 0.0, 0.5][int(lane) % 3])
    else:
        sfx.add(pulse(45, 0.08, 0.5, 0.4, decay=0.03), t, 0.12, [-0.5, 0.0, 0.5][int(lane) % 3])
for t, outcome in CUES.get("finish", []):
    if outcome == "win":
        for i, m in enumerate((81, 84, 88)):
            sfx.add(bell(m, 1.2), t + i * 0.06, 0.07, 0.2)
    elif outcome == "dead":
        sfx.add(thud(120, 40, 0.5), t, 0.4)
        sfx.add(filt(noise(0.4), "lowpass", 900) * np.exp(-T(0.4) / 0.1), t, 0.15)
for i, m in enumerate((69, 72, 76, 81)):             # end card
    sfx.add(bell(m, 2.4), T_END + 0.3 + i * 0.12, 0.08, -0.3 + i * 0.2)


# ---------------------------------------------------------------- mix
def pingpong(x, d=0.375, fb=0.35, taps=6):
    y, m = np.zeros_like(x), x.mean(axis=0)
    for k in range(1, taps + 1):
        s = int(d * k * SR)
        y[k % 2, s:] += fb ** k * m[: N - s]
    return filt(y, "lowpass", 3500)


def reverb(x, decay=0.5, length=2.4):
    t = T(length)
    ir = np.stack([filt(noise(length), "lowpass", 6000) * np.exp(-t / decay) for _ in range(2)])
    ir[:, : int(0.015 * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    return np.stack([conv(x[c], ir[c])[:N] for c in range(2)])


pad_mix, arp_mix, ld_x, drums_x = 2.0 * pad_mix, 1.5 * arp_mix, 2.0 * filt(ld.x, "lowpass", 4500), 0.45 * drums.x
music = drums_x + duck * (pad_mix + 0.8 * bass_mix + arp_mix) + ld_x
music += 0.3 * pingpong(arp_mix + 1.4 * ld_x) + 0.18 * reverb(pad_mix + arp_mix + ld_x + 0.5 * drums_x)
fx = 1.4 * (sfx.x + 0.22 * reverb(sfx.x, 0.35, 1.5))
mix = music + fx
env = np.interp(np.arange(N) / SR, [0, 0.25, DUR - 2.0, DUR], [0, 1, 1, 0])
mix = mix * env / np.quantile(np.abs(mix), 0.9995) * 0.8
mix = np.where(np.abs(mix) < 0.8, mix, np.sign(mix) * (0.8 + 0.18 * np.tanh((np.abs(mix) - 0.8) / 0.18)))

for name, x in (("drums", drums_x), ("bass", bass_mix), ("pad", pad_mix), ("arp", arp_mix), ("lead", ld_x), ("sfx", sfx.x), ("mix", mix)):
    print(f"{name:6s} rms {20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12):6.1f} dB  peak {np.abs(x).max():.2f}")
with wave.open(str(OUT_PATH), "wb") as f:
    f.setnchannels(2)
    f.setsampwidth(2)
    f.setframerate(SR)
    f.writeframes((np.clip(mix.T, -1, 1) * 32767).astype("<i2").tobytes())
print("wrote", OUT_PATH.name, "(cues found)" if CUES else "(no cues.json: music only, run node render.cjs --cues cues.json)")
