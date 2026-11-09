"""Diagnostic probe for the rekordbox grid writer (NOT the generator).

This implements the parts of `BeatAnalyzer_2_0::UnitBeatAdjust` that have been
recovered from the binary, so the model can be checked against real audio:

  * 4:1 decimation of the stereo stream, mono = (L+R)/2 of every 4th frame
  * the five biquads (HighPathFilter2D / LowPathFilter2D, Q = 1/sqrt(2))
  * the three int16 envelopes the analyser keeps:
        A = LP1000(|HP2000(x)|)     ("high", this+0x70, used for scoring)
        B = LP25  (|LP100 (x)|)     ("low",  this+0x78, used by calcLowInteg)
        C = LP1000(|x|)             ("all",  this+0x80, used by calcAllInteg)
  * createBeatData(bpm, startPos): the nominal grid in 11025 units
  * manageWave(): beat-aligned ±50 ms profile of envelope A, score = max
  * the per-beat-in-bar profiles used to find the downbeat
  * calcLowInteg()/calcAllInteg(): window integrals over B / C

Nothing here writes a grid; `detect.py` is still the only generator and it is
still not exact. See REVERSE_ENGINEERING.md for the disassembly notes.

Usage:
  python writer_probe.py track.m4a
  python writer_probe.py track.m4a --bpm 122 --start 0
  python writer_probe.py track.m4a --scan 121.9 122.1
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

SR = 44100
SRD = SR // 4            # 11025
HALF = int(SRD * 0.05)   # 551, from (int)(state[0xc] * 0.05)
Q = 0.7071067811865476


# ---------------------------------------------------------------- signal chain

def load_i16(path: str | Path) -> np.ndarray:
    """Decode to interleaved int16 stereo at 44.1 kHz, like BA_SetSample sees it."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le",
           "-ac", "2", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="<i2").reshape(-1, 2)


def _coeffs(freq: float, kind: str) -> tuple[float, ...]:
    """RBJ biquad, as in HighPathFilter2D::create / LowPathFilter2D::create."""
    w = 2.0 * np.pi * freq / SRD
    s, c = np.sin(w), np.cos(w)
    alpha = s * Q
    n = 1.0 / (1.0 + alpha)
    if kind == "hp":
        b = (1.0 + c) * 0.5 * n
        return (b, -(1.0 + c) * n, b, -2.0 * c * n, (1.0 - alpha) * n)
    b = (1.0 - c) * 0.5 * n
    return (b, (1.0 - c) * n, b, -2.0 * c * n, (1.0 - alpha) * n)


def biquad(x: np.ndarray, freq: float, kind: str) -> np.ndarray:
    """Direct-form I, double precision, one sample at a time (operate(..., 1))."""
    b0, b1, b2, a1, a2 = _coeffs(freq, kind)
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(len(x)):
        v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, x[i]
        y2, y1 = y1, v
        y[i] = v
    return y


def clamp16(a: np.ndarray) -> np.ndarray:
    """fcvtzs + clamp to [0, 32767]."""
    return np.clip(np.trunc(a), 0, 32767)


def envelopes(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (high, low, all) int16 envelopes at 11025 Hz."""
    x = (samples[::4, 0].astype(np.float64) + samples[::4, 1]) * 0.5
    high = clamp16(biquad(np.abs(biquad(x, 2000, "hp")), 1000, "lp"))
    low = clamp16(biquad(np.abs(biquad(x, 100, "lp")), 25, "lp"))
    allb = clamp16(biquad(np.abs(x), 1000, "lp"))
    return high, low, allb


# ------------------------------------------------------------------ grid / score

def create_beat_data(bpm: float, start: int, length: int, count: int) -> np.ndarray:
    """UnitBeatAdjust::createBeatData -- nominal grid, integer 11025 Hz units."""
    step = SRD * 60.0 / bpm
    limit = length // 4
    pos = []
    for i in range(count):
        t = step * i + start
        if t >= limit:
            break
        pos.append(int(t + 0.5))
    t = step * count + start
    while t < limit:
        pos.append(int(t + 0.5))
        t += step
    return np.asarray(pos, dtype=np.int64)


def manage_wave(env: np.ndarray, grid: np.ndarray, count: int | None = None):
    """UnitBeatAdjust::manageWave -- score = max of the summed beat profile."""
    limit = len(env)
    pos = grid if count is None else grid[:count]
    pos = pos[(pos - HALF >= 0) & (pos + HALF <= limit)][1:-1]
    if len(pos) < 3:
        return -1.0, np.zeros(2 * HALF), len(pos)
    idx = pos[:, None] - HALF + np.arange(2 * HALF)[None, :]
    profile = env[idx].sum(axis=0)
    return float(profile.max()), profile, len(pos)


def bar_profiles(env: np.ndarray, grid: np.ndarray) -> list[np.ndarray]:
    """The four per-beat-in-bar profiles `beatPositionAdjust` builds."""
    limit = len(env)
    out = [np.zeros(2 * HALF) for _ in range(4)]
    pos = grid[(grid - HALF >= 0) & (grid + HALF <= limit)]
    for i, p in enumerate(pos):
        out[i % 4] += env[p - HALF:p + HALF]
    return out


def window_integral(env: np.ndarray, bpm: float, pos: int) -> float:
    """calcLowInteg / calcAllInteg."""
    n = int(SRD * 30.0 / bpm + 0.5)
    start = int(pos * 0.25 + 0.5)
    if n < 1 or start + n - 1 > len(env):
        return -1.0
    return float(env[start:start + n].sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio")
    ap.add_argument("--bpm", type=float, default=122.0)
    ap.add_argument("--start", type=int, default=0, help="grid start, 11025 units")
    ap.add_argument("--count", type=int, default=1024)
    ap.add_argument("--scan", nargs=2, type=float, metavar=("LO", "HI"),
                    help="scan BPM range, 0.001 steps, report the score peak")
    args = ap.parse_args()

    samples = load_i16(args.audio)
    high, low, allb = envelopes(samples)
    print(f"{args.audio}: {len(high)} envelope samples @ {SRD} Hz")

    step = SRD * 60.0 / args.bpm
    print(f"bpm {args.bpm:.6f} -> step {step:.6f} units ({step*4/44.1:.3f} ms)")

    if args.scan:
        lo, hi = args.scan
        bpms = np.arange(lo, hi + 1e-12, 0.001)
        scores = []
        for bpm in bpms:
            grid = create_beat_data(bpm, args.start, len(samples), args.count)
            scores.append(manage_wave(high, grid)[0])
        scores = np.asarray(scores)
        best = bpms[int(scores.argmax())]
        print(f"score peak: bpm {best:.3f}  score {scores.max():.0f} "
              f"(median {np.median(scores):.0f})")
        return

    grid = create_beat_data(args.bpm, args.start, len(samples), args.count)
    score, profile, used = manage_wave(high, grid)
    print(f"grid: {len(grid)} beats, {used} scored, score {score:.0f}, "
          f"profile peak at {int(profile.argmax()) - HALF:+d} units")
    for i, prof in enumerate(bar_profiles(high, grid)):
        print(f"  bar position {i + 1}: peak {int(prof.argmax()) - HALF:+5d} units "
              f"({(prof.argmax() - HALF) * 4 / 44.1:+7.2f} ms)")
    print(f"low-band integral at grid[1]: {window_integral(low, args.bpm, int(grid[1])):.0f}")
    print(f"all-band integral at grid[1]: {window_integral(allb, args.bpm, int(grid[1])):.0f}")


if __name__ == "__main__":
    main()
