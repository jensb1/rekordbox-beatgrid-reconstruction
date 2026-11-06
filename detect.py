"""End-to-end high-precision beat detection, as used by rekordbox 7.

Pipeline (recovered from rekordbox 7.2.18):

  audio -> 314-dim madmom features -> BLSTM fold 7 or 8 -> beat/downbeat
  activations -> DBNDownBeatTrackingProcessor HMM -> beat positions

The detector is a 1:1 copy; see validate.py for the checks against madmom and
against rekordbox's own TensorFlow SavedModel.

Usage:
  python detect.py track.mp3 --model model8 --json beats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import hmm
from features import FPS, HOP, load_audio, load_filterbanks, features
from network import activations


def detect_activations(signal: np.ndarray, model: str = "model8",
                       chunk_seconds: float = 60.0,
                       overlap_seconds: float = 30.0) -> np.ndarray:
    """Run the network, optionally in rekordbox's overlapping chunks."""
    filterbanks = load_filterbanks()
    rate = 44100
    total_frames = int(np.ceil(len(signal) / HOP)) + 1
    if chunk_seconds <= 0 or len(signal) <= chunk_seconds * rate:
        return activations(features(signal, filterbanks), model)

    chunk = int(chunk_seconds * rate)
    step = int((chunk_seconds - overlap_seconds) * rate)
    accum = np.zeros((total_frames, 2), dtype=np.float64)
    weight = np.zeros(total_frames, dtype=np.float64)
    start = 0
    while start < len(signal):
        stop = min(len(signal), start + chunk)
        # keep the tail chunk full-sized where possible
        if stop - start < chunk and start > 0:
            start = max(0, stop - chunk)
        piece = signal[start:stop]
        act = activations(features(piece, filterbanks), model)[:, 1:3]
        frame = int(round(start / HOP))
        end = min(total_frames, frame + len(act))
        accum[frame:end] += act[: end - frame]
        weight[frame:end] += 1.0
        if stop >= len(signal):
            break
        start += step
    weight[weight == 0] = 1.0
    return (accum / weight[:, None]).astype(np.float32)


def detect(signal: np.ndarray, model: str = "model8",
           chunk_seconds: float = 60.0) -> np.ndarray:
    return hmm.downbeat_tracking(detect_activations(signal, model, chunk_seconds))


FOCUS_OFFSET = 2048 / 44100  # half of the 4096-sample STFT window


def constant_grid(beats: np.ndarray, duration: float,
                  bpm: float | None = None,
                  phase: float | None = None) -> dict:
    """Fit the constant-BPM grid.

    Recovered from rekordbox's focus step on untouched auto-analysed tracks:

    - BPM is the least-squares tempo rounded to a whole BPM.
    - The grid is offset by half the 4096-sample STFT window (46.44 ms).

    The remaining focus step (`beatPositionAdjust`) nudges the phase by a few
    milliseconds using low-pass filtered audio; see README.md.
    """
    times = beats[:, 0]
    if bpm is None:
        slope, _ = np.polyfit(np.arange(len(times)), times, 1)
        bpm = float(np.round(60.0 / slope))
    period = 60.0 / bpm
    if phase is None:
        phase = float(times[0] + FOCUS_OFFSET)
    if phase < 0:
        phase += period
    # allow a beat just past the container's reported duration (encoder delay)
    count = int(np.floor((duration + 0.03 - phase) / period)) + 1
    us = np.round((phase + np.arange(count) * period) * 1e6).astype(np.int64)
    return {
        "bpm": bpm,
        "count": int(len(us)),
        "first": float(us[0] / 1e6),
        "beats": [
            {"bar": (i % 4) + 1, "bpm": int(round(bpm * 100)), "us": int(u)}
            for i, u in enumerate(us)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--model", default="model8", choices=("model7", "model8"))
    parser.add_argument("--chunk-seconds", type=float, default=60.0,
                        help="0 disables chunking")
    parser.add_argument("--seconds", type=float, default=0,
                        help="only analyse the first N seconds (debug)")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--grid", action="store_true",
                        help="also emit a constant-BPM grid")
    args = parser.parse_args()

    signal = load_audio(args.audio)
    if args.seconds:
        signal = signal[: int(args.seconds * 44100)]
    duration = len(signal) / 44100.0
    beats = detect(signal, args.model, args.chunk_seconds)
    result = {
        "model": args.model,
        "duration": duration,
        "count": int(len(beats)),
        "first": float(beats[0, 0]) if len(beats) else None,
        "last": float(beats[-1, 0]) if len(beats) else None,
        "beats_per_bar": 4,
        "beats": beats.tolist(),
    }
    if args.grid and len(beats) > 1:
        result["grid"] = constant_grid(beats, duration)
    summary = {k: v for k, v in result.items() if k not in ("beats", "grid")}
    if "grid" in result:
        summary["grid"] = {k: v for k, v in result["grid"].items() if k != "beats"}
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
