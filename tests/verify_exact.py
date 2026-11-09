"""Exactness gate for the reconstructed rekordbox beat-grid writer.

For every reference track (audio + rekordbox ANLZ), this script:

1. generates a grid from the audio only,
2. compares bar, BPM*100 and PQT2 microsecond timestamps,
3. exits 0 only if every value matches exactly.

The reference PQTZ/PQT2 values are used only as expected output; they are never
passed to the generator.

Usage:
  python tests/verify_exact.py
  python tests/verify_exact.py --only hutchula
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import detect  # noqa: E402
from features import load_audio  # noqa: E402

REFERENCE = Path(__file__).resolve().parent / "reference"


def tags(path: Path) -> dict[str, bytes]:
    data = path.read_bytes()
    if data[:4] != b"PMAI":
        raise ValueError(f"{path}: not an ANLZ file")
    at = struct.unpack(">I", data[4:8])[0]
    out: dict[str, bytes] = {}
    while at + 12 <= len(data):
        name = data[at : at + 4].decode("latin1")
        size = struct.unpack(">I", data[at + 8 : at + 12])[0]
        out[name] = data[at : at + size]
        at += size
    return out


def expected_grid(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dat = tags(folder / "ANLZ.DAT")
    q = dat["PQTZ"]
    count = struct.unpack(">I", q[20:24])[0]
    records = np.frombuffer(q, dtype=">u2", count=count * 4, offset=24).reshape(-1, 4)
    bars = records[:, 0].astype(np.int64)
    bpms = records[:, 1].astype(np.int64)
    ms = (records[:, 2].astype(np.int64) << 16) | records[:, 3].astype(np.int64)
    us = ms * 1000

    ext_path = folder / "ANLZ.EXT"
    if ext_path.exists():
        ext = tags(ext_path)
        if "PQT2" in ext:
            pqt2 = ext["PQT2"]
            n = struct.unpack(">I", pqt2[40:44])[0]
            if n != count:
                raise ValueError(f"{folder}: PQT2 count {n} != PQTZ count {count}")
            frac = np.frombuffer(pqt2, dtype=">u2", count=n, offset=56).astype(np.int64)
            us = us + frac
    return bars, bpms, us


def generated_grid(folder: Path, model: str, chunk_seconds: float):
    audio = load_audio(folder / "audio.m4a")
    duration = len(audio) / 44100.0
    beats = detect.detect(audio, model, chunk_seconds)
    grid = detect.constant_grid(beats, duration)
    bars = np.array([b["bar"] for b in grid["beats"]], dtype=np.int64)
    bpms = np.array([b["bpm"] for b in grid["beats"]], dtype=np.int64)
    us = np.array([b["us"] for b in grid["beats"]], dtype=np.int64)
    return bars, bpms, us


def compare(name: str, folder: Path, model: str, chunk_seconds: float) -> bool:
    e_bars, e_bpms, e_us = expected_grid(folder)
    a_bars, a_bpms, a_us = generated_grid(folder, model, chunk_seconds)
    ok = True
    print(f"\n{name}: expected {len(e_us)} beats, generated {len(a_us)}")
    if len(e_us) != len(a_us):
        print(f"  COUNT MISMATCH: expected {len(e_us)}, got {len(a_us)}")
        ok = False
    n = min(len(e_us), len(a_us))
    diffs = []
    for i in range(n):
        fields = []
        if e_bars[i] != a_bars[i]:
            fields.append(f"bar {e_bars[i]} != {a_bars[i]}")
        if e_bpms[i] != a_bpms[i]:
            fields.append(f"bpm {e_bpms[i]} != {a_bpms[i]}")
        if e_us[i] != a_us[i]:
            fields.append(f"us {e_us[i]} != {a_us[i]} (delta {a_us[i]-e_us[i]:+d})")
        if fields:
            diffs.append((i, fields, int(e_us[i]), int(a_us[i])))
    if diffs:
        ok = False
        print(f"  MISMATCHES: {len(diffs)}/{n}")
        for i, fields, eu, au in diffs[:10]:
            print(f"    beat {i}: " + "; ".join(fields))
        if len(diffs) > 10:
            print(f"    ... {len(diffs)-10} more")
    if ok:
        print("  EXACT MATCH")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model8", choices=("model7", "model8"))
    parser.add_argument("--chunk-seconds", type=float, default=60.0)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    folders = []
    for folder in sorted(REFERENCE.iterdir() if REFERENCE.exists() else []):
        if not folder.is_dir():
            continue
        if args.only and folder.name not in args.only:
            continue
        missing = [
            p.name
            for p in (folder / "audio.m4a", folder / "ANLZ.DAT")
            if not p.exists()
        ]
        if missing:
            print(f"{folder.name}: missing {', '.join(missing)}; skipping")
            continue
        folders.append(folder)
    if not folders:
        raise SystemExit(
            "No reference tracks found. Add tests/reference/<name>/audio.m4a and ANLZ.DAT."
        )

    results = [compare(f.name, f, args.model, args.chunk_seconds) for f in folders]
    if not all(results):
        raise SystemExit(1)
    print("\nAll reference tracks match exactly.")


if __name__ == "__main__":
    main()
