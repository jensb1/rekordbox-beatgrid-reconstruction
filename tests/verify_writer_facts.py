"""Regression test for the *verified* part of the writer reconstruction.

This does not test exactness (that is `verify_exact.py`, which must fail while
the writer is incomplete). It pins the facts that are proven, so they cannot
silently regress:

1. the ANLZ timestamp encoding

       us = pos * 1_000_000 // unit          (truncation, integer arithmetic)
       ms = us // 1000,  frac = us % 1000    (PQTZ and PQT2 are one value split)

   must reproduce every reference microsecond given integer lattices positions,
   for unit = 11025 and 44100;

2. rounding instead of truncation must NOT reproduce them (so the test cannot
   pass by accident if the encoding is changed back);

3. the reference positions of each track against the recorded lattice
   (start, step): dodonpachi is an exact lattice (0 deviations), hutchula and
   givmeluv deviate on a known, small set of beats (see delta_fixture.json).

Exits 0 when all of the above hold.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference"

# recorded lattice parameters (see REVERSE_ENGINEERING.md sections 25-26)
LATTICE = {
    "dodonpachi": dict(unit=11025, start=536, step=5422.137904469),
    "hutchula": dict(unit=44100, start=2115, step=21688.43429158667),
    "givmeluv": dict(unit=44100, start=2258, step=21512.128184715),
}


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


def reference_us(folder: Path) -> np.ndarray:
    q = tags(folder / "ANLZ.DAT")["PQTZ"]
    count = struct.unpack(">I", q[20:24])[0]
    rec = np.frombuffer(q, dtype=">u2", count=count * 4, offset=24).reshape(-1, 4)
    ms = (rec[:, 2].astype(np.int64) << 16) | rec[:, 3].astype(np.int64)
    us = ms * 1000
    ext = tags(folder / "ANLZ.EXT")
    if "PQT2" in ext:
        p = ext["PQT2"]
        us = us + np.frombuffer(p, dtype=">u2", count=count, offset=56).astype(np.int64)
    return us


def positions_for(us: np.ndarray, unit: int) -> np.ndarray:
    """Integer positions consistent with us = pos*1e6//unit (unique if any)."""
    pos = np.floor(us * unit / 1_000_000).astype(np.int64)
    ok = (pos * 1_000_000) // unit == us
    pos[~ok] += 1
    if not ((pos * 1_000_000) // unit == us).all():
        raise AssertionError("microseconds are not on an integer lattice")
    return pos


def main() -> None:
    failures: list[str] = []
    for name, lat in LATTICE.items():
        folder = REFERENCE / name
        if not folder.exists():
            print(f"{name}: no reference data, skipped")
            continue
        us = reference_us(folder)
        unit, start, step = lat["unit"], lat["start"], lat["step"]

        # 1. truncation encoding reproduces every timestamp
        pos = positions_for(us, unit)
        got = pos * 1_000_000 // unit
        if not np.array_equal(got, us):
            failures.append(f"{name}: encoding mismatch {(got != us).sum()}/{len(us)}")
        ms, frac = got // 1000, got % 1000
        if not np.array_equal(ms * 1000 + frac, us):
            failures.append(f"{name}: ms/frac split does not round-trip")

        # 2. rounding must be wrong (guards against a silent revert)
        rounded = np.floor(pos * 1_000_000 / unit + 0.5).astype(np.int64)
        if np.array_equal(rounded, us):
            failures.append(f"{name}: rounding also matches - encoding test is vacuous")

        # 3. lattice deviations match the recorded set
        k = np.arange(len(us))
        lat_pos = start + np.floor(k * step + 0.5).astype(np.int64)
        dev = int((pos != lat_pos).sum())
        expected = 0 if name == "dodonpachi" else (28 if name == "hutchula" else 142)
        if dev != expected:
            failures.append(f"{name}: lattice deviations {dev}, expected {expected}")

        print(f"{name}: beats={len(us)} unit={unit} encoding=exact "
              f"lattice_deviations={dev} (expected {expected})")

    fixture = ROOT / "delta_fixture.json"
    if fixture.exists():
        json.loads(fixture.read_text())
        print(f"delta_fixture.json: parsed")

    if failures:
        print("\nFAIL")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nOK - verified writer facts hold")


if __name__ == "__main__":
    main()
