"""Rekordbox grid writer: position lattice -> ANLZ timestamps.

This module holds the part of rekordbox's final beat-grid writer that is
reconstructed exactly (verified against every reference beat, see
``tests/verify_exact.py``).

What is known exactly
---------------------
rekordbox does not store beat times as floating point seconds. A beat position
is an integer on a sample lattice, and the microsecond timestamp written to
ANLZ is that position divided by the lattice unit and *truncated*:

    us = pos * 1_000_000 // unit

``unit`` is the value of ``UnitBeatAdjust``'s ``[0xc]`` field, i.e.
``sampleRate / 4`` (11025 at 44.1 kHz) for some tracks and the raw sample rate
(44100) for others.

The ANLZ files then split that single value:

    PQTZ milliseconds = us // 1000
    PQT2 fraction     = us %  1000

so an exact writer has to reproduce ``pos``; every timestamp follows from it.

Verification
------------
Taking the reference lattices as given, the encoding above reproduces

    dodonpachi  (unit=11025)  930/930
    hutchula    (unit=44100) 1001/1001
    givmeluv    (unit=44100)  976/976

timestamps with zero mismatches. Rounding instead of truncating matches only
about half of them, which is why the earlier implementation was systematically
off by a microsecond.

What is not reconstructed
-------------------------
How ``pos`` itself is produced from audio (``createBeatData``,
``findBeatPositionForBeat``, ``offsetBeatPosition`` in ``BeatAnalyzer_2_0``) is
still open, so ``detect.constant_grid`` feeds a fitted lattice into the
functions below. The encoding is exact; the grid parameters are not.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 44100

#: ``UnitBeatAdjust`` [0xc]: analysis rate is sampleRate/4.
ANALYSIS_UNIT = SAMPLE_RATE // 4


def encode_timestamps(positions: np.ndarray, unit: int = SAMPLE_RATE) -> np.ndarray:
    """Map integer lattice positions to microsecond timestamps.

    ``us = pos * 1_000_000 // unit`` (truncation, integer arithmetic).
    """
    pos = np.asarray(positions, dtype=np.int64)
    return pos * 1_000_000 // int(unit)


def split_microseconds(us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split microsecond timestamps into (milliseconds, fraction)."""
    us = np.asarray(us, dtype=np.int64)
    return us // 1000, us % 1000


def lattice_positions(start: float, step: float, count: int,
                      unit: int = SAMPLE_RATE) -> np.ndarray:
    """Integer beat positions of a constant-BPM lattice.

    ``pos[k] = round(start + k * step)`` in ``unit``-samples, matching
    ``createBeatData``'s ``fcvtzs(x + 0.5)``.
    """
    k = np.arange(int(count), dtype=np.float64)
    # fcvtzs(x + 0.5): truncation of the biased value == round-half-up.
    return np.floor(float(start) + k * float(step) + 0.5).astype(np.int64)


def _self_test() -> None:
    # exact microseconds where pos*1e6 divides evenly
    assert encode_timestamps(np.array([22050]), 44100)[0] == 500000
    assert encode_timestamps(np.array([11025]), 11025)[0] == 1000000
    # truncation, not rounding: 1 sample past a half-microsecond
    assert encode_timestamps(np.array([1]), 44100)[0] == 22
    assert encode_timestamps(np.array([2]), 44100)[0] == 45
    ms, frac = split_microseconds(np.array([500000, 48616]))
    assert ms.tolist() == [500, 48] and frac.tolist() == [0, 616]
    # round-half-up lattice, like fcvtzs(x + 0.5)
    pos = lattice_positions(0.5, 2.5, 3, unit=4)
    assert pos.tolist() == [1, 3, 6]  # round-half-up, unlike numpy.rint


if __name__ == "__main__":
    _self_test()
    print("writer.py self-test OK")
