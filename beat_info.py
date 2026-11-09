"""Port of the *specified* parts of BeatAnalyzer_2_0::peakss::BeatInfoAnalyzer.

Only blocks that have been read off the disassembly and recorded in
REVERSE_ENGINEERING.md §§42-53 are implemented here; each function names its
section and says explicitly what is still unverified. Nothing in this module is
used by the generator yet.

BeatInfo (24 bytes, §42/§53):
    int   pos        offset 0    position in 1/44100 s units
    int   pad        offset 4
    float interval   offset 8    inter-beat interval, 6e6/bpm*100 (0.1 ms units)
    int   numBar     offset 12   bar number (1-4)
    int   x          offset 16
    int   flag       offset 20

Provenance:
    find_near_pos            §42   complete
    tempo_histogram          §§44-46   coarse 10 BPM then fine 1 BPM, exp10-based bins
    interval_filter          §47/§49  keep within +-4 %, accepting i, 2i, i/2
    segment_loop             §50   estimate -> filter -> emit -> repeat
"""

from __future__ import annotations

import numpy as np

BEAT_STRIDE = 24


class BeatInfo:
    __slots__ = ("pos", "pad", "interval", "num_bar", "x", "flag")

    def __init__(self, pos: int, interval: float, num_bar: int = 1,
                 pad: int = 0, x: int = -1, flag: int = 0) -> None:
        self.pos = int(pos)
        self.pad = int(pad)
        self.interval = float(interval)
        self.num_bar = int(num_bar)
        self.x = int(x)
        self.flag = int(flag)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"BeatInfo(pos={self.pos}, iv={self.interval:.4f}, bar={self.num_bar})"


def find_near_pos(beats: list[BeatInfo], target: int, tol: int) -> int:
    """§42: index of the beat closest to `target`, requiring pos <= target for
    the forward scan and |pos - target| <= tol. Returns -1 when nothing matches.

    The reference returns v[best].pos and writes the index through a pointer; we
    return the index only (the caller has the list).
    """
    n = len(beats)
    if n == 0:
        return -1
    lo, hi = 0, n - 1
    while lo <= hi:                      # last index with pos <= target
        mid = (lo + hi) // 2
        if beats[mid].pos > target:
            hi = mid - 1
        else:
            lo = mid + 1
    start = lo - 1
    if start < 0:
        return -1
    best, best_d = -1, 1 << 62
    for i in range(start, n):
        d = abs(beats[i].pos - target)
        if d > tol:
            break
        if beats[i].pos > target:
            break
        if d < best_d:
            best, best_d = i, d
    return best


def _histogram(beats: list[BeatInfo], scale: float, nbins: int) -> np.ndarray:
    """Common body of the tempo histograms (§§44-46).

    bin = int(((scale * 600000.0) / interval + 5.0) / 100)
    scale = exp10(0) = 1  -> 400 bins, round(bpm/10)
    scale = exp10(1) = 10 -> 4000 bins, round(bpm)
    """
    hist = np.zeros(nbins, dtype=np.int64)
    for b in beats:
        if b.interval <= 0.0:
            continue
        idx = int((scale * 600000.0) / b.interval + 5.0) // 100
        if 0 <= idx < nbins:
            hist[idx] += 1
    return hist


def tempo_histogram(beats: list[BeatInfo]) -> tuple[int, int]:
    """Coarse (10 BPM) then fine (1 BPM) tempo selection.

    Returns (coarse_bin, fine_bin); the fine bin is the BPM estimate.

    UNVERIFIED (§46): the window used for the fine pass, and whether the coarse
    index or a beat index is carried between the two passes. Here the fine pass
    runs over the whole list.
    """
    coarse = _histogram(beats, 1.0, 400)
    fine = _histogram(beats, 10.0, 4000)
    return int(np.argmax(coarse)), int(np.argmax(fine))


def tempo_interval_ms(bpm_bin: int) -> float:
    """Interval in milliseconds for a fine-histogram bin (§§46/§49).

    The histogram bins are `round(bpm)` computed from the record's interval in
    ms; the `600000/bin` the binary forms equals 10x that, i.e. 0.1 ms units.
    This helper keeps this module in ms, which is what BA_GetBeatList writes.
    """
    return 60000.0 / max(bpm_bin, 1)


def interval_filter(beats: list[BeatInfo], tempo_interval: float) -> list[BeatInfo]:
    """§47/§49: keep beats whose interval is within +-4 % of `tempo_interval`,
    also accepting the 2x and 0.5x harmonics; drop the rest.

    The second (tighter) band in the binary is NOT reproduced here: §47 describes
    it as centred on the tolerance rather than on the tempo, which is
    unconfirmed (§49).
    """
    tol = min(tempo_interval * 0.04, 20.0)
    lo, hi = tempo_interval - tol, tempo_interval + tol
    keep = []
    for b in beats:
        i = b.interval
        if lo < i < hi or lo < 2.0 * i < hi or lo < 0.5 * i < hi:
            keep.append(b)
    return keep


def segment_loop(beats: list[BeatInfo]) -> list[tuple[BeatInfo, int]]:
    """§50: estimate tempo on the remaining list, keep the consistent beats,
    emit the first of them, repeat on the leftovers.

    Returns [(emitted_beat, fine_bpm_bin), ...]. The exact emit/erase order is
    UNVERIFIED (§50/§51); this reproduces the described control flow only.
    """
    remaining = list(beats)
    out: list[tuple[BeatInfo, int]] = []
    while remaining:
        _, fine = tempo_histogram(remaining)
        tempo_interval = tempo_interval_ms(fine)
        consistent = interval_filter(remaining, tempo_interval)
        if not consistent:
            break
        out.append((consistent[0], fine))
        ids = {id(b) for b in consistent}
        remaining = [b for b in remaining if id(b) not in ids]
    return out


def _self_test() -> None:
    iv = 6000000.0 / 12200.0                     # 491.803..., as the analyser uses
    beats = [BeatInfo(pos=1000 * i + 536, interval=iv, num_bar=(i % 4) + 1)
             for i in range(200)]
    # find_near_pos
    assert find_near_pos(beats, 536, 10000) == 0
    assert find_near_pos(beats, 536 + 1000 * 5 + 3, 10) == 5
    assert find_near_pos([], 0, 10) == -1
    # histogram: 0.1 ms interval units -> 122 BPM
    coarse, fine = tempo_histogram(beats)
    assert coarse == 12, coarse                  # 122/10
    assert fine == 122, fine                     # 122
    # filter keeps tempo-consistent beats and their harmonics
    kept = interval_filter(beats + [BeatInfo(pos=0, interval=iv * 1.5)], iv)
    assert len(kept) == 200, len(kept)
    # segmentation emits at least the first segment
    segs = segment_loop(beats)
    assert segs and segs[0][1] == 122





# ---------------------------------------------------------------------------
# segAdjust (§56/§58/§59) -- partial port
# ---------------------------------------------------------------------------

SAMPLE_SCALE_DIV = 1024.0


def seg_adjust(beats: list[BeatInfo], candidates: list[int],
               sample_rate: int = 44100, beats_per_bar: int = 4) -> list[int]:
    """Port of the traced part of `BeatInfoAnalyzer::segAdjust`.

    Traced (§56, §58, §59):

        scale  = round(sample_rate * 1024 / 1000) / 1024        # 44.1 at 44.1 kHz
        pred_k = scale * cumsum(interval_ms)                    # samples
        off_k  = trunc(pred_k / 8)                              # 8-sample quantum
        i      = index in `candidates` nearest the neighbour's position
        pos_k  = candidates[i] +/- off_k, clamped to [first, last] and >= 0

    UNVERIFIED: which direction (+/-) applies, what the clamp bounds are exactly,
    and that `candidates` is the vector at `BeatInfoAnalyzeInfo+8` (a candidate
    grid rather than an audio peak list -- §60). This function exists so the
    formula is executable and can be tested the moment those are settled.
    """
    scale = round(sample_rate * SAMPLE_SCALE_DIV / 1000.0) / SAMPLE_SCALE_DIV
    if not beats or not candidates:
        return []
    cand = sorted(int(c) for c in candidates)
    lo, hi = cand[0], cand[-1]
    out: list[int] = []
    acc = 0.0
    prev = cand[0]
    for k, b in enumerate(beats):
        acc += b.interval
        pred = scale * acc
        off = int(pred / 8.0)
        # nearest candidate to the running position
        i = min(range(len(cand)), key=lambda j: abs(cand[j] - prev))
        pos = cand[i] + off if k == 0 else cand[i] - off
        pos = max(lo, min(hi, pos))
        pos = max(0, pos)
        out.append(pos)
        prev = cand[i]
    return out


def _self_test_seg() -> None:
    iv = 60000.0 / 122.0                      # ms per beat at 122 BPM
    beats = [BeatInfo(pos=0, interval=iv) for _ in range(9)]
    cands = [int(round(i * 44.1 * iv)) for i in range(9)]
    out = seg_adjust(beats, cands)
    assert len(out) == len(beats)
    assert out[0] >= 0 and all(p >= 0 for p in out)
    # the accumulation of 8 beats at 122 BPM is ~ 8*21688.5 samples
    acc = 44.1 * 8 * iv
    assert abs(acc - 8 * 21688.5) < 1.0, acc


if __name__ == "__main__":
    _self_test()
    _self_test_seg()
    print("beat_info.py self-tests OK")
