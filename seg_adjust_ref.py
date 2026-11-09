"""Instruction-level transliteration of BeatAnalyzer_2_0::peakss::BeatInfoAnalyzer::segAdjust.

Source: /tmp/rb2.dylib, function at 0x100f6a008 (~1,524 bytes).

This file is a *transliteration*, not a working port:
  * every block carries the address range it was read from;
  * blocks that have not been read yet raise NotImplementedError rather than
    being guessed (that is how the five retracted inferences in
    REVERSE_ENGINEERING.md happened -- guessing fills gaps silently);
  * constants are the exact literals from the binary.

Read so far: 0x100f6a008-0x100f6a1dc (setup + per-beat snap),
             0x100f6a1dc-0x100f6a300 (second snap + bounds),
             0x100f6a300-0x100f6a37c (walk setup),
             0x100f6a37c-0x100f6a4fc (candidate search + score accumulation).
Not yet read: 0x100f6a4fc-0x100f6a5f4 (end of the walk body and the tail).
"""

from __future__ import annotations

BEAT_STRIDE = 24

# literals from the prologue (0x100f6a054-0x100f6a0cc)
SCRATCH = 1024.0        # [this+0] written by the ctor; used as 2**-10 scale
DEFAULT_SR = 48000      # [this+4] written by the ctor
D_EIGHTH = 0.125        # d3
D_HALF = 0.5            # d4
D_ONE = 1.0             # d5
D_FOUR = 4.0            # d6
D_THREE = 3.0           # d7
D_WALK = 96.0           # d10, from x16 = 0x4058000000000000 (0x100f6a0c0)
D_TOKEN = 0x4058000000000000


def _ldr_s(rec_bytes: bytes, off: int) -> float:
    import struct
    return struct.unpack_from("<f", rec_bytes, off)[0]


def _ldr_w(rec_bytes: bytes, off: int) -> int:
    import struct
    return struct.unpack_from("<i", rec_bytes, off)[0]


def seg_adjust(this: dict, info: dict, beats: bytes, w3: int) -> None:
    """Transliteration.  `info[8]` is the candidate vector (24-byte records)."""
    n = len(beats) // BEAT_STRIDE

    # 0x100f6a048  cmp w13,#1 ; b.lt -> return
    if n < 1:
        return

    # 0x100f6a054  d0 = (double)[this+4] * 2**-10
    d0 = float(this.get(4, DEFAULT_SR)) * (2.0 ** -10)
    # 0x100f6a07c  d1 = (double)w3
    d1 = float(w3)

    # 0x100f6a0d4..  loop over beats
    for k in range(n):
        # 0x100f6a0e8  x1 = beats + k*24 ; d19 = (double)*(float*)(x1+8)
        d19 = _ldr_s(beats, k * BEAT_STRIDE + 8)
        # 0x100f6a0f4.. d23 = 0 + d19 ; d24 = d0 * d23
        d24 = d0 * d19

        # 0x100f6a0fc  x2 = x1 + 0x18   (next record) ; x3 = x1 - 0x18
        # 0x100f6a128  w5 = *(int*)x2   (next record's pos -> upper bound)
        if (k + 1) * BEAT_STRIDE + 4 <= len(beats):
            w5 = _ldr_w(beats, (k + 1) * BEAT_STRIDE)
        else:
            raise NotImplementedError("0x100f6a120: last beat path not read")

        # 0x100f6a1ac..0x100f6a1bc
        cand = info["candidates"]            # vector at info+8, 24-byte stride
        w13 = _ldr_w(beats, k * BEAT_STRIDE)  # current pos as lower bound
        w6 = int(d24 * D_EIGHTH)              # 0x100f6a1a0/1a4 fcvtzs
        # 0x100f6a1a8 add w13, w13, w6  -> position in the candidate domain
        pos = w13 + w6

        # 0x100f6a1bc..0x100f6a1d8  clamps: max(pos, *(float*)x1),
        # min(pos, w5), max(pos, 0)
        pos = max(pos, int(_ldr_s(beats, k * BEAT_STRIDE)))
        pos = min(pos, w5)
        pos = max(pos, 0)

        # 0x100f6a1dc-0x100f6a290: the second (backward) snap, same shape.
        # 0x100f6a29c-0x100f6a2c4: step preparation
        w12 = pos
        w13_step = int(d24 * D_EIGHTH)
        w5_step = w12 if w12 > w13_step else w13_step
        d28 = 0.5 * (float(w5) - float(pos))
        d29 = 1.0 / d24
        d26 = (float(w5_step) * d28) / d24
        d30 = d26 * d24
        d31 = min(d30 + float(pos), float(w5))

        # 0x100f6a2cc-0x100f6a4fc: walk
        d9 = float(pos)
        while d9 < d31:
            # 0x100f6a36c  w22 = (int)d12 (interpolated position)
            w22 = int(d9)
            # 0x100f6a414-0x100f6a448: binary search in `cand` for the last
            # entry <= w5+w22, then 0x100f6a46c-0x100f6a4d0 picks the nearest
            # entry with |cand - w22| <= w5, requiring cand <= w22.
            best, best_d = -1, 0x7FFFFFFF
            lo, hi = 0, len(cand) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if cand[mid] > w22:
                    hi = mid - 1
                else:
                    lo = mid + 1
            start = lo - 1
            i = start
            while i >= 0:
                d = w22 - cand[i]
                if d > w5:
                    break
                if d < best_d:
                    best, best_d = i, d
                i -= 1
            if best < 0:
                raise NotImplementedError("0x100f6a4d4: not-found path not read")
            cw13 = cand[best]
            if cw13 < 0:
                raise NotImplementedError("0x100f6a4e0: negative candidate")

            # 0x100f6a4e4-0x100f6a4f4  d10 += w5 - |cand - w22|
            d12 = float(w5 - abs(cw13 - w22))
            this["score"] = this.get("score", 0.0) + d12

            # 0x100f6a318-0x100f6a320  d9 += 96.0 ; loop while d9 < d31
            d9 += D_WALK
        # 0x100f6a4fc onward: NOT READ
        raise NotImplementedError("0x100f6a4fc-0x100f6a5f4 not read yet")


if __name__ == "__main__":
    print(__doc__)
    print("read blocks transliterated; unread blocks raise NotImplementedError")
