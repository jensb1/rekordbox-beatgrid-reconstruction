"""Instruction-level transliteration of BeatInfoAnalyzer::gapCorrect (v2).

Source: /tmp/rb2.dylib, 0x100f6a5fc (1,168 bytes).
Signature (from the call sites): gapCorrect(this, vector<BeatInfo>&, int, int, int).

Read so far: 0x100f6a5fc-0x100f6a748.  Unread blocks raise NotImplementedError.

Note on units: the interval written is `step / ([this+4] * 2**-10)` and
`[this+4]` is the constructor value 48000 for this call path (no store to it
exists in the analyser range -- re-checked with both `#4]` and `#0x4]` forms).
That makes the produced interval 21688/46.875 = 462.67 for a 122 BPM beat, which
does NOT match BA_GetBeatList's 6e6/bpm*100 = 491.8. Recording the formula
verbatim instead of adjusting it to fit an assumption.
"""

from __future__ import annotations

BEAT_STRIDE = 24
DEFAULT_SR = 48000
SCALE = 2.0 ** -10


def _pos(buf: bytes, i: int) -> int:
    import struct
    return struct.unpack_from("<i", buf, i * BEAT_STRIDE)[0]


def _field10(buf: bytes, i: int) -> int:
    import struct
    return struct.unpack_from("<i", buf, i * BEAT_STRIDE + 0x10)[0]


def gap_correct(this: dict, beats: bytes, w2: int, w3: int, w4: int) -> bytes:
    import struct

    n = len(beats) // BEAT_STRIDE
    # 0x100f6a630  w9 = max(w3, 0)
    w9 = w3 if w3 > 0 else 0
    # 0x100f6a638..0x100f6a65c  if w4 < 0: w4 = n - 1
    if w4 < 0:
        w4 = n - 1

    # 0x100f6a66c..0x100f6a698
    d0 = float(_pos(beats, w9))            # start pos
    d11 = float(_pos(beats, w4))           # end pos
    w26 = w4 - w9                          # number of beats spanned
    d1 = float(w26)
    d8 = (d11 - d0) / d1                   # average step

    # 0x100f6a69c..0x100f6a6a8
    w27 = int(d0 / d8)                     # fcvtzs(d0/d8)
    d9 = d8 * float(int(d0 / d8)) - d0     # fmsub: remainder (negative)
    if w27 < 0:                            # 0x100f6a6ec tbnz
        raise NotImplementedError("0x100f6a6ec: w27 < 0 path not read")

    # 0x100f6a6ac..0x100f6a6d4
    w13 = _field10(beats, w9)
    w9f = _field10(beats, w4)
    # 0x100f6a6bc..0x100f6a6e4  interval = step / (this[4] * 2**-10)
    d0s = float(this.get(4, DEFAULT_SR)) * SCALE
    interval = d8 / d0s                    # fcvt s10 = (float)interval

    # 0x100f6a704  w28 = (int)d9   (the remainder, as an int)
    w28 = int(d9)

    # 0x100f6a708..0x100f6a72c  push one record:
    #   pos = w28, pad = 0, interval = s10, [+0xc] = -1, [+0x10] = -1, flag = 0
    rec = bytearray(BEAT_STRIDE)
    struct.pack_into("<i", rec, 0x00, w28)
    struct.pack_into("<i", rec, 0x04, 0)
    struct.pack_into("<f", rec, 0x08, interval)
    struct.pack_into("<i", rec, 0x0C, -1)
    struct.pack_into("<i", rec, 0x10, -1)
    rec[0x14] = 0

    records = [bytes(rec)]

    # 0x100f6a7c4-0x100f6a7e0: emit w27 further records, advancing by d8 each time
    while w27 != 0:
        d9 += d8
        w27 -= 1
        w28 = int(d9)                      # fcvtzs
        r = bytearray(BEAT_STRIDE)
        struct.pack_into("<i", r, 0x00, w28)
        struct.pack_into("<i", r, 0x04, 0)
        struct.pack_into("<f", r, 0x08, interval)
        struct.pack_into("<i", r, 0x0C, -1)
        struct.pack_into("<i", r, 0x10, -1)
        r[0x14] = 0
        records.append(bytes(r))

    # 0x100f6a7e4-0x100f6a808: mark the record at index w27 and stamp w13
    idx = w27 if w27 >= 0 else 0
    r = bytearray(records[idx])
    r[0x14] = 1                             # strb w10(=1), [x9,#0x14]
    struct.pack_into("<i", r, 0x10, w13)    # str w13, [x9,#0x10]
    records[idx] = bytes(r)

    # 0x100f6a80c-0x100f6a850: emit w26 further records, same shape, plus two
    # bytes copied from the stack ([sp+0x1d], [sp+0x1f]) to +0x15 and +0x17.
    for _ in range(max(w26, 0)):
        d9 += d8
        w23 = int(d9)
        r = bytearray(BEAT_STRIDE)
        struct.pack_into("<i", r, 0x00, w23)
        struct.pack_into("<i", r, 0x04, 0)
        struct.pack_into("<f", r, 0x08, interval)
        struct.pack_into("<i", r, 0x0C, -1)
        struct.pack_into("<i", r, 0x10, -1)
        r[0x14] = 0
        records.append(bytes(r))

    # 0x100f6a91c onward: tail not read yet (post-loop bookkeeping)
    return b"".join(records)


if __name__ == "__main__":
    print(__doc__)
