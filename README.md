# rekordbox beat-grid detector — reconstruction

A Python reconstruction of the beat/downbeat detector used by **rekordbox 7** for
automatic beat-grid analysis.

The short version: rekordbox’s high-precision analysis is a C++/TensorFlow port
of **madmom’s joint beat and downbeat tracker** (Böck, Krebs & Widmer, ISMIR
2016). This repository contains the reconstructed feature pipeline, BLSTM
network, DBN/HMM decoder and a CLI that produces beat/downbeat positions from an
audio file.

It is intended for interoperability and research. It is not affiliated with or
endorsed by AlphaTheta/Pioneer DJ.

## What rekordbox does

The installed rekordbox 7 app contains:

- `madmom::analyze`, `madmom::downbeat_tracking`
- `madmom::BarStateSpace`, `madmom::RNNDownBeatTrackingObservationModel`
- `madmom::HiddenMarkovModel<...>::viterbi`
- `models/detect_beat/model7/` and `model8/`

The two SavedModels are TF/Keras ports of madmom’s public models
`downbeats_blstm_7.pkl` and `downbeats_blstm_8.pkl`. The extracted weights are
float-for-float identical to those models.

The detector parameters recovered from the binary are:

| parameter | value |
|---|---|
| sample rate | 44100 Hz |
| fps | 100 |
| STFT frame sizes | 1024, 2048, 4096 |
| STFT hop | 441 |
| filterbank | log, 3/6/12 bands per octave, 30–17000 Hz |
| features | 314 dims (`2 × (21+45+91)`) |
| network | 3 × bidirectional peephole LSTM, 25 units/direction |
| output | softmax, columns 1:2 = beat, downbeat |
| beats per bar | 4 |
| min/max BPM | 70 / 180 |
| num tempi | 60 |
| transition lambda | 100 |
| observation lambda | 16 |
| activation threshold | 0.05 |
| beat correction | on |

## Proof

Run `validate.py` with TensorFlow and madmom installed. It compares against
rekordbox’s own TensorFlow SavedModel and madmom’s `DBNDownBeatTrackingProcessor`.

On a 60 s chunk of a test track:

```
features vs rekordbox SavedModel: max=2.079e-06 mean=3.984e-08
network vs rekordbox SavedModel:  max=5.186e-06 mean=2.978e-08
full feature+network vs SavedModel: max=3.517e-06 mean=2.804e-08
madmom beats: (129, 2), our beats: (129, 2)
HMM check: max diff 0.000e+00, exact=True
```

So the feature extraction, network and HMM are identical to
rekordbox/madmom to float32 rounding.

## Usage

Requirements: Python 3.11+, `numpy`, `ffmpeg` on `PATH`.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# detect beats and a constant grid (uses the committed models in models/)
.venv/bin/python detect.py track.mp3 --model model8 --chunk-seconds 60 --grid --json beats.json
```

The model weights and filterbanks are committed in `models/`. To refresh them
directly from madmom's public model repository:

```sh
.venv/bin/python fetch_models.py
```

`extract_model_data.py` is the alternative path: it reads the same weights and
filterbanks from a locally installed rekordbox app:

```
/Applications/rekordbox 7/rekordbox.app/Contents/Resources/models/detect_beat/modelN
```

Generated files go to `extracted/` and are ignored by git.

## Validation setup

```sh
python3.11 -m venv /tmp/mmvenv
/tmp/mmvenv/bin/pip install numpy scipy tensorflow==2.16.1 cython
/tmp/mmvenv/bin/pip install "git+https://github.com/CPJKU/madmom.git"

/tmp/mmvenv/bin/python validate.py track.mp3 --seconds 60 \
  --model model8 \
  --tf "/Applications/rekordbox 7/rekordbox.app/Contents/Resources/models/detect_beat/model8" \
  --madmom
```

## Status / limitation

The detector is exact. The remaining difference to a rekordbox-written ANLZ
grid is the grid writer, `BeatAnalyzer_2_0::UnitBeatAdjust`, which is only
partially reconstructed. `detect.py --grid` implements a constant-BPM fit with a
recovered half-window offset; it matches the ANLZ PQTZ BPM and beat count, but
the phase and the PQT2 microseconds are not bit-identical.

What *is* reconstructed exactly is the timestamp encoding. A beat position is
an integer on a sample lattice, and the microsecond value in ANLZ is that
position divided by the lattice unit and **truncated**:

```
us = pos * 1_000_000 // unit          # floor, not round
ms = us // 1000,  frac = us % 1000    # PQTZ and PQT2 are one value split
```

`unit` is `UnitBeatAdjust`'s `[0xc]` field (`sampleRate/4 = 11025`) for one
reference track and the raw sample rate (44100) for the other two. With the
reference lattices as given, this reproduces 930/930, 1001/1001 and 976/976
reference timestamps with zero mismatches; rounding instead of truncating
matches only about half of them. See `writer.py` for the implementation and a
self-test.

The detector and encoding are therefore exact; what is *not* reconstructed is
how `pos` is produced from audio (`createBeatData`, `findBeatPositionForBeat`,
`offsetBeatPosition`). `detect.py --grid` feeds a fitted lattice into the exact
encoder, so the format matches while the grid parameters do not:

* `PQTZ` stores a single constant BPM per track (`12200`, `12300`) and every beat
  repeats it; the beat *positions*, however, correspond to a slightly different
  rate (e.g. 122.00051 for a stored 122.00, 123.00038 for a stored 123.00), i.e.
  the positions are not produced by evaluating the stored BPM — they come from
  accumulating per-beat intervals, which is why the grid is a near-lattice rather
  than an exact one;
* it is not on `getPeakBpm`'s 0.001 BPM search grid nor on `bpmAdjust`'s 0.05
  grid, so a further refinement downstream of both exists;
* for Hutchula and Giv Me Luv the reference positions are only ever +-1 unit
  away from a constant-step lattice with a periodic (~8-beat) deviation pattern.

What the writer does *not* do is also measured: driven directly, both analyser
versions in the installed build (`BeatAnalyzer_1_0` and `2_0`) return the beat
positions they are given, unchanged (`bpmAdjust` returns its input BPM to the last
bit, and `segAdjust` / `gapCorrect` only re-space the list they receive). So the
writer shifts, filters and extends a beat list; it does not create the grid. The
sample-precision positions must therefore already exist in the analysis output —
and the analysis path reachable in this build emits beats on a 10 ms frame grid,
which cannot express them. Reproducing the reference grids from audio would
require the analysis stage that produces sub-frame positions, which is not
reachable through the installed build's public entry points.

The exactness gate is `tests/verify_exact.py`. It generates the grid from audio
only and compares bar, BPM×100 and PQT2 microseconds against local reference
ANLZ files. Current state on the three reference tracks:

```
                mismatching beats   first mismatch
dodonpachi      914 / 930           beat 2,  +23 us
givmeluv        957 / 976           beat 0, +454 us
hutchula       1001 / 1001          beat 0, -454 us
```

So the remaining gap is the fitted phase/period of the lattice, not the
BLSTM/HMM detector and not the timestamp encoding. The limitation stays until
`tests/verify_exact.py` prints `EXACT MATCH` for every reference track.

## Licensing

The source code in this repository is MIT-licensed (see `LICENSE`).

The model data in `models/` is **CC BY-NC-SA 4.0** (see `models/README.md` and
`models/LICENSE-CC-BY-NC-SA-4.0.txt`). It is madmom’s public model data; it is
not MIT-licensed and may not be used commercially without permission from the
original rights holders.

## Files

```
models/                  madmom folds 7/8 (CC BY-NC-SA 4.0) + filterbanks
fetch_models.py          download/convert the public madmom models
protobuf_min.py          minimal protobuf reader (no TensorFlow needed)
extract_weights.py       read TensorBundle variables.index/data
extract_model_data.py    extract weights + filterbanks to extracted/*.npz
features.py              audio -> 314-dim madmom features
network.py               BLSTM + dense forward pass
hmm.py                   madmom DBN/HMM decoder
detect.py                end-to-end CLI
validate.py              proof against TensorFlow and madmom
inspect_model.py         inspect the SavedModel graph
```

## Attribution

The HMM/decoder logic follows **madmom** (BSD-3-Clause, CPJKU) and the model
weights are madmom’s public downbeat models (folds 7 and 8). See `NOTICE.md`.
No rekordbox binaries, user databases, audio or artwork are included.
