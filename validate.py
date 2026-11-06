"""Prove that the local detector matches rekordbox / madmom.

Three independent checks:

1. Features: our numpy feature chain vs the 314-dim `output_1` of
   rekordbox's own TensorFlow SavedModel.
2. Network: our numpy BLSTM vs the `output_2` of the SavedModel (using the
   same features, so this isolates the network).
3. Decoder: our HMM vs madmom's DBNDownBeatTrackingProcessor on the same
   activations.

Run with a Python that has tensorflow and/or madmom installed, e.g.:

  /tmp/mmvenv/bin/python validate.py track.mp3 --seconds 60 \
      --tf "/Applications/rekordbox 7/rekordbox.app/Contents/Resources/models/detect_beat/model8"

Without --tf/--madmom the corresponding checks are skipped.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import hmm  # noqa: E402
from features import load_audio, features, load_filterbanks  # noqa: E402
from network import activations  # noqa: E402


def report(name, a, b):
    diff = np.abs(np.asarray(a) - np.asarray(b))
    print(f"{name}: shape={diff.shape} max={diff.max():.3e} mean={diff.mean():.3e}")
    return diff.max()


def check_tf(audio, model_path):
    try:
        import tensorflow as tf
    except ImportError:
        print("tensorflow not installed, skipping TF check")
        return False
    model = tf.saved_model.load(model_path)
    out = model.signatures["serving_default"](
        fltp=tf.constant(audio[None, :], dtype=tf.float32)
    )
    tf_features = out["output_1"].numpy().astype(np.float32)
    tf_activations = out["output_2"].numpy().astype(np.float32)

    our_features = features(audio, load_filterbanks())
    report("features vs rekordbox SavedModel", our_features, tf_features)

    # Use the SavedModel's features for the network check, so the two checks
    # are independent.
    model_name = Path(model_path.rstrip("/")).name
    our_activations = activations(tf_features, model_name)[:, 1:3]
    report("network vs rekordbox SavedModel", our_activations, tf_activations)

    our_end_to_end = activations(our_features, model_name)[:, 1:3]
    report("full feature+network vs SavedModel", our_end_to_end, tf_activations)
    return True


def check_madmom(activations_array):
    try:
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    except ImportError:
        print("madmom not installed, skipping HMM check")
        return False
    processor = DBNDownBeatTrackingProcessor(
        beats_per_bar=[4], fps=100, min_bpm=70, max_bpm=180, num_tempi=60,
        transition_lambda=100, observation_lambda=16, threshold=0.05,
        correct=True,
    )
    reference = processor(activations_array[:, 1:3])
    ours = hmm.downbeat_tracking(activations_array[:, 1:3])
    print(f"madmom beats: {reference.shape}, our beats: {ours.shape}")
    if ours.shape != reference.shape:
        print("HMM check: SHAPE MISMATCH")
        return False
    equal = np.array_equal(ours, reference)
    print(f"HMM check: max diff {np.abs(ours - reference).max():.3e}, exact={equal}")
    return equal


def parse_pqtz(path):
    data = Path(path).read_bytes()
    if data[:4] != b"PMAI":
        raise ValueError("not an ANLZ file")
    at = struct.unpack(">I", data[4:8])[0]
    tags = {}
    while at + 12 <= len(data):
        name = data[at : at + 4].decode("latin1")
        size = struct.unpack(">I", data[at + 8 : at + 12])[0]
        tags[name] = data[at : at + size]
        at += size
    q = tags["PQTZ"]
    n = struct.unpack(">I", q[20:24])[0]
    beats = []
    for i in range(n):
        bar, bpm, ms = struct.unpack(">HHI", q[24 + 8 * i : 32 + 8 * i])
        beats.append((ms / 1000.0, bar, bpm / 100.0))
    return np.array(beats)


def check_pqtz(audio, dat_path):
    reference = parse_pqtz(dat_path)
    beats = hmm.downbeat_tracking(detect_activations(audio))
    print(f"rekordbox PQTZ: {len(reference)} beats, first {reference[0,0]:.3f}s")
    print(f"our beats:      {len(beats)} beats, first {beats[0,0]:.3f}s")
    print("NOTE: a manually edited rekordbox grid is not ground truth for the "
          "automatic detector; compare only auto-analysed tracks.")


def detect_activations(audio, model="model8"):
    return activations(features(audio, load_filterbanks()), model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--model", default="model8", choices=("model7", "model8"))
    parser.add_argument("--tf", help="path to rekordbox's detect_beat/modelN directory")
    parser.add_argument("--madmom", action="store_true")
    parser.add_argument("--pqtz", help="ANLZ.DAT with a rekordbox PQTZ grid")
    args = parser.parse_args()

    audio = load_audio(args.audio)
    if args.seconds:
        audio = audio[: int(args.seconds * 44100)]
    print(f"audio: {len(audio)/44100:.1f}s")

    if args.tf:
        check_tf(audio, args.tf)
    if args.madmom:
        act = detect_activations(audio, args.model)
        check_madmom(act)
    if args.pqtz:
        check_pqtz(audio, args.pqtz)


if __name__ == "__main__":
    main()
