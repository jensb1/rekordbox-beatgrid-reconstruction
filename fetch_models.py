"""Download madmom's public downbeat models and convert them to the npz layout
used by network.py.

The models are CC BY-NC-SA 4.0 (CPJKU/madmom_models), not BSD. They are stored
separately from the MIT-licensed code; see models/README.md.

Usage:
  python fetch_models.py [--folds 7 8] [--out models]
"""

from __future__ import annotations

import argparse
import pickle
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://raw.githubusercontent.com/CPJKU/madmom_models/master/downbeats/2016"
LAYERS = (21, 22, 23)


class _Stub:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            self.__dict__.update(state[1])
        else:
            self.__dict__["_state"] = state


_CLASSES = {}


def _find_class(module, name):
    if module == "numpy.core.multiarray" and name == "_reconstruct":
        return np._core.multiarray._reconstruct
    if module == "numpy" and name == "ndarray":
        return np.ndarray
    if module == "numpy" and name == "dtype":
        return np.dtype
    key = (module, name)
    if key not in _CLASSES:
        _CLASSES[key] = type(name, (_Stub,), {})
    return _CLASSES[key]


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _find_class(module, name)


def load_madmom_model(path: Path):
    with path.open("rb") as fh:
        return _Unpickler(fh, encoding="latin1").load()


def convert(net) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    bidis = [l for l in net.layers if type(l).__name__ == "BidirectionalLayer"]
    feeds = [l for l in net.layers if type(l).__name__ == "FeedForwardLayer"]
    if len(bidis) != 3 or len(feeds) != 1:
        raise SystemExit(f"unexpected model layout: {[type(l).__name__ for l in net.layers]}")
    for layer_index, layer in zip(LAYERS, bidis):
        for direction in ("forward", "backward"):
            lstm = getattr(layer, "fwd_layer" if direction == "forward" else "bwd_layer")
            gates = [lstm.input_gate, lstm.forget_gate, lstm.cell, lstm.output_gate]
            base = (
                f"bidirectional_{layer_index}/{direction}_rnn_{layer_index}"
                f"/peephole_lstm_cell_{layer_index}"
            )
            out[f"{base}/kernel"] = np.concatenate(
                [g.weights for g in gates], axis=1
            ).astype(np.float32)
            out[f"{base}/recurrent_kernel"] = np.concatenate(
                [g.recurrent_weights for g in gates], axis=1
            ).astype(np.float32)
            out[f"{base}/bias"] = np.concatenate(
                [g.bias for g in gates]
            ).astype(np.float32)
            for name, gate in (
                ("input", lstm.input_gate),
                ("forget", lstm.forget_gate),
                ("output", lstm.output_gate),
            ):
                out[f"{base}/{name}_gate_peephole_weights"] = np.asarray(
                    gate.peephole_weights, dtype=np.float32
                )
    out["dense_7/kernel"] = np.asarray(feeds[0].weights, dtype=np.float32)
    out["dense_7/bias"] = np.asarray(feeds[0].bias, dtype=np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[7, 8])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "models")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        url = f"{BASE}/downbeats_blstm_{fold}.pkl"
        target = args.out / f"downbeats_blstm_{fold}.pkl"
        if not target.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, target)
        weights = convert(load_madmom_model(target))
        out = args.out / f"model{fold}_weights.npz"
        np.savez_compressed(out, **weights)
        print(f"wrote {out} ({len(weights)} tensors)")


if __name__ == "__main__":
    main()
