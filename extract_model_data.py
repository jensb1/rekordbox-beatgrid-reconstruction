"""Extract rekordbox's detect_beat model into plain numpy files.

Writes tools/python/beatgrid/extracted/:
  model7.npz, model8.npz        Keras variables, keyed variables/0..37
  model7_weights.npz, ...       the same weights under readable layer names
  filterbanks.npz               the three log filterbanks used for the input features

The model files are read from the installed rekordbox app; nothing is copied
into the repository by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from extract_weights import MODEL_ROOT, load_model
from protobuf_min import DTYPES, message, one, shape, tensor_array, text

HERE = Path(__file__).parent
OUT = HERE / "extracted"

LAYERS = [21, 22, 23]
GATES = ["kernel", "recurrent_kernel", "bias", "input_gate_peephole_weights",
         "forget_gate_peephole_weights", "output_gate_peephole_weights"]


def checkpoint_order() -> list[str]:
    names = []
    for layer in LAYERS:
        for direction in ("forward", "backward"):
            base = f"bidirectional_{layer}/{direction}_rnn_{layer}/peephole_lstm_cell_{layer}"
            names += [f"{base}/{gate}" for gate in GATES]
    names += ["dense_7/kernel", "dense_7/bias"]
    return names


def readable_weights(model: str) -> dict[str, np.ndarray]:
    variables = load_model(model, OUT)
    order = checkpoint_order()
    return {name: variables[f"variables/{i}"] for i, name in enumerate(order)}


def graph_functions(path: Path):
    sm = message(path.read_bytes())
    mg = message(sm[2][0])
    graph = message(mg[2][0])
    library = message(graph[2][0])
    for raw in library.get(1, []):
        f = message(raw)
        name = text(one(message(one(f, 1, b"")), 1, b""))
        yield name, [message(node) for node in f.get(3, [])]


def const_arrays(nodes):
    for node in nodes:
        if text(one(node, 2, b"")) != "Const":
            continue
        name = text(one(node, 1, b""))
        for entry in node.get(5, []):
            e = message(entry)
            if text(one(e, 1, b"")) != "value":
                continue
            attr = message(one(e, 2, b""))
            dtype, dims, values = tensor_array(one(attr, 8, b""))
            yield name, dtype, dims, values


def extract_filterbanks(model: str) -> dict[str, np.ndarray]:
    path = MODEL_ROOT / "detect_beat" / model / "saved_model.pb"
    wanted = {(512, 21): "fb1024", (1024, 45): "fb2048", (2048, 91): "fb4096"}
    found: dict[str, np.ndarray] = {}
    for _, nodes in graph_functions(path):
        for name, dtype, dims, values in const_arrays(nodes):
            key = tuple(dims)
            if key in wanted and values is not None and (name == "Const" or name.endswith("/Const")):
                found[wanted[key]] = np.asarray(values, dtype=np.float32)
    missing = set(wanted.values()) - set(found)
    if missing:
        raise SystemExit(f"missing filterbanks: {missing}")
    return found


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model in ("model7", "model8"):
        weights = readable_weights(model)
        np.savez_compressed(OUT / f"{model}_weights.npz", **weights)
        print(f"{model}: {len(weights)} named tensors")
        for name in ("dense_7/kernel", "dense_7/bias"):
            print(f"  {name}: {weights[name].shape}")
    filterbanks = extract_filterbanks("model7")
    np.savez_compressed(OUT / "filterbanks.npz", **filterbanks)
    print("filterbanks:", {k: v.shape for k, v in filterbanks.items()})

    # Sanity check: models 7 and 8 must be different trained folds.
    a = np.load(OUT / "model7_weights.npz")
    b = np.load(OUT / "model8_weights.npz")
    same = all(np.array_equal(a[k], b[k]) for k in a.files)
    print("model7 == model8:", same)


if __name__ == "__main__":
    main()
