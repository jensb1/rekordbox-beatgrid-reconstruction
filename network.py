"""Forward pass of rekordbox's detect_beat networks (madmom BLSTM folds 7/8).

The Keras model is the TF port of madmom's `downbeats_blstm_{7,8}.pkl`:
three bidirectional peephole-LSTM layers (25 units per direction) and a dense
softmax layer with 3 classes (non-beat, beat, downbeat).  `activations()`
returns the full softmax; rekordbox exposes columns 1:3 as beat/downbeat
activations at 100 fps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def sigmoid(x):
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    out[~positive] = ex / (1.0 + ex)
    return out


def lstm_forward(x, cell):
    """One direction of a peephole LSTM over x (frames, inputs)."""
    kernel = cell["kernel"]
    recurrent = cell["recurrent_kernel"]
    bias = cell["bias"]
    peep_i = cell["input_gate_peephole_weights"]
    peep_f = cell["forget_gate_peephole_weights"]
    peep_o = cell["output_gate_peephole_weights"]
    hidden = recurrent.shape[0]
    projected = x @ kernel + bias
    h = np.zeros(hidden, dtype=np.float32)
    c = np.zeros(hidden, dtype=np.float32)
    out = np.empty((x.shape[0], hidden), dtype=np.float32)
    for t in range(x.shape[0]):
        p = projected[t]
        i = sigmoid(p[:hidden] + h @ recurrent[:, :hidden] + peep_i * c)
        f = sigmoid(p[hidden : 2 * hidden] + h @ recurrent[:, hidden : 2 * hidden] + peep_f * c)
        g = np.tanh(p[2 * hidden : 3 * hidden] + h @ recurrent[:, 2 * hidden : 3 * hidden])
        c = g * i + c * f
        o = sigmoid(p[3 * hidden :] + h @ recurrent[:, 3 * hidden :] + peep_o * c)
        h = np.tanh(c) * o
        out[t] = h
    return out


def bidirectional(x, layer, index):
    base = f"bidirectional_{index}"
    forward = lstm_forward(
        x, {k.split("/")[-1]: v for k, v in layer.items() if k.startswith(f"{base}/forward_rnn_")}
    )
    backward = lstm_forward(
        x[::-1],
        {k.split("/")[-1]: v for k, v in layer.items() if k.startswith(f"{base}/backward_rnn_")},
    )[::-1]
    return np.hstack((forward, backward))


def activations(features: np.ndarray, model: str = "model7") -> np.ndarray:
    """Return (frames, 2) beat/downbeat activations for 314-dim features."""
    weights = np.load(HERE / "extracted" / f"{model}_weights.npz")
    layer = {k: weights[k] for k in weights.files}

    out = features.astype(np.float32)
    for index in (21, 22, 23):
        out = bidirectional(out, layer, index)
    logits = out @ layer["dense_7/kernel"] + layer["dense_7/bias"]
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("features", help=".npy file with (frames, 314) features")
    parser.add_argument("--model", default="model7", choices=("model7", "model8"))
    parser.add_argument("--out", default="activations.npy")
    args = parser.parse_args()
    np.save(args.out, activations(np.load(args.features), args.model)[:, 1:3])
