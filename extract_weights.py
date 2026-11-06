"""Extract the weights of rekordbox's detect_beat SavedModel without TensorFlow.

The checkpoint is a TensorBundle: variables.index is a LevelDB table whose
values are BundleEntryProto messages pointing into variables.data-*.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from protobuf_min import DTYPES, message, one, shape, text, varint

MODEL_ROOT = Path("/Applications/rekordbox 7/rekordbox.app/Contents/Resources/models")
TABLE_MAGIC = 0xDB4775248B80FB57
DTYPE_MAP = {
    1: np.float32,
    2: np.float64,
    3: np.int32,
    4: np.uint8,
    5: np.int16,
    6: np.int8,
    7: object,
    9: np.int64,
    10: np.bool_,
    19: np.float16,
}


def read_block(data: bytes, offset: int, size: int) -> list[tuple[bytes, bytes]]:
    if not size:
        return []
    # The TensorBundle table stores the body size in the BlockHandle and puts
    # the 5-byte trailer (compression type + crc32) after it.
    body = data[offset : offset + size]
    num_restarts = struct.unpack_from("<I", body, len(body) - 4)[0]
    end = len(body) - 4 - 4 * num_restarts
    pos = 0
    entries = []
    previous = b""
    while pos < end:
        shared, pos = varint(body, pos)
        non_shared, pos = varint(body, pos)
        value_len, pos = varint(body, pos)
        key = previous[:shared] + body[pos : pos + non_shared]
        previous = key
        pos += non_shared
        value = body[pos : pos + value_len]
        pos += value_len
        entries.append((key, value))
    return entries


def read_table(path: Path) -> dict[str, bytes]:
    data = path.read_bytes()
    footer = data[-48:]
    magic = struct.unpack_from("<Q", footer, 40)[0]
    assert magic == TABLE_MAGIC, hex(magic)
    metaindex_offset, p = varint(footer, 0)
    metaindex_size, p = varint(footer, p)
    index_offset, p = varint(footer, p)
    index_size, p = varint(footer, p)
    entries: dict[str, bytes] = {}
    for _, handle in read_block(data, index_offset, index_size):
        offset, p = varint(handle, 0)
        size, p = varint(handle, p)
        for key, value in read_block(data, offset, size):
            entries[key.decode("utf-8", "replace")] = value
    return entries


def entry_proto(raw: bytes) -> dict:
    msg = message(raw)
    dims = shape(one(msg, 2, b"")) if 2 in msg else ([], False)
    return {
        "dtype": DTYPES.get(one(msg, 1, 0), one(msg, 1, 0)),
        "shape": dims[0],
        "shard": one(msg, 3, 0),
        "offset": one(msg, 4, 0),
        "size": one(msg, 5, 0),
        "crc32c": one(msg, 6, 0),
    }


def load_model(name: str, out_dir: Path) -> dict[str, np.ndarray]:
    root = MODEL_ROOT / "detect_beat" / name
    variables = root / "variables"
    entries = read_table(variables / "variables.index")
    shards: dict[int, bytes] = {}
    weights: dict[str, np.ndarray] = {}
    meta = {}
    for key, raw in entries.items():
        if not key.endswith("/.ATTRIBUTES/VARIABLE_VALUE"):
            continue
        variable = key[: -len("/.ATTRIBUTES/VARIABLE_VALUE")]
        info = entry_proto(raw)
        if info["dtype"] == "string":
            continue
        shard = info["shard"]
        if shard not in shards:
            files = sorted(variables.glob(f"variables.data-{shard:05d}-of-*"))
            shards[shard] = files[0].read_bytes()
        dtype = DTYPE_MAP[int(next(k for k, v in DTYPES.items() if v == info["dtype"]))]
        count = int(np.prod(info["shape"])) if info["shape"] else 1
        array = np.frombuffer(
            shards[shard], dtype=dtype, count=count, offset=info["offset"]
        ).reshape(info["shape"])
        # Keep the Keras variable name without the SavedObject wrapper prefix.
        short = variable.split("/", 1)[1] if "/" in variable else variable
        weights[short] = array.copy()
        meta[short] = info
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}.npz", **weights)
    (out_dir / f"{name}.json").write_text(
        json.dumps({k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in weights.items()}, indent=2)
    )
    return weights


if __name__ == "__main__":
    import sys

    out = Path(__file__).parent / "extracted"
    for name in sys.argv[1:] or ["model7", "model8"]:
        weights = load_model(name, out)
        print(f"{name}: {len(weights)} tensors, {sum(w.size for w in weights.values())} values")
        for key in sorted(weights):
            print(f"  {key}: {weights[key].shape}")
