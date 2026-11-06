"""Minimal protobuf wire-format reader.

Enough to walk TensorFlow SavedModel / GraphDef / TensorProto without the
TensorFlow install (which is ~1 GB and not needed for inspection).
"""

from __future__ import annotations


def varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, pos
        shift += 7


def fields(buf: bytes):
    """Yield (field_number, wire_type, value) tuples.

    wire type 0 -> int, 1 -> 8 raw bytes, 2 -> bytes, 5 -> 4 raw bytes.
    """
    pos = 0
    n = len(buf)
    while pos < n:
        key, pos = varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == 2:
            size, pos = varint(buf, pos)
            value, pos = buf[pos : pos + size], pos + size
        elif wire == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"wire type {wire} at {pos}")
        yield number, wire, value


def message(buf: bytes) -> dict[int, list]:
    out: dict[int, list] = {}
    for number, wire, value in fields(buf):
        out.setdefault(number, []).append(value)
    return out


def one(msg: dict[int, list], number: int, default=None):
    values = msg.get(number)
    return values[0] if values else default


def text(value: bytes) -> str:
    return value.decode("utf-8", "replace")


# TensorFlow DataType enum, only the ones that show up in these models.
DTYPES = {
    0: "invalid",
    1: "float32",
    2: "float64",
    3: "int32",
    4: "uint8",
    5: "int16",
    6: "int8",
    7: "string",
    8: "complex64",
    9: "int64",
    10: "bool",
    11: "qint8",
    12: "quint8",
    13: "qint32",
    14: "bfloat16",
    15: "qint16",
    16: "quint16",
    17: "uint16",
    18: "complex128",
    19: "float16",
    22: "uint32",
    23: "uint64",
}


def shape(buf: bytes) -> tuple[list[int | None], bool]:
    msg = message(buf)
    dims = []
    for dim in msg.get(2, []):
        d = message(dim)
        dims.append(one(d, 1, 0))
    return dims, bool(one(msg, 3, 0))


def tensor(buf: bytes) -> dict:
    msg = message(buf)
    result = {
        "dtype": DTYPES.get(one(msg, 1, 0), one(msg, 1, 0)),
        "shape": shape(one(msg, 2, b"")) if 2 in msg else ([], False),
        "content": one(msg, 4),
    }
    for name, number, fmt in (
        ("float_val", 5, "f"),
        ("double_val", 6, "d"),
        ("int_val", 7, "q"),
        ("int64_val", 10, "q"),
        ("bool_val", 11, "?"),
        ("half_val", 13, "H"),
    ):
        if number in msg:
            raw = b"".join(msg[number])
            # packed repeated values, but tolerate non-packed encodings
            if number in (5, 6, 11) and len(raw) != 4 * len(msg[number]):
                pass
            size = {"f": 4, "d": 8, "q": 8, "?": 1, "H": 2}[fmt]
            if len(raw) % size == 0:
                result[name] = list(__import__("struct").unpack(f"<{len(raw)//size}{fmt}", raw))
    return result


def tensor_array(buf: bytes):
    """Decode a TensorProto into a numpy array (without importing TF)."""
    import numpy as np

    msg = message(buf)
    dtype = DTYPES.get(one(msg, 1, 0), "float32")
    dims = shape(one(msg, 2, b""))[0] if 2 in msg else []
    count = int(np.prod(dims)) if dims else 1
    np_dtype = {
        "float32": np.float32, "float64": np.float64, "int32": np.int32,
        "int64": np.int64, "bool": np.bool_, "uint8": np.uint8,
        "int16": np.int16, "int8": np.int8, "float16": np.float16,
    }.get(dtype)
    if np_dtype is None:
        return dtype, dims, None
    if 4 in msg:
        raw = b"".join(msg[4])
        return dtype, dims, np.frombuffer(raw, dtype=np_dtype, count=count).reshape(dims)
    for field, np_type in ((5, np.float32), (6, np.float64), (7, np.int32), (10, np.int64), (11, np.bool_), (13, np.float16)):
        if field in msg:
            values = msg[field]
            # packed repeated values arrive as a single bytes blob, non-packed
            # as individual varints/fixed values
            if isinstance(values[0], bytes):
                raw = b"".join(values)
                if field in (7, 10):
                    # packed varints
                    decoded = []
                    pos = 0
                    while pos < len(raw):
                        value, pos = varint(raw, pos)
                        if value >= 1 << 63:
                            value -= 1 << 64
                        decoded.append(value)
                    arr = np.asarray(decoded, dtype=np_type)
                else:
                    size = np.dtype(np_type).itemsize
                    arr = np.frombuffer(raw[: (len(raw) // size) * size], dtype=np_type)
            else:
                arr = np.asarray(values, dtype=np_type)
            if count == arr.size:
                return dtype, dims, arr.reshape(dims)
            return dtype, dims, None
    return dtype, dims, None
