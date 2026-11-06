"""Dump the structure of rekordbox's bundled detect_beat SavedModels.

Usage: .venv/bin/python tools/python/beatgrid/inspect_model.py [model7|model8]
Reads /Applications/rekordbox 7/.../Resources/models/detect_beat/<model>.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from protobuf_min import message, one, shape, text, varint

MODEL_ROOT = Path("/Applications/rekordbox 7/rekordbox.app/Contents/Resources/models")
NAME = sys.argv[1] if len(sys.argv) > 1 else "model7"
DIR = MODEL_ROOT / "detect_beat" / NAME


def parse_nodes(raw_nodes):
    nodes = []
    for raw in raw_nodes:
        n = message(raw)
        attrs = {}
        for entry in n.get(5, []):
            e = message(entry)
            key = text(one(e, 1, b""))
            av = message(one(e, 2, b""))
            if 6 in av:
                attrs[key] = {"type": av[6][0]}
            elif 7 in av:
                attrs[key] = {"shape": shape(av[7][0])}
            elif 3 in av:
                attrs[key] = {"i": av[3][0]}
            elif 2 in av:
                attrs[key] = {"s": text(av[2][0])}
            elif 8 in av:
                t = message(av[8][0])
                attrs[key] = {"tensor_shape": shape(one(t, 2, b"")) if 2 in t else None}
            elif 4 in av:
                attrs[key] = {"f": av[4][0]}
            elif 5 in av:
                attrs[key] = {"b": av[5][0]}
            elif 1 in av:
                attrs[key] = {"list": "..."}
            else:
                attrs[key] = {}
        nodes.append(
            {
                "name": text(one(n, 1, b"")),
                "op": text(one(n, 2, b"")),
                "input": [text(i) for i in n.get(3, [])],
                "attrs": attrs,
            }
        )
    return nodes


def parse_graph(buf: bytes):
    g = message(buf)
    nodes = parse_nodes(g.get(1, []))
    lib = message(one(g, 2, b"")) if 2 in g else {}
    funcs = {}
    for raw in lib.get(1, []):
        f = message(raw)
        fname = text(one(message(one(f, 1, b"")), 1, b""))
        funcs[fname] = parse_nodes(f.get(3, []))
    return nodes, funcs


sm = message(DIR.joinpath("saved_model.pb").read_bytes())
print("SavedModel fields:", sorted(sm))
metas = sm.get(2, [])
print("meta graphs:", len(metas))
mg = message(metas[0])
print("MetaGraph fields:", sorted(mg))

graph_nodes, funcs = parse_graph(one(mg, 2, b""))
print("graph nodes:", len(graph_nodes), "functions:", len(funcs))
print("\nTop-level graph ops:", Counter(n["op"] for n in graph_nodes).most_common())
print("\nTop-level graph nodes:")
for n in graph_nodes:
    if n["op"] in ("Const", "Placeholder", "StatefulPartitionedCall", "PartitionedCall", "Identity"):
        a = n["attrs"]
        desc = ""
        if "shape" in a:
            desc = f" shape={a['shape']}"
        if "tensor_shape" in a:
            desc = f" tensor={a['tensor_shape']}"
        if "s" in a:
            desc = f" s={a['s']!r}"
        if "t" in a:
            desc = f" type={a['t']}"
        print(f"  {n['op']:24} {n['name']} {n['input'][:2]}{desc}")

# signature defs
print("\nSignatureDefs:")
for raw in mg.get(5, []):
    e = message(raw)
    print(" ", text(one(e, 1, b"")), message(one(e, 2, b"")))
    print("   inputs:", one(e, 3, b"")[:200])
    print("   outputs:", one(e, 3, b"")[:200])

print("\nFunctions:")
for fname, nodes in funcs.items():
    ops = Counter(n["op"] for n in nodes)
    print(f"  {fname}: {len(nodes)} nodes, ops={dict(ops.most_common(12))}")

# Find the serving function and print its layers in order.
serving = next((f for f in funcs if "serving_default" in f), None)
if serving:
    print(f"\n=== {serving} nodes (first 200) ===")
    for n in funcs[serving][:200]:
        a = n["attrs"]
        extra = ""
        for key in ("T", "dtype", "value"):
            if key in a:
                extra += f" {key}={a[key]}"
        print(f"  {n['op']:24} {n['name']:60} <- {n['input']}{extra}")
