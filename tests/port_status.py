"""Checklist of the writer reconstruction: what is ported, what is validated, what is not.

Each entry is one function of the reconstruction chain, with

    addr      binary address in the reference dylib (documentation only)
    symbol    demangled name
    py        "module:function" of the Python port, or None
    harness   path of the harness that calls the binary function, or None
    evidence  "validated <n>/<m>" when a harness run agreed, else why not

Exit code is 0 only when every in-scope entry is validated against the binary.
Anything else prints as an unchecked box, so this file doubles as the to-do list.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# status: validated -> ticked; anything else -> open
ENTRIES = [
    # ---- detector / analysis -------------------------------------------------
    dict(symbol="Analyzer::setSample(float*, float*, int, bool&)", addr="0x100f32bb0",
         py=None, harness="tests/hp17.cpp",
         evidence="not ported; harness only reaches the model-less path (§79)"),
    dict(symbol="Analyzer::analyze(bool&)", addr="0x100f32f60",
         py=None, harness=None,
         evidence="not ported; guards + argument setup only (§43, §82)"),
    dict(symbol="SimpleEnvelopeMaker::analyze(float, vector<BeatInfo>&)", addr="0x100f31f50",
         py=None, harness=None,
         evidence="not ported; identified as the peak-list producer (§89)"),
    dict(symbol="BeatSensor::put(float const*, int, void*)", addr="0x100f49f7c",
         py=None, harness="tests/fn_sensor4.cpp",
         evidence="not ported; runs in isolation with create(sr,0,n,8,6) (§81)"),
    dict(symbol="BeatSensor::selectBand()", addr="0x100f4a620",
         py=None, harness=None, evidence="not ported (§89)"),
    dict(symbol="PeakBuffer::calcBeat(float)", addr="0x100f5c190",
         py=None, harness=None, evidence="not ported; loop traced (§90)"),
    dict(symbol="PeakBuffer::calcBeat_s(...)", addr="0x100f5c27c",
         py=None, harness=None,
         evidence="not ported; both band tables recovered, traversal traced (§90, §91)"),
    # ---- beat list analysis --------------------------------------------------
    dict(symbol="BeatInfoAnalyzer::analyze(...)", addr="0x100f5d1c4",
         py="beat_info:tempo_histogram",
         harness=None,
         evidence="partly ported (tempo histograms §44-46); not validated, segmentation unverified (§51)"),
    dict(symbol="BeatInfoAnalyzer::findNearPos(...)", addr="0x100f5d00c",
         py="beat_info:find_near_pos", harness="tests/fn_findnear.cpp",
         evidence="validated 200/200 against the binary"),
    dict(symbol="BeatInfoAnalyzer::segAdjust(...)", addr="0x100f6a008",
         py="beat_info:seg_adjust", harness="tests/fn_seg2.cpp",
         evidence="transliterated whole (§76); known-wrong units, passes positions through (§77)"),
    dict(symbol="BeatInfoAnalyzer::gapCorrect(vector&, int)", addr="0x100f6a5fc",
         py="gap_correct_ref:gap_correct", harness="tests/fn_gap.cpp",
         evidence="partly transliterated; aborts on jittered input (§77)"),
    dict(symbol="BeatInfoAnalyzer::judgeHendouDynamicPeak()", addr="0x100f6aa8c",
         py=None, harness=None, evidence="not ported"),
    # ---- writer (UnitBeatAdjust) --------------------------------------------
    dict(symbol="UnitBeatAdjust::create(int, int)", addr="0x100f02da0",
         py=None, harness="tests/ua_analyze2.cpp",
         evidence="not ported; harness works and reports the state fields"),
    dict(symbol="UnitBeatAdjust::putSample(short*, int)", addr="0x100f0303c",
         py="writer_probe:envelopes", harness=None,
         evidence="decimation + 5 biquads ported (§12); not validated numerically"),
    dict(symbol="UnitBeatAdjust::manageWave(...)", addr="0x100f03a1c",
         py="writer_probe:manage_wave", harness=None, evidence="ported, not validated"),
    dict(symbol="UnitBeatAdjust::bpmAdjust(float&, vector&, int)", addr="0x100f032dc",
         py=None, harness="tests/ua_analyze2.cpp",
         evidence="not ported; returns the input bpm unchanged, grids are integer-step (§88)"),
    dict(symbol="UnitBeatAdjust::beatPositionAdjust(vector&)", addr="0x100f03ab8",
         py=None, harness="tests/ua_analyze2.cpp", evidence="not ported"),
    dict(symbol="UnitBeatAdjust::createBeatData(double, int)", addr="0x100f04cbc",
         py="writer_probe:create_beat_data", harness=None, evidence="ported, not validated"),
    # ---- ANLZ encoding (proven) ---------------------------------------------
    dict(symbol="ANLZ timestamp encoding (us = pos*1e6//unit)", addr="n/a",
         py="writer:encode_timestamps", harness="tests/verify_writer_facts.py",
         evidence="validated 2907/2907 reference timestamps"),
    dict(symbol="PQTZ ms + PQT2 frac split", addr="n/a",
         py="writer:split_microseconds", harness="tests/verify_writer_facts.py",
         evidence="validated 2907/2907 (round-trip)"),
]


def check(entry) -> tuple[bool, str]:
    """A tick requires: a port exists, a harness exists, and the evidence is a validation."""
    reasons = []
    if entry["py"]:
        mod, _, fn = entry["py"].partition(":")
        try:
            m = importlib.import_module(mod)
            if not hasattr(m, fn):
                reasons.append(f"{entry['py']} missing")
        except Exception as exc:                      # noqa: BLE001 - report, don't raise
            reasons.append(f"import {mod} failed: {exc}")
    else:
        reasons.append("no port")
    if entry["harness"]:
        if not (ROOT / entry["harness"]).exists():
            reasons.append(f"{entry['harness']} missing")
    else:
        reasons.append("no harness")
    ev = entry["evidence"]
    if not ev.startswith("validated"):
        reasons.append(ev)
    return (not reasons), "; ".join(reasons)


def main() -> None:
    ok = 0
    open_items = []
    print(f"{'':3} {'function':58} {'port':28} evidence")
    print("-" * 118)
    for e in ENTRIES:
        tick, why = check(e)
        box = "[x]" if tick else "[ ]"
        port = f"{e['py'] or '-':28}"
        print(f"{box} {e['symbol'][:58]:58} {port} {('OK' if tick else why)[:60]}")
        if tick:
            ok += 1
        else:
            open_items.append(e["symbol"])
    print("-" * 118)
    print(f"{ok}/{len(ENTRIES)} validated")
    if open_items:
        print("\nopen:")
        for s in open_items:
            print(f"  [ ] {s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
