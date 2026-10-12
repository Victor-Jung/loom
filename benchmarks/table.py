"""Benchmark table: Loom against ttnn on every program with a baseline.

Loom: the halving selection (or the best passing candidate) of a run.py
result tag. ttnn: the newest `tmp_output/sota/<program>_*_g<grid>.json` of the same
argument shapes; the fastest passing config (PCC >= 0.99) that launches a
single device operation per call, and the fastest that launches several
(op counts from ttnn graph capture, recorded by sota.py).

    python benchmarks/table.py --tag rsloop [--grid 12x10]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def arg_shapes(result_dir: Path, func: str) -> list[list[int]]:
    """Argument shapes of `func` in the run's final IR."""
    text = (result_dir / "IRs" / "p03_bufferized.mlir").read_text()
    head = text[text.index(f"func.func @{func}("):]
    sig = head[:head.index(")")]
    return [[int(x) for x in m.split("x")] for m in re.findall(r"memref<([0-9x]+)x[a-z0-9]+>", sig)]


def loom_runs(tag: str) -> list[tuple[str, list[list[int]], float]]:
    """(program, argument shapes, ms) for every result dir of `tag`."""
    out = []
    for f in sorted(glob.glob(str(ROOT / "test/bench" / "*" / f"{tag}*" / "results.json"))):
        data = json.load(open(f))
        sel = data["summary"].get("selected")
        cands = [c for c in data["candidates"] if c.get("stage") == "ok"]
        if sel:
            ms, func = sel["device_ms"], sel["function"]
        elif cands:
            best = min(cands, key=lambda c: c["device_ms"])
            ms, func = best["device_ms"], best["function"]
        else:
            continue
        d = Path(f).parent
        out.append((d.parent.name, arg_shapes(d, func), ms))
    return out


def ttnn_best(program: str, shapes: list[list[int]], grid: str) -> dict[str, tuple[float, int]]:
    """Fastest passing config per kind in the newest sota file of these shapes:
    "fused" when one call launches a single device operation (counted by graph
    capture, `device_ops`), "multi" otherwise; values are (ms, op count)."""
    files = sorted(glob.glob(str(ROOT / "tmp_output/sota" / f"{program}_*_g{grid}.json")),
                   key=os.path.getmtime)
    for f in reversed(files):
        data = json.load(open(f))
        if data.get("shapes") != shapes:
            continue
        best: dict[str, tuple[float, int]] = {}
        for r in data["rows"]:
            base = r.get("baseline")
            if not base or "device_ms" not in r or r.get("pcc", 0) < 0.99 or "reference" in base:
                continue
            ops = r.get("device_ops")
            if ops is None:
                continue  # measured before op counting: cannot tell fused from not
            kind = "fused" if len(ops) == 1 else "multi"
            if kind not in best or r["device_ms"] < best[kind][0]:
                best[kind] = (r["device_ms"], len(ops))
        return best
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="rsloop")
    ap.add_argument("--grid", default="12x10")
    a = ap.parse_args()
    print("| program | shape | Loom | ttnn single op | ttnn multi-op (ops) | Loom / single | Loom / multi |")
    print("|---|---|---|---|---|---|---|")
    ratios_f, ratios_c = [], []
    for p, shapes, lm in loom_runs(a.tag):
        best = ttnn_best(p, shapes, a.grid)
        if not best:
            continue
        f = best["fused"][0] if "fused" in best else None
        c = best["multi"][0] if "multi" in best else None
        cell = lambda v: f"{v:.3f}" if v is not None else "--"
        rf = lm / f if f else None
        rc = lm / c if c else None
        if rf: ratios_f.append(rf)
        if rc: ratios_c.append(rc)
        shape = " ".join("x".join(map(str, s)) for s in shapes[:-1])
        cm = f"{cell(c)} ({best['multi'][1]})" if c else "--"
        print(f"| {p} | {shape} | {cell(lm)} | {cell(f)} | {cm} | {cell(rf)} | {cell(rc)} |")
    g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
    print(f"\nLoom / ttnn single op: geomean {g(ratios_f):.3f} over {len(ratios_f)}, "
          f"Loom faster on {sum(r < 1 for r in ratios_f)}; "
          f"Loom / ttnn multi-op: geomean {g(ratios_c):.3f} over {len(ratios_c)}, faster on {sum(r < 1 for r in ratios_c)}")


if __name__ == "__main__":
    main()
