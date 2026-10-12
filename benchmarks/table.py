"""Benchmark table: Loom against ttnn on every program with a baseline.

Loom: the halving selection (or the best passing candidate) of a run.py
result tag. ttnn: the newest `tmp_output/sota/<program>_*_g<grid>.json` that
sota.py wrote for the program; per kind (fused or single op, composite) the
fastest config with PCC >= 0.99.

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


def ttnn_best(program: str, shapes: list[list[int]], grid: str) -> dict[str, float]:
    """Fastest passing config per kind in the newest sota file of these shapes."""
    files = sorted(glob.glob(str(ROOT / "tmp_output/sota" / f"{program}_*_g{grid}.json")),
                   key=os.path.getmtime)
    for f in reversed(files):
        data = json.load(open(f))
        if data.get("shapes") != shapes:
            continue
        best: dict[str, float] = {}
        for r in data["rows"]:
            base = r.get("baseline")
            if not base or "device_ms" not in r or r.get("pcc", 0) < 0.99 or "reference" in base:
                continue
            kind = "composite" if "composite" in base else "fused"
            best[kind] = min(best.get(kind, math.inf), r["device_ms"])
        return best
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="rsloop")
    ap.add_argument("--grid", default="12x10")
    a = ap.parse_args()
    print("| program | shape | Loom | ttnn fused / op | ttnn composite | Loom / fused | Loom / composite |")
    print("|---|---|---|---|---|---|---|")
    ratios_f, ratios_c = [], []
    for p, shapes, lm in loom_runs(a.tag):
        best = ttnn_best(p, shapes, a.grid)
        if not best:
            continue
        f, c = best.get("fused"), best.get("composite")
        cell = lambda v: f"{v:.3f}" if v is not None else "--"
        rf = lm / f if f else None
        rc = lm / c if c else None
        if rf: ratios_f.append(rf)
        if rc: ratios_c.append(rc)
        shape = " ".join("x".join(map(str, s)) for s in shapes[:-1])
        print(f"| {p} | {shape} | {cell(lm)} | {cell(f)} | {cell(c)} | {cell(rf)} | {cell(rc)} |")
    g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
    print(f"\nLoom / ttnn fused or op: geomean {g(ratios_f):.3f} over {len(ratios_f)}, "
          f"Loom faster on {sum(r < 1 for r in ratios_f)}; "
          f"Loom / composite: geomean {g(ratios_c):.3f} over {len(ratios_c)}, faster on {sum(r < 1 for r in ratios_c)}")


if __name__ == "__main__":
    main()
