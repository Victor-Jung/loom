#!/usr/bin/env python
"""Re-time the best candidates of a results.json several times, interleaved.

Takes the `--top` fastest candidates of a run (by first-round board time),
plus every candidate whose loop tree matches a finalist of `--compare` runs
(e.g. the beam's rsfpu2 run of the same program), and times each `--reps`
times in fresh processes, round-robin across candidates so slow drift of the
board hits all of them alike. A finalist whose tree the run lacks (a sampled
run) is timed from its own run, in the same rounds, with origin "compare".
Writes remeasure.json next to results.json: per candidate the samples (the
first-round time first), median, min and max.

    python -u benchmarks/remeasure.py test/bench/mask_softmax/exh --top 250 --reps 5 \\
        --compare test/bench/mask_softmax/rsfpu2
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run import arg_shapes, run_candidate_in_process  # noqa: E402


def tree_of(function: str) -> str:
    """The loop-tree signature in a variant name (`__order_<sig>__x...`)."""
    m = re.search(r"__order_(.+?)__x", function)
    return m.group(1) if m else function


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="output directory holding results.json, e.g. test/bench/mask_softmax/exh")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--compare", nargs="*", default=[], help="runs whose finalists' trees are also re-timed")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=150)
    ap.add_argument("--fidelity", default="HiFi4")
    a = ap.parse_args()

    out = Path(a.run)
    data = json.loads((ROOT / out / "results.json").read_text())
    ok = [r for r in data["candidates"] if r.get("stage") == "ok"]
    ok.sort(key=lambda r: r["device_ms"])
    by_tree = {}
    for r in ok:
        by_tree.setdefault(tree_of(r["function"]), r)
    # Candidates are keyed by (run, index): indices repeat across runs.
    # A merged run (benchmarks/merge_runs.py) records the batch each row came from.
    src = lambda r: Path(r.get("run", out))
    key = lambda r: (str(src(r)), r["index"])
    chosen = {key(r): r for r in ok[:a.top]}
    compared, extra = {}, set()
    for c in a.compare:
        cdata = json.loads((ROOT / c / "results.json").read_text())
        sel = (cdata["summary"].get("selected") or {}).get("index")
        for r in cdata["candidates"]:
            t = tree_of(r["function"])
            entry = {"run": c, "index": r["index"], "selected": r["index"] == sel}
            if t in by_tree:
                k = key(by_tree[t])
                chosen.setdefault(k, by_tree[t])
            elif r.get("stage") == "ok":
                r = {**r, "run": c}
                k = key(r)
                chosen.setdefault(k, r)
                extra.add(k)
            else:
                continue
            compared.setdefault(k, []).append(entry)
    program = out.parts[-2]
    if program == "mamba_chunk_scan":
        from benchmarks.mamba_space import run_candidate
        timer = lambda r: run_candidate(Path(r["kdir"]), src(r) / "IRs" / "p03_bufferized.mlir",
                                        a.iters, a.timeout, a.fidelity)
    else:
        timer = lambda r: run_candidate_in_process(program, Path(r["kdir"]), arg_shapes(src(r), r["function"]),
                                                   a.iters, 3, a.timeout, a.fidelity, True, "trace", 20)
    cands = sorted(chosen.values(), key=lambda r: r["device_ms"])
    samples = {key(r): [r["device_ms"]] for r in cands}
    print(f"{program}: re-timing {len(cands)} candidates x {a.reps} "
          f"({len(compared) - len(extra)} matched and {len(extra)} added from {len(a.compare)} compared run(s))")
    t0 = time.perf_counter()
    for rep in range(a.reps):
        for r in cands:
            res = timer(r)
            samples[key(r)].append(res.get("device_ms", float("nan")))
        print(f"rep {rep + 1}/{a.reps} done at {time.perf_counter() - t0:.0f} s")
        rows = []
        for r in cands:
            k = key(r)
            s = [v for v in samples[k] if v == v]
            rows.append({"index": r["index"], "run": k[0], "origin": "compare" if k in extra else "run",
                         "function": r["function"], "tree": tree_of(r["function"]),
                         "T_total": r.get("T_total"), "estimate": r.get("estimate"),
                         "samples": samples[k], "median": statistics.median(s) if s else None,
                         "min": min(s) if s else None, "max": max(s) if s else None,
                         "compared": compared.get(k, [])})
        rows.sort(key=lambda x: x["median"] if x["median"] is not None else float("inf"))
        (ROOT / out / "remeasure.json").write_text(json.dumps(
            {"reps_done": rep + 1, "reps": a.reps, "top": a.top, "compare": a.compare, "candidates": rows},
            indent=2))
    print(f"wrote {out}/remeasure.json")


if __name__ == "__main__":
    main()
