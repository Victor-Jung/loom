#!/usr/bin/env python
"""Re-time the selected candidate of several runs of one program side by
side: each run's pick, `--reps` times, round-robin in fresh processes, so
slow drift of the board hits all of them alike. Writes compare_picks.json
into the first run's directory: per run the pick's function, the samples
(its selection time first), median, min and max.

    python -u benchmarks/compare_picks.py test/bench/swiglu/rsfpu2 test/bench/swiglu/rs_pm2 --reps 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run import arg_shapes, run_candidate_in_process  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories of one program, e.g. test/bench/swiglu/rsfpu2")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=150)
    ap.add_argument("--fidelity", default="HiFi4")
    a = ap.parse_args()

    runs = [Path(r) for r in a.runs]
    program = runs[0].parts[-2]
    picks = []
    for run in runs:
        data = json.loads((ROOT / run / "results.json").read_text())
        sel = data["summary"].get("selected") or {}
        row = next((r for r in data["candidates"] if r["function"] == sel.get("function")), None)
        if row is None:
            print(f"{run}: no selected candidate, skipped")
            continue
        picks.append({"run": str(run), "row": row, "samples": [sel["device_ms"]]})
    t0 = time.perf_counter()
    for rep in range(a.reps):
        for p in picks:
            r = p["row"]
            res = run_candidate_in_process(program, Path(r["kdir"]), arg_shapes(Path(p["run"]), r["function"]),
                                           a.iters, 3, a.timeout, a.fidelity, True, "trace", 20)
            p["samples"].append(res.get("device_ms", float("nan")))
        print(f"rep {rep + 1}/{a.reps} done at {time.perf_counter() - t0:.0f} s")
    out = []
    for p in picks:
        s = [v for v in p["samples"] if v == v]
        out.append({"run": p["run"], "function": p["row"]["function"], "estimate": p["row"].get("estimate"),
                    "T_total": p["row"].get("T_total"), "samples": p["samples"],
                    "median": statistics.median(s) if s else None,
                    "min": min(s) if s else None, "max": max(s) if s else None})
    (ROOT / runs[0] / "compare_picks.json").write_text(json.dumps({"reps": a.reps, "picks": out}, indent=2))
    for p in out:
        print(f"{p['run']}: median {p['median']:.4f} ms")
    print(f"wrote {runs[0]}/compare_picks.json")


if __name__ == "__main__":
    main()
