#!/usr/bin/env python
"""Summarize a search-space study: a run that measured many trees (exhaustive
or sampled), its re-timing (remeasure.json), and the beam run it is compared
with.

For the sampled or exhaustive run: how many trees passed, the best time, how
many trees lie within 3% and 5% of the best, the rank correlation of the
tuner's estimate and of the solver's T_total with board time over the whole
set. For the beam run: where its selected tree and its finalists rank in the
set (by first-round time), and the gap to the best by re-timed medians.

    python benchmarks/space_report.py test/bench/mask_softmax/exh --beam test/bench/mask_softmax/rsfpu2
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tree_of(function: str) -> str:
    m = re.search(r"__order_(.+?)__x", function)
    return m.group(1) if m else function


def avg_ranks(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = avg_ranks(a), avg_ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


def report(run: Path, beam: Path | None) -> dict:
    data = json.loads((ROOT / run / "results.json").read_text())
    rows = data["candidates"]
    ok = [r for r in rows if r.get("stage") == "ok"]
    ok.sort(key=lambda r: r["device_ms"])
    best = ok[0]["device_ms"]
    res = {"run": str(run), "candidates": len(rows), "ok": len(ok),
           "failed": {s: sum(1 for r in rows if r.get("stage") == s) for s in ("lower", "run", "pcc")},
           "best_first_round_ms": best, "best_tree": tree_of(ok[0]["function"]),
           "within_3pct": sum(r["device_ms"] <= best * 1.03 for r in ok),
           "within_5pct": sum(r["device_ms"] <= best * 1.05 for r in ok),
           "median_over_best": ok[len(ok) // 2]["device_ms"] / best,
           "worst_over_best": ok[-1]["device_ms"] / best}
    dev = [r["device_ms"] for r in ok]
    for key in ("estimate", "T_total"):
        pairs = [(r[key], r["device_ms"]) for r in ok if r.get(key) is not None]
        res[f"spearman_{key}"] = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if pairs:
            top = min(pairs)[0]
            tied = [d for v, d in pairs if v == top]
            res[f"{key}_top1_over_best"] = [min(tied) / best, max(tied) / best, len(tied)]
    rank = {tree_of(r["function"]): i + 1 for i, r in enumerate(ok)}
    by_tree = {tree_of(r["function"]): r for r in ok}
    rem = ROOT / run / "remeasure.json"
    med = {}
    if rem.exists():
        rdata = json.loads(rem.read_text())
        med = {c["tree"]: c for c in rdata["candidates"] if c.get("median") is not None}
        best_med = min(c["median"] for c in med.values())
        res["remeasure"] = {"reps_done": rdata["reps_done"], "timed": len(med), "best_median_ms": best_med,
                            "best_median_tree": min(med.values(), key=lambda c: c["median"])["tree"]}
    if beam:
        bdata = json.loads((ROOT / beam / "results.json").read_text())
        sel = (bdata["summary"].get("selected") or {}).get("index")
        fin = []
        for r in bdata["candidates"]:
            t = tree_of(r["function"])
            item = {"index": r["index"], "selected": r["index"] == sel, "tree": t,
                    "beam_run_ms": r.get("device_ms"), "rank_in_space": rank.get(t),
                    "space_ms": by_tree[t]["device_ms"] if t in by_tree else None}
            if t in med:
                item["median_ms"] = med[t]["median"]
                item["median_over_best_median"] = med[t]["median"] / res["remeasure"]["best_median_ms"]
            fin.append(item)
        res["beam"] = {"run": str(beam), "finalists": fin,
                       "selected": next((f for f in fin if f["selected"]), None)}
        if "remeasure" in res:
            res["beam"]["best_finalist_over_best_median"] = min(
                (f["median_over_best_median"] for f in fin if "median_over_best_median" in f), default=None)
        if not res.get("remeasure"):
            ranks = [f["rank_in_space"] for f in fin if f["rank_in_space"]]
            res["beam"]["best_finalist_rank"] = min(ranks) if ranks else None
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--beam", default=None)
    a = ap.parse_args()
    print(json.dumps(report(Path(a.run), Path(a.beam) if a.beam else None), indent=2))


if __name__ == "__main__":
    main()
