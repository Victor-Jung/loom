#!/usr/bin/env python
"""CSV tables of the search-space studies: one row per study (summary) and
one row per measured tree (trees), from each study's results.json and
remeasure.json and the beam run it is compared with.

    python benchmarks/space_tables.py --out ~/KnowledgeBase/data
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.space_report import spearman, tree_of  # noqa: E402

# (program, shape, study run, kind, space size, beam run)
STUDIES = [
    ("layernorm_residual", "M8192 N1024", "exh", "exhaustive", "229", "rsfpu2"),
    ("gqa_decode", "B8 G8 S4096 D128", "exh", "exhaustive", "342", "rsfpu2"),
    ("mask_softmax", "B2 H32 S1024 N1024", "exh", "exhaustive", "5170", "rsfpu2"),
    ("rotary", "H32 T4096 D128", "rw500", "random walk (500 x 8 moves)", "12509", "rsfpu2"),
    ("bias_add_full", "B16 M2048 N2048", "rw500", "random walk (500 x 8 moves)", "16624", "rsfpu2"),
    ("mamba_chunk_scan", "L1920 N32 H128 G4 D128 C192", "rw1000", "random walk (1000 x 8 moves)",
     ">=10523", "beam_rs"),
]


def rank_of(values: list[float], v: float) -> tuple[int, int]:
    """Rank of `v` among `values`, lower first: (1 + number strictly lower, ties)."""
    return 1 + sum(x < v for x in values), sum(x == v for x in values)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    summary, trees = [], []
    for program, shape, run, kind, space, beam in STUDIES:
        base = ROOT / "test/bench" / program
        rows = json.loads((base / run / "results.json").read_text())["candidates"]
        ok = sorted((r for r in rows if r.get("stage") == "ok"), key=lambda r: r["device_ms"])
        rem = json.loads((base / run / "remeasure.json").read_text())["candidates"]
        med = {(c["tree"], c.get("origin") == "compare"): c for c in rem if c.get("median") is not None}
        b = json.loads((base / beam / "results.json").read_text())
        sel_f = (b["summary"].get("selected") or {}).get("function")
        fin = {tree_of(r["function"]) for r in b["candidates"]}
        sel = tree_of(sel_f) if sel_f else None

        first = ok[0]["device_ms"]
        est = [r["estimate"] for r in ok]
        tt = [r["T_total"] for r in ok]
        best = min((c for (t, cmp), c in med.items() if not cmp), key=lambda c: c["median"])
        best_row = next(r for r in ok if tree_of(r["function"]) == best["tree"])
        e_rank, e_ties = rank_of(est, best_row["estimate"])
        t_rank, t_ties = rank_of(tt, best_row["T_total"])
        fin_med = {t: c["median"] for (t, _), c in med.items() if t in fin}
        space_rank = {tree_of(r["function"]): i + 1 for i, r in enumerate(ok)}

        def top1(key: str) -> str:
            lo = min(r[key] for r in ok)
            d = [r["device_ms"] / first for r in ok if r[key] == lo]
            return f"{min(d):.3f}-{max(d):.3f} ({len(d)} tied)"

        ratios = [r["device_ms"] / first for r in ok]
        summary.append({
            "program": program, "shape": shape, "study": kind, "space_trees": space,
            "measured": len(rows), "passed": len(ok),
            "best_tree": best["tree"], "best_median_ms": round(best["median"], 4),
            "within_3pct": sum(x <= 1.03 for x in ratios), "within_5pct": sum(x <= 1.05 for x in ratios),
            "median_over_best": round(statistics.median(ratios), 3),
            "p90_over_best": round(sorted(ratios)[int(0.9 * (len(ratios) - 1))], 3),
            "worst_over_best": round(ratios[-1], 3),
            "spearman_estimate": round(spearman(est, [r["device_ms"] for r in ok]), 3),
            "spearman_T_total": round(spearman(tt, [r["device_ms"] for r in ok]), 3),
            "best_tree_estimate_rank": f"{e_rank} ({e_ties} tied)",
            "best_tree_T_total_rank": f"{t_rank} ({t_ties} tied)",
            "estimate_top1_over_best": top1("estimate"), "T_total_top1_over_best": top1("T_total"),
            "beam_selected": sel or "",
            "beam_selected_over_best": round(fin_med[sel] / best["median"], 3) if sel in fin_med else "",
            "beam_best_finalist_over_best": round(min(fin_med.values()) / best["median"], 3) if fin_med else "",
            "beam_selected_rank_first_round": space_rank.get(sel, "not sampled"),
            "remeasured_trees": len(med), "remeasure_reps": len(rem[0]["samples"]) if rem else 0,
        })

        def tree_row(r: dict, origin: str) -> dict:
            t = tree_of(r["function"])
            c = med.get((t, origin == "beam finalist (not in space run)"))
            return {
                "program": program, "study": run, "origin": origin, "tree": t,
                "first_round_ms": round(r["device_ms"], 4),
                "first_round_over_best": round(r["device_ms"] / first, 4),
                "first_round_rank": space_rank.get(t, "") if origin == "space" else "",
                "median_ms": round(c["median"], 4) if c else "",
                "min_ms": round(c["min"], 4) if c else "", "max_ms": round(c["max"], 4) if c else "",
                "measurements": len(c["samples"]) if c else 1,
                "estimate": r.get("estimate"), "estimate_rank": rank_of(est, r["estimate"])[0],
                "T_total": r.get("T_total"), "T_total_rank": rank_of(tt, r["T_total"])[0],
                "beam_finalist": t in fin, "beam_selected": t == sel,
                "pcc": round(r["pcc"], 5) if r.get("pcc") is not None else "", "function": r["function"],
            }

        trees += [tree_row(r, "space") for r in ok]
        seen = {tree_of(r["function"]) for r in ok}
        trees += [tree_row(r, "beam finalist (not in space run)")
                  for r in b["candidates"] if r.get("stage") == "ok" and tree_of(r["function"]) not in seen]

    for name, data in (("space_study_summary.csv", summary), ("space_study_trees.csv", trees)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {out / name} ({len(data)} rows)")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
