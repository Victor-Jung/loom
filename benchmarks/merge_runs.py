#!/usr/bin/env python
"""Merge the results.json of runs compiled in batches (`part=K/N` in the
tuning options) into one results.json, so the reports and the re-timing treat
them as one candidate set. Rows keep their batch directory in `run` and get a
global `index`; the batch's own index stays in `part_index`.

    python benchmarks/merge_runs.py test/bench/mask_softmax/exh test/bench/mask_softmax/exh_p*
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    out, parts = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    rows, summaries = [], []
    for p in parts:
        data = json.loads((ROOT / p / "results.json").read_text())
        summaries.append({"run": str(p), **data["summary"]})
        for r in data["candidates"]:
            rows.append({**r, "run": str(p), "part_index": r.get("index")})
    for i, r in enumerate(rows, start=1):
        r["index"] = i
    (ROOT / out).mkdir(parents=True, exist_ok=True)
    (ROOT / out / "results.json").write_text(json.dumps(
        {"summary": {"merged_from": [str(p) for p in parts], "parts": summaries}, "candidates": rows}, indent=2))
    print(f"{out}: {len(rows)} candidates from {len(parts)} run(s)")


if __name__ == "__main__":
    main()
