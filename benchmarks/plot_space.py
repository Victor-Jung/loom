#!/usr/bin/env python
"""Figure of a search-space study: the board-time distribution of every
measured tree, the tuner estimate against board time, and the re-timed best
trees, with the finalists and the selection of a beam run marked.

    python benchmarks/plot_space.py test/bench/gqa_decode/exh --beam test/bench/gqa_decode/rsfpu2 \\
        --title "gqa_decode (B8 G8 S4096 D128)" -o ~/KnowledgeBase/figures/gqa_decode_exh.png

Panels: (a) trees sorted by first-round board time; (b) histogram of board
time over the best; (c) estimate vs board time, with Spearman's rho;
(d) re-timed trees (remeasure.json) by median, min-max bars.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.space_report import spearman  # noqa: E402

FINALIST = dict(marker="o", mfc="none", mec="tab:blue", ms=8, ls="none")
SELECTED = dict(marker="*", color="tab:red", ms=14, ls="none")


def tree_of(function: str) -> str:
    m = re.search(r"__order_(.+?)__x", function)
    return m.group(1) if m else function


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--beam", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    run = ROOT / a.run
    ok = [r for r in json.loads((run / "results.json").read_text())["candidates"] if r.get("stage") == "ok"]
    ok.sort(key=lambda r: r["device_ms"])
    trees = [tree_of(r["function"]) for r in ok]
    ms = np.array([r["device_ms"] for r in ok])
    est = np.array([r["estimate"] if r.get("estimate") is not None else np.nan for r in ok])
    best = ms[0]
    fin, sel = set(), None
    if a.beam:
        b = json.loads((ROOT / a.beam / "results.json").read_text())
        fin = {tree_of(r["function"]) for r in b["candidates"]}
        s = (b["summary"].get("selected") or {}).get("function")
        sel = tree_of(s) if s else None
    is_fin = np.array([t in fin for t in trees])
    is_sel = np.array([t == sel for t in trees])

    rem = run / "remeasure.json"
    med = json.loads(rem.read_text())["candidates"] if rem.exists() else []
    med = [c for c in med if c.get("median") is not None]

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax = ax.ravel()

    x = np.arange(1, len(ms) + 1)
    ax[0].plot(x, ms, ".", color="0.5", ms=4, label="tree")
    ax[0].plot(x[is_fin], ms[is_fin], **FINALIST, label="beam finalists")
    ax[0].plot(x[is_sel], ms[is_sel], **SELECTED, label="beam selected")
    ax[0].axhline(best * 1.05, color="tab:green", ls=":", lw=1, label="best +5%")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("rank by board time")
    ax[0].set_ylabel("board time (ms, log)")
    ax[0].set_title(f"(a) {len(ms)} trees, sorted (one measurement each)")
    ax[0].legend(fontsize=8)

    ratio = ms / best
    bins = np.logspace(0, np.log10(ratio.max() * 1.01), 30)
    ax[1].hist(ratio, bins=bins, color="0.6", edgecolor="white")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("board time / best (log)")
    ax[1].set_ylabel("trees")
    for q, c in [(50, "tab:orange"), (90, "tab:purple")]:
        v = np.percentile(ratio, q)
        ax[1].axvline(v, color=c, ls="--", lw=1, label=f"p{q} = {v:.2f}x")
    ax[1].set_title(f"(b) distribution (best = {best:.3f} ms)")
    ax[1].legend(fontsize=8)

    has = ~np.isnan(est)
    rho = spearman(list(est[has]), list(ms[has]))
    ax[2].scatter(est[has], ms[has], s=10, color="0.5", label="tree")
    ax[2].plot(est[is_fin & has], ms[is_fin & has], **FINALIST, label="beam finalists")
    ax[2].plot(est[is_sel & has], ms[is_sel & has], **SELECTED, label="beam selected")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("tuner estimate (log)")
    ax[2].set_ylabel("board time (ms, log)")
    ax[2].set_title(f"(c) estimate vs board (Spearman {rho:.2f})")
    ax[2].legend(fontsize=8)

    if med:
        med.sort(key=lambda c: c["median"])
        m0 = med[0]["median"]
        y = np.array([c["median"] for c in med])
        lo = y - np.array([c["min"] for c in med])
        hi = np.array([c["max"] for c in med]) - y
        xr = np.arange(1, len(med) + 1)
        f2 = np.array([c["tree"] in fin for c in med])
        s2 = np.array([c["tree"] == sel for c in med])
        ax[3].axhspan(m0, m0 * 1.01, color="tab:green", alpha=0.12, label="best median +1%")
        ax[3].errorbar(xr, y, yerr=[lo, hi], fmt=".", color="0.4", ecolor="0.7", capsize=2,
                       label="median, min-max")
        ax[3].plot(xr[f2], y[f2], **FINALIST, label="beam finalists")
        ax[3].plot(xr[s2], y[s2], **SELECTED, label="beam selected")
        reps = len(med[0]["samples"])
        ax[3].set_xlabel("rank by re-timed median")
        ax[3].set_ylabel("board time (ms)")
        ax[3].set_title(f"(d) {len(med)} fastest trees re-timed ({reps} measurements each)")
        ax[3].legend(fontsize=8)
    else:
        ax[3].axis("off")

    fig.suptitle(a.title or a.run)
    fig.tight_layout()
    out = Path(a.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    print("ratio p0,10,25,50,75,90,100:", np.round(np.percentile(ratio, [0, 10, 25, 50, 75, 90, 100]), 2))
    print("ms    p0,10,25,50,75,90,100:", np.round(np.percentile(ms, [0, 10, 25, 50, 75, 90, 100]), 3))
    if med:
        print("re-timed:", [(c["tree"], round(c["median"] / m0, 4)) for c in med[:15]])


if __name__ == "__main__":
    main()
