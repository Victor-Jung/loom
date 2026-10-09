"""Placement-aware ranking of solved candidates.

The tuner decides which loops run on how many cores; the exploration then
places those axes on the physical mesh dims and picks a multicast per copy.
The candidate name records that placement:

    ..._x11x1_y6__d0i1_d1i0_d2i2__f012__dim_x_level0_bc11_dim_y_level0_bc6_n_n...
        cores per dim     axis -> dim      one label per copy: multicast or n

`features` reads it; `predicted_ms` combines the solver's T_total with the
placement features under weights fitted on trace-timed candidates
(tmp_output/fit/fit_placement.py). `rerank` orders solved results by it.
Enabled by LOOM_RANK=placement.
"""
from __future__ import annotations

import os
import re

FEATURES = ["cores_x", "cores_y", "axes_x", "axes_y", "mcast_x", "mcast_y", "fanout_x", "fanout_y", "copies"]

# ms per unit, fitted with per-program intercepts on 264 trace-timed
# candidates of 22 programs (2026-10-09). Leave-one-program-out with the
# feature subset also chosen inside each fold, the reranking picks 1.079x the
# fastest measured candidate against 1.076x for the solver's own order: no
# validated gain, hence opt-in.
WEIGHTS: dict[str, float] = {"fanout_x": 0.007631865, "fanout_y": 0.006105307,
                             "copies": -0.028430839}

_LABEL = re.compile(r"dim_([a-z])_level\d+_bc(\d+)|(?<![a-z])n(?=_|$)")


def features(name: str) -> dict[str, float]:
    """Placement features of a candidate (variant or function) name."""
    f = {k: 0.0 for k in FEATURES}
    m = re.search(r"__x([\dx]+)_y([\dy]+)__", name)
    if m:
        for dim, spec in (("x", m.group(1)), ("y", m.group(2))):
            parts = [int(p) for p in spec.split(dim) if p]
            prod = 1
            for p in parts:
                prod *= p
            f[f"cores_{dim}"] = prod
            f[f"axes_{dim}"] = sum(1 for p in parts if p > 1)
    seg = re.search(r"__f\d+__(.*?)(?:__is_double_buffer|__tile_|$)", name)
    if seg:
        for dim, bc in _LABEL.findall(seg.group(1)):
            f["copies"] += 1
            if dim in ("x", "y"):
                f[f"mcast_{dim}"] += 1
                f[f"fanout_{dim}"] += int(bc)
    return f


def predicted_ms(name: str, t_total: float, weights: dict[str, float] | None = None) -> float:
    w = WEIGHTS if weights is None else weights
    f = features(name)
    return w.get("T_total", 0.0) * t_total + sum(w.get(k, 0.0) * v for k, v in f.items())


def enabled() -> bool:
    return os.environ.get("LOOM_RANK", "") == "placement"


def rerank(ranked: list[dict], name_of, t_of) -> list[dict]:
    """`ranked` re-sorted by predicted time; ties keep the solver's order."""
    keyed = [(predicted_ms(name_of(r), t_of(r)), i, r) for i, r in enumerate(ranked)]
    keyed.sort(key=lambda k: (k[0], k[1]))
    return [r for _, _, r in keyed]
