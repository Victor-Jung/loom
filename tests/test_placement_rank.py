"""Placement features from candidate names and the placement-aware ranking."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("placement", ROOT / "loom/solver/placement.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

OUTER = ("_outer_scale__order_m1Sn11Sn6_m_n_h9o3__x11x1_y6__d0i1_d1i0_d2i2__f012__"
         "dim_x_level0_bc11_dim_y_level0_bc6_n_n_n__is_double_buffer1__tile_m2048__tile_n32")
MASK = "_mask_softmax__order_b2Sh3Sm2Sm8_m_b_h__x2x3x2_y8__d0i2_d1i3_d2i1_d3i0__f0123__n_dim_y_level0_bc8_n"


def test_features_read_dims_and_multicasts() -> None:
    f = P.features(OUTER)
    assert (f["cores_x"], f["cores_y"], f["axes_x"], f["axes_y"]) == (11, 6, 1, 1)
    assert (f["mcast_x"], f["fanout_x"], f["mcast_y"], f["fanout_y"], f["copies"]) == (1, 11, 1, 6, 5)
    g = P.features(MASK)
    assert (g["cores_x"], g["axes_x"], g["cores_y"]) == (12, 3, 8)
    assert (g["mcast_x"], g["mcast_y"], g["fanout_y"], g["copies"]) == (0, 1, 8, 3)


def test_rerank_orders_by_predicted_time_and_keeps_ties_stable() -> None:
    ranked = [{"n": MASK, "t": 100}, {"n": OUTER, "t": 100}, {"n": MASK, "t": 100}]
    w = {"T_total": 1.0, "mcast_x": -5.0}
    old = dict(P.WEIGHTS)
    try:
        P.WEIGHTS.clear(); P.WEIGHTS.update(w)
        out = P.rerank(ranked, lambda r: r["n"], lambda r: r["t"])
    finally:
        P.WEIGHTS.clear(); P.WEIGHTS.update(old)
    assert out[0] is ranked[1] and out[1] is ranked[0] and out[2] is ranked[2]


def test_ranking_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("LOOM_RANK", raising=False)
    assert not P.enabled()
    monkeypatch.setenv("LOOM_RANK", "placement")
    assert P.enabled()
