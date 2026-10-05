"""Divisibility constraints use the loop extent, not the tile's upper bound.

The mamba chunk loop is tiled with `block_size=1`, so its tile symbol
`tile_c` has upper bound 1 while the loop runs over 10 chunks. The spatial
divisibility constraint used to divide that upper bound, producing
`(1 / tile_c) mod cores == 0`, which rejected every mapping that splits the
chunk loop (1736 of 1775 mamba variants). The extent now comes from the
trip-count expression `ceildivui(extent, tile)`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P00 = ROOT / "tests" / "fixtures" / "etg_mamba_p00.mlir"
HW_SPEC = ROOT / "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x12y10.mlir"

loom_pipeline = pytest.importorskip("loom_pipeline")


@pytest.fixture(scope="module")
def variants() -> list:
    if not HW_SPEC.exists():
        pytest.skip(f"hardware spec missing: {HW_SPEC}")
    _, etg_json = loom_pipeline.run_exploration(
        input_mlir=P00.read_text(), hw_spec_file=str(HW_SPEC), produce_etg=True
    )
    return json.loads(etg_json)


def _divisible(variant: dict) -> list:
    return [c["Divisible"] for c in variant["constraint_scope"]["hard_constraints"]
            if "Divisible" in c]


def test_chunk_split_divides_the_chunk_count(variants: list) -> None:
    chunk_split = [v for v in variants
                   if (v.get("variant_name") or v.get("name", "")).startswith(
                       "_mamba_chunk_scan__x6_y1y1y10__d0i0_d1i0_d2i1_d3i2")]
    assert chunk_split, "variant splitting the chunk loop over 10 cores not found"
    tile_c = [d for d in _divisible(chunk_split[0]) if '"tile_c"' in json.dumps(d)]
    assert tile_c, "no divisibility constraint on tile_c"
    for d in tile_c:
        assert d["x"] == {"Div": [{"Const": 10}, {"Sym": "tile_c"}]}, d
        assert d["by"] == {"Const": 10}, d


def test_no_constraint_divides_an_extent_of_one(variants: list) -> None:
    for v in variants:
        for d in _divisible(v):
            assert d["x"]["Div"][0] != {"Const": 1}, (v.get("variant_name"), d)


def test_spatially_split_tiles_must_divide_their_extent(variants: list) -> None:
    # The solver rounds division up, so `(10 / tile_c) mod 10 == 0` alone
    # accepts tiles that leave a partial last tile; every split symbol also
    # carries `(extent / tile) * tile == extent`.
    chunk_split = [v for v in variants
                   if (v.get("variant_name") or v.get("name", "")).startswith(
                       "_mamba_chunk_scan__x6_y1y1y10__d0i0_d1i0_d2i1_d3i2")][0]
    exact = [c["Eq"] for c in chunk_split["constraint_scope"]["hard_constraints"] if "Eq" in c]
    wanted = [{"Const": 10}, {"Sym": "tile_c"}]
    assert any(e[0] == {"Mul": [{"Div": wanted}, {"Sym": "tile_c"}]} and e[1] == {"Const": 10}
               for e in exact), exact
