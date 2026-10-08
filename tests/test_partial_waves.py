"""Splits whose cores do not divide their tiles get a correct last wave.

`loom-partial-waves` runs right after the block sizes are materialized. For
a wave loop whose tile count is not a multiple of its cores it computes
`valid = index < tiles` from the loop's recomposition and

- guards the rest of the wave body with `scf.if valid` when no copy in the
  body multicasts (every kernel is cloned from the same function, so all
  skip the same iterations);
- otherwise shadows the iteration: DRAM views are clamped into their
  tensors and DRAM stores carry `if valid`, because a multicast sender
  waits for every receiver of its group.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
HW12x10 = ROOT / "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x12y10.mlir"

loom_pipeline = pytest.importorskip("loom_pipeline")


def materialize(p00: str, schedule: str, sizes: dict[str, int]) -> str:
    tuned = loom_pipeline.run_mapping_tune(p00, "fixed", schedule, False, str(HW12x10))
    explored, _ = loom_pipeline.run_exploration(input_mlir=tuned, hw_spec_file=str(HW12x10),
                                                produce_etg=False)
    func = re.findall(r"func\.func @(\S+?)\(", explored)[0]
    blocks = {"__loom_candidate_order__": [func], func: [sizes]}
    return loom_pipeline.run_materialization(input_mlir=explored, block_sizes_json=json.dumps(blocks))


def test_uneven_split_without_multicast_is_guarded() -> None:
    # x + y over [b, m, n]: no shared operand, m over 3 cores, 8 tiles.
    p00 = (FIXTURES / "tune_ewise3d_p00.mlir").read_text()
    out = materialize(p00, "split(m,3)", {"tile_b": 1, "tile_m": 32, "tile_n": 32})
    assert 'loom.partial_wave = "guard"' in out, out[-3000:]
    assert "scf.if" in out
    assert " if %" not in out  # no predicated copy on the guard path


def test_uneven_split_with_multicast_is_shadowed() -> None:
    # scaled_ewise multicasts s over the b axis: b over 3 cores, 8 tiles.
    p00 = (FIXTURES / "tune_scaled_ewise_p00.mlir").read_text()
    out = materialize(p00, "split(b,3)", {"tile_b": 1, "tile_m": 32, "tile_n": 32})
    assert 'loom.partial_wave = "shadow"' in out, out[-3000:]
    stores = [l for l in out.splitlines() if "loom.copy" in l and "dst_mem_space @mem_DRAM" in l]
    assert stores and all(" if %" in l for l in stores), stores
    assert "arith.select" in out  # clamped DRAM offsets


def test_even_split_is_untouched() -> None:
    # Pre-split fixture: m over 8 cores; with 32-row tiles m has 8 tiles.
    p00 = (FIXTURES / "presplit_scaled_ewise_p00.mlir").read_text()
    explored, _ = loom_pipeline.run_exploration(input_mlir=p00, hw_spec_file=str(HW12x10),
                                                produce_etg=False)
    func = re.findall(r"func\.func @(\S+?)\(", explored)[0]
    sizes = {s: 32 for s in re.findall(r"loom\.sym @(\w+)", explored)}
    blocks = {"__loom_candidate_order__": [func], func: [sizes]}
    out = loom_pipeline.run_materialization(input_mlir=explored, block_sizes_json=json.dumps(blocks))
    assert "loom.partial_wave" not in out
