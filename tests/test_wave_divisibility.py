"""Spatial splits no longer need cores dividing tiles (phase G).

A wave loop's last wave may be partial: `loom-partial-waves` guards or
shadows the iterations past the end at materialization. The ETG therefore
carries no `Divisible(tiles, cores)` constraint for a split loop; the tile
must still divide its extent (an `Eq` constraint).
"""


from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P00 = ROOT / "tests" / "fixtures" / "tune_scaled_ewise_p00.mlir"
HW_SPEC = ROOT / "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x12y10.mlir"

loom_pipeline = pytest.importorskip("loom_pipeline")


def presplit(p00: str) -> str:
    """The exploration only places spatial loops the tuner split (maxpar)."""
    return loom_pipeline.run_mapping_tune(p00, "maxpar", "", False, str(HW_SPEC))


def test_spatially_split_loops_carry_no_core_divisibility() -> None:
    if not HW_SPEC.exists():
        pytest.skip(f"hardware spec missing: {HW_SPEC}")
    _, etg_json = loom_pipeline.run_exploration(
        input_mlir=presplit(P00.read_text()), hw_spec_file=str(HW_SPEC), produce_etg=True
    )
    variants = json.loads(etg_json)
    assert variants
    for variant in variants:
        constraints = variant["constraint_scope"]["hard_constraints"]
        assert not [c for c in constraints if "Divisible" in c], variant.get("variant_name")
        assert [c for c in constraints if "Eq" in c], variant.get("variant_name")
