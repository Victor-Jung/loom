"""Spatial splits must divide the loop they split.

A wave loop over a spatial split has no guard for a partial last wave, so
the ETG constrains `extent / tile` to be divisible by the number of cores of
every spatially split loop. On a 12x10 mesh the m loop of `scaled_ewise`
(M = 2048) is split over the x and y dimensions in many ways; each variant
with more than one core must carry a `Divisible` hard constraint whose
divisor is that variant's core count.
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


def test_spatially_split_loops_carry_a_divisibility_constraint() -> None:
    if not HW_SPEC.exists():
        pytest.skip(f"hardware spec missing: {HW_SPEC}")
    _, etg_json = loom_pipeline.run_exploration(
        input_mlir=P00.read_text(), hw_spec_file=str(HW_SPEC), produce_etg=True
    )
    variants = json.loads(etg_json)
    checked = 0
    for variant in variants:
        name = variant.get("variant_name") or variant.get("name") or ""
        m = re.search(r"__x([\d]+)_y([\d]+)__", name) or re.search(r"__x([\dx]+)_y([\dy\d]+)__", name)
        if not m:
            continue
        cores = 1
        for part in (m.group(1), m.group(2)):
            for f in re.findall(r"\d+", part):
                cores *= int(f)
        constraints = variant["constraint_scope"]["hard_constraints"]
        divisible = [c for c in constraints if "Divisible" in c]
        if cores == 1:
            assert not divisible, name
            continue
        checked += 1
        assert divisible, f"{name}: no Divisible constraint for {cores} cores"
        divisors = {json.dumps(c["Divisible"]["by"]) for c in divisible}
        assert json.dumps({"Const": cores}) in divisors, (name, divisors)
    assert checked > 0
