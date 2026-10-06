"""A cyclic split index divided by its core count is the wave alone.

The tuner recomposes a split loop as `axis + wave * cores`. A program that
indexes an operand by `h // k` (mamba's group index) then divides that
composite; syntactically the quotient depends on the spatial axis, so the
reuse and broadcast analyses would refetch the operand on every core. The
`loom-simplify-split-index` pass folds `(axis + wave * 8) / 8` to `wave`
and `% 8` to `axis` when the axis is bounded below the divisor, and the
exploration then multicasts the group-indexed operands along the head axis.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
HW12x10 = ROOT / "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x12y10.mlir"
LOOM_OPT = Path(os.environ.get(
    "LOOM_OPT", ROOT / "third_party/loom-dataflow/build/tool/loom-opt/loom-opt"))


def test_divide_and_remainder_of_split_index_fold() -> None:
    if not LOOM_OPT.exists():
        pytest.skip(f"loom-opt not built at {LOOM_OPT}")
    res = subprocess.run(
        [str(LOOM_OPT), "--loom-simplify-split-index", str(FIXTURES / "split_index_divrem.mlir")],
        capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    out = res.stdout
    # h / 8 and h % 8 are gone; h / 4 stays because axis / 4 is not constant.
    assert "arith.divui %" in out and out.count("arith.divui") == 1, out
    assert "arith.remui" not in out, out
    # The stores read the wave and the axis directly.
    body = out[out.index("scf.for"):]
    assert re.search(r"memref\.store %arg2, %arg0\[%\w+\]", body), body
    assert re.search(r"memref\.store %arg1, %arg0\[%arg2\]", body), body


def test_mamba_head_split_multicasts_group_operands() -> None:
    loom_pipeline = pytest.importorskip("loom_pipeline")
    if not HW12x10.exists():
        pytest.skip(f"hardware spec missing: {HW12x10}")
    p00 = (FIXTURES / "etg_mamba_p00.mlir").read_text()
    tuned = loom_pipeline.run_mapping_tune(p00, "fixed", "split(h,8);split(c,10)", False, str(HW12x10))
    assert "d0 + d1 * 8" in tuned
    out, _ = loom_pipeline.run_exploration(input_mlir=tuned, hw_spec_file=str(HW12x10),
                                           produce_etg=False)
    names = re.findall(r"func\.func @(\S+?)\(", out)
    assert names
    for name in names:
        # Every placement multicasts along the head axis: cb and C (group
        # indexed) and D (chunk invariant) give three broadcast copies.
        assert len(re.findall(r"bc8", name)) >= 2, name
        assert "bc10" in name, name
    assert "arith.divui" not in out, "the group index still divides the split head index"
