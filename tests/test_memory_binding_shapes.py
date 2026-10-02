"""Memory binding accepts programs with zero or several loop-carried loops.

Liveness is computed over the loop tree: a value used inside a loop it is not
defined in lives to that loop's end, every loop-carried loop fuses its own
init/iter_arg/yield/result, and a program without any accumulation loop binds
with per-value lifetimes. These fixtures are the two shapes the former
"exactly one innermost loop-carried scf.for" guard rejected.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
HW_SPEC = ROOT / "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x1y1.mlir"

loom_pipeline = pytest.importorskip("loom_pipeline")


def explore(fixture: Path) -> str:
    if not HW_SPEC.exists():
        pytest.skip(f"hardware spec missing: {HW_SPEC}")
    out, _ = loom_pipeline.run_exploration(
        input_mlir=fixture.read_text(), hw_spec_file=str(HW_SPEC), produce_etg=False
    )
    return out


def test_no_loop_carried_loop_binds() -> None:
    out = explore(FIXTURES / "bind_ewise3d_p00.mlir")
    assert re.search(r"func\.func @", out)
    assert "loom.alloc" in out and "loom.semaphore_take" in out
    assert "iter_args" not in out


def test_two_nested_loop_carried_loops_bind() -> None:
    out = explore(FIXTURES / "bind_chain_fused_p00.mlir")
    assert re.search(r"func\.func @", out)
    assert out.count("iter_args(") >= 2
    assert "loom.alloc" in out and "loom.semaphore_take" in out
