"""Regressions for rank-2 vector slices through the Loom pipeline.

A kernel that slices a vector out of a higher-rank tensor, e.g. ``v[tile_m, :]``
producing ``[tile_m, 1]``, exercises several passes that previously assumed
every load is a full 2-D tile. ``kernels/_regress_vecslice.py`` is the minimal
kernel that reproduces it.

These are integration tests: they run the real pipeline, so they are skipped
when the compiled ``loom_pipeline`` extension is unavailable.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("loom_pipeline", reason="requires the built loom-dataflow extension")

REPO = Path(__file__).resolve().parents[1]
KERNEL = REPO / "kernels" / "_regress_vecslice.py"

# loom.alloc with a single dimension, e.g. "loom.alloc [32] on @L1"
RANK1_ALLOC = re.compile(r"loom\.alloc \[\s*\d+\s*\] on @L1")
ALLOC = re.compile(r"loom\.alloc ")
TAKE = re.compile(r"loom\.semaphore_take ")


@pytest.fixture(scope="module")
def bufferized_ir(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("vecslice")
    cfg = out / "cfg.json"
    cfg.write_text(json.dumps({
        "output_path": str(out / "run"),
        "hw_spec": "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus.mlir",
    }))
    proc = subprocess.run(
        [sys.executable, str(KERNEL), "-M256_N256_K256", "--config", str(cfg),
         "--njobs", "4", "--debug", "--topk-candidates", "1", "--topk-block-size", "1"],
        cwd=REPO, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, f"pipeline failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    ir = out / "run" / "IRs" / "p03_bufferized.mlir"
    assert ir.exists(), "pipeline produced no bufferized IR"
    return ir.read_text()


def test_vector_slice_keeps_rank_two(bufferized_ir: str) -> None:
    """A [n, 1] slice must stay rank 2 all the way to the bufferized IR.

    trace_shape previously dropped *every* static-1 dim from a rank-reducing
    subview rather than only the dims the subview actually drops, so a slice of
    sizes [1,1,1,?,1] producing memref<?x1xf16> collapsed to rank 1. That then
    tripped the L1 footprint estimator's "L1 alloc rank is smaller than 2".
    """
    offenders = RANK1_ALLOC.findall(bufferized_ir)
    assert not offenders, f"rank-1 L1 allocations reached the bufferized IR: {offenders}"


def test_vector_slice_alloc_is_present(bufferized_ir: str) -> None:
    """Guard the fixture itself: if the [n, 1] alloc stops being generated the
    test above would pass vacuously."""
    assert re.search(r"loom\.alloc \[\d+, 1\]", bufferized_ir), (
        "reproducer no longer produces a unit-minor-dim allocation; "
        "test_vector_slice_keeps_rank_two would pass for the wrong reason"
    )


def test_every_alloc_is_bound(bufferized_ir: str) -> None:
    """Every loom.alloc needs at least one loom.semaphore_take, or TT codegen
    fails with "host CB emission requires at least one explicit
    loom.semaphore_take for each loom.alloc".

    The vector-slice allocation used to be unbound. deriveTensorSizes() in
    canonical_bufferization_to_loom_pass dropped every static-1 dim when
    computing sizes from a subview, so for a [1,1,1,?,1] subview producing
    memref<?x1xf16> it produced one size against a rank-2 result. The rank
    check then rejected the op, leaving it as a generic
    bufferization.to_tensor. Memory binding only matches
    loom.bufferize_to_tensor, so the load was never bound and its
    semaphore_take was dead-code eliminated.
    """
    n_alloc = len(ALLOC.findall(bufferized_ir))
    n_take = len(TAKE.findall(bufferized_ir))
    assert n_alloc <= n_take, (
        f"{n_alloc} loom.alloc but only {n_take} loom.semaphore_take: "
        "some allocation was never bound to a buffer"
    )
