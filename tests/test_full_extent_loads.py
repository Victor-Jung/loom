"""Full-extent slices bind, type, and store like any other tile.

`attention_fullrow` reads `k[:, :]` and `v[:, :]`. Three things used to go
wrong: the frontend typed the transposed slice from a block-size symbol it
could not resolve, so the module did not parse; the slice's identity subview
is folded away and memory binding only bound loads behind a `memref.subview`,
so the operand read DRAM directly and the ETG could not trace it; and the
stored matmul result was kept apart from its own fill init, leaving a copy
chain that later folded into an alloc without a take.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "tests" / "fixtures" / "bind_attention_fullrow_p00.mlir"
HW_SPEC = ROOT / "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x1y1.mlir"

loom_pipeline = pytest.importorskip("loom_pipeline")


@pytest.fixture(scope="module")
def explored() -> str:
    if not HW_SPEC.exists():
        pytest.skip(f"hardware spec missing: {HW_SPEC}")
    out, _ = loom_pipeline.run_exploration(
        input_mlir=FIXTURE.read_text(), hw_spec_file=str(HW_SPEC), produce_etg=True
    )
    return out


def test_frontend_types_the_transposed_slice_statically() -> None:
    pytest.importorskip("helion")
    from kernels.attention_fullrow import AttentionFullRow

    p00 = AttentionFullRow.generate_mlir()
    (matmul,) = [l for l in p00.splitlines() if "linalg.matmul" in l and "transposed" in l]
    assert "tensor<64x256xf16>" in matmul, matmul
    assert "func.func" in loom_pipeline.run_mapping_tune(p00, "identity")


def test_full_extent_loads_are_copied_into_l1(explored: str) -> None:
    assert not re.search(r"bufferize_to_tensor %arg\d", explored)
    assert len(re.findall(r"loom\.subview %arg[12]\[0, 0\] \[256, 64\]", explored)) == 2


def test_every_alloc_has_a_take(explored: str) -> None:
    for alloc in re.findall(r"(%\w+) = loom\.alloc", explored):
        assert re.search(rf"loom\.semaphore_take {re.escape(alloc)} ", explored), alloc
    assert "linalg.copy" not in explored
