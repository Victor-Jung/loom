"""Row reductions lower for any input rank.

The TTKernel reduction rewrite used to accept only the rank-3
`[batch, m, n]` input shape of flash attention. `row_softmax` reduces a
rank-2 `[m, n]` tile along its rows twice (max, then sum); both must lower
to `reduce_tile` with the row reduce dimension, and the fixture must reach
the split kernels.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "row_softmax_p03.mlir"
LOWER = ROOT / "third_party/loom2ttkernel/lower.sh"
OPT = Path(os.environ.get(
    "TILELOOM_TO_TTKERNEL_OPT",
    ROOT / "third_party/loom2ttkernel/build/bin/tileloom_to_ttkernel_opt"))


@pytest.fixture(scope="module")
def compute_kernel(tmp_path_factory) -> str:
    if not OPT.exists():
        pytest.skip(f"tileloom_to_ttkernel_opt not built at {OPT}")
    out = tmp_path_factory.mktemp("softmax")
    env = dict(os.environ,
               LOWER_MLIR_OUTPUT_DIR=str(out / "mlir"),
               SPLIT_KERNEL_OUTPUT_DIR=str(out / "kernels"))
    result = subprocess.run([str(LOWER), str(FIXTURE), "1"], capture_output=True,
                            text=True, timeout=600, cwd=ROOT, env=env)
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    return (out / "kernels" / "compute.cpp").read_text()


def test_rank2_row_reductions_lower_to_reduce_tile(compute_kernel: str) -> None:
    inits = re.findall(r"reduce_init<PoolType::(\w+), ReduceDim::REDUCE_ROW", compute_kernel)
    assert inits == ["MAX", "SUM"], inits
    assert compute_kernel.count("reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW") == 1
    assert compute_kernel.count("reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW") == 1


def test_softmax_epilogue_lowers(compute_kernel: str) -> None:
    assert "exp_tile" in compute_kernel
    # The row max and sum broadcasts are folded into their consumers: x - max
    # and the multiply by 1 / sum read the reduced tile as an FPU broadcast
    # operand, so no broadcast is materialized and the reciprocal runs once.
    assert "unary_bcast" not in compute_kernel
    assert compute_kernel.count("sub_tiles_bcast_cols(") == 1
    assert compute_kernel.count("mul_tiles_bcast_cols(") == 1
    assert compute_kernel.count("recip_tile(") == 1


def test_softmax_epilogue_materializes_without_fpu_folding(tmp_path) -> None:
    if not OPT.exists():
        pytest.skip(f"tileloom_to_ttkernel_opt not built at {OPT}")
    env = dict(os.environ,
               TILELOOM_TO_TTKERNEL_OPTIONS="fpu-elementwise=false",
               LOWER_MLIR_OUTPUT_DIR=str(tmp_path / "mlir"),
               SPLIT_KERNEL_OUTPUT_DIR=str(tmp_path / "kernels"))
    result = subprocess.run([str(LOWER), str(FIXTURE), "1"], capture_output=True,
                            text=True, timeout=600, cwd=ROOT, env=env)
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    kernel = (tmp_path / "kernels" / "compute.cpp").read_text()
    assert "unary_bcast<BroadcastType::COL>" in kernel
    assert "_tiles_bcast_cols(" not in kernel
