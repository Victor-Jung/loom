"""Elementwise and matmul-epilogue lowering in loom2ttkernel.

A broadcast of a reduced tile is read by its consumers in place of a
materialized copy: as the broadcast operand of an FPU op, or unpacked into DST
with `unary_bcast`. A division by a broadcast becomes one reciprocal and a
multiply. The zero fill of a matmul accumulator that the first k block
overwrites is dropped, and the B-reader merge accepts an epilogue load.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
LOWER = ROOT / "third_party/loom2ttkernel/lower.sh"
OPT = Path(os.environ.get(
    "TILELOOM_TO_TTKERNEL_OPT",
    ROOT / "third_party/loom2ttkernel/build/bin/tileloom_to_ttkernel_opt"))


def lower(fixture: str, out: Path, options: str = "") -> Path:
    if not OPT.exists():
        pytest.skip(f"tileloom_to_ttkernel_opt not built at {OPT}")
    env = dict(os.environ,
               LOWER_MLIR_OUTPUT_DIR=str(out / "mlir"),
               SPLIT_KERNEL_OUTPUT_DIR=str(out / "kernels"),
               TILELOOM_TO_TTKERNEL_OPT=str(OPT))
    if options:
        env["TILELOOM_TO_TTKERNEL_OPTIONS"] = options
    result = subprocess.run([str(LOWER), str(FIXTURES / fixture), "1"],
                            capture_output=True, text=True, timeout=600,
                            cwd=ROOT, env=env)
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    return out / "kernels"


def test_column_softmax_reads_row_broadcasts_on_the_fpu(tmp_path) -> None:
    kernel = (lower("col_softmax_p03.mlir", tmp_path) / "compute.cpp").read_text()
    assert "unary_bcast" not in kernel
    assert kernel.count("sub_tiles_bcast<BroadcastType::ROW>(") == 1
    assert kernel.count("mul_tiles_bcast_rows(") == 1
    assert kernel.count("recip_tile(") == 1


def test_division_by_loop_carried_sum_unpacks_its_reciprocal(tmp_path) -> None:
    # The sum is read again by the next block, so its reciprocal goes to a
    # buffer of its own; exp(x - max) * (1 / sum) reads it with unary_bcast.
    kernel = (lower("softmax_twopass_p03.mlir", tmp_path) / "compute.cpp").read_text()
    assert "unary_bcast_init" not in kernel
    assert kernel.count("unary_bcast<BroadcastType::COL>(") == 1
    assert kernel.count("recip_tile(") == 1
    assert kernel.count("sub_tiles_bcast_cols(") == 2


def test_matmul_accumulator_zero_fill_is_dropped(tmp_path) -> None:
    kernel = (lower("gemm_bias_gelu_p03.mlir", tmp_path / "on") / "compute.cpp").read_text()
    assert "fill_tile(" not in kernel
    kept = (lower("gemm_bias_gelu_p03.mlir", tmp_path / "off",
                  "elide-accumulator-zero-fill=false") / "compute.cpp").read_text()
    assert "fill_tile(" in kept


def test_b_reader_merge_keeps_epilogue_loads_on_the_reader(tmp_path) -> None:
    kernels = lower("gemm_bias_gelu_p03.mlir", tmp_path,
                    "matmul-merge-b-reader-into-writer=true")
    reader = (kernels / "reader.cpp").read_text()
    writer = (kernels / "writer.cpp").read_text()
    # A and the bias on the reader, B and the output store on the writer.
    assert reader.count("noc_async_read_tile(") == 2
    assert writer.count("noc_async_read_tile(") == 1
    assert writer.count("noc_async_write_tile(") == 1
