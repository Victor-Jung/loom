"""A reduction that accumulates into a loop-carried tensor lowers in place.

`softmax_twopass` computes `s = s + sum(exp(x - m))` over tiled `n`: after
tensor canonicalisation the sum generic writes straight into the carried
`s`, whose buffer is a one-block circular buffer. Each iteration must read
the previous partial sum back into DST before reducing (`copy_tile`), and
drive the buffer with a wait/pop before the reserve/push so the block is
replaced rather than appended; appending blocked the second iteration in
`cb_reserve_back` for ever.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "softmax_twopass_p03.mlir"
LOWER = ROOT / "third_party/loom2ttkernel/lower.sh"
OPT = Path(os.environ.get(
    "TILELOOM_TO_TTKERNEL_OPT",
    ROOT / "third_party/loom2ttkernel/build/bin/tileloom_to_ttkernel_opt"))


@pytest.fixture(scope="module")
def compute_kernel(tmp_path_factory) -> str:
    if not OPT.exists():
        pytest.skip(f"tileloom_to_ttkernel_opt not built at {OPT}")
    out = tmp_path_factory.mktemp("twopass")
    env = dict(os.environ,
               LOWER_MLIR_OUTPUT_DIR=str(out / "mlir"),
               SPLIT_KERNEL_OUTPUT_DIR=str(out / "kernels"))
    result = subprocess.run([str(LOWER), str(FIXTURE), "1"], capture_output=True,
                            text=True, timeout=600, cwd=ROOT, env=env)
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    return (out / "kernels" / "compute.cpp").read_text()


def _sum_reduce_block(compute_kernel: str) -> str:
    start = compute_kernel.index("reduce_init<PoolType::SUM")
    head = compute_kernel.rfind("cb_wait_front", 0, start)
    return compute_kernel[head:compute_kernel.index("cb_push_back", start) + 40]


def test_carried_sum_preloads_the_previous_partial(compute_kernel: str) -> None:
    block = _sum_reduce_block(compute_kernel)
    cb = re.search(r"reduce_init<PoolType::SUM[^(]*\(\w+, \w+, (\w+)\)", block).group(1)
    assert f"copy_tile_init({cb})" in block
    assert re.search(rf"copy_tile\({cb}, \w+, \w+\)", block), block


def test_carried_sum_replaces_its_block(compute_kernel: str) -> None:
    block = _sum_reduce_block(compute_kernel)
    cb = re.search(r"reduce_init<PoolType::SUM[^(]*\(\w+, \w+, (\w+)\)", block).group(1)
    order = [m.group(1) for m in re.finditer(
        rf"(cb_wait_front|cb_pop_front|cb_reserve_back|cb_push_back)\({cb},", block)]
    assert order == ["cb_wait_front", "cb_pop_front", "cb_reserve_back", "cb_push_back"], order


def test_max_pass_is_not_treated_as_accumulating(compute_kernel: str) -> None:
    start = compute_kernel.index("reduce_init<PoolType::MAX")
    block = compute_kernel[compute_kernel.rfind("tile_regs_acquire", 0, start):start]
    assert "copy_tile_init" not in block
