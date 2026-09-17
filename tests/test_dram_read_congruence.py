"""Sub-tile DRAM reads must satisfy Blackhole's NOC address rule.

tt-metal requires (l1_addr % A) == (dram_addr % A) for DRAM reads, where A is
NOC_DRAM_READ_ALIGNMENT_BYTES: 64 on Blackhole, 32 on Wormhole. The generated
vector loads place 32-byte chunks whose DRAM offsets are 32 apart into L1 face
slots 512 apart, so on Blackhole one read of every pair would be rejected by the
NOC -- which surfaces as a hang, not an error, unless the watcher is enabled.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPLIT = Path(__file__).resolve().parents[1] / "third_party" / "loom2ttkernel" / "split_kernel.py"
spec = importlib.util.spec_from_file_location("split_kernel", SPLIT)
sk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sk)

VECTOR_LOAD = [
    "  uint64_t temp_312 = v27.get_noc_addr(v133 / v24, v133 % v24);\n",
    "  noc_async_read(temp_312, v131, v9);\n",
    "  uint64_t temp_322 = v27.get_noc_addr(v134 / v24, v134 % v24);\n",
    "  noc_async_read(temp_322, v135, v9);\n",
]


def test_raw_vector_loads_are_rewritten() -> None:
    out, count = sk._apply_dram_read_congruence(VECTOR_LOAD)
    assert count == 2
    body = "".join(out)
    assert "loom_read_congruent(v27, v133 / v24, v133 % v24, v131, v9);" in body
    assert "loom_read_congruent(v27, v134 / v24, v134 % v24, v135, v9);" in body
    assert "noc_async_read(temp_" not in body


def test_tile_reads_are_left_alone() -> None:
    """noc_async_read_tile is page-addressed and always congruent."""
    lines = ["  noc_async_read_tile(v138, v32, v136);\n", "  noc_async_read_barrier();\n"]
    out, count = sk._apply_dram_read_congruence(lines)
    assert count == 0 and out == lines


def test_kernels_without_vector_loads_are_untouched() -> None:
    """Flash attention and matmul read whole tiles only; they must not even
    receive the helper."""
    lines = ["void kernel_main() {\n", "  noc_async_read_tile(a, b, c);\n", "}\n"]
    out, count = sk._apply_dram_read_congruence(lines)
    assert count == 0 and out == lines


def test_helper_preserves_indentation() -> None:
    out, _ = sk._apply_dram_read_congruence(
        ["    uint64_t temp_1 = acc.get_noc_addr(x / p, x % p);\n",
         "    noc_async_read(temp_1, dst, len);\n"]
    )
    assert out[-1].startswith("    loom_read_congruent(")


def test_helper_block_is_self_contained() -> None:
    block = "".join(sk._build_reader_congruence_helper())
    assert "loom_read_congruent" in block
    assert "noc_async_read_barrier();" in block   # bounce must complete before the shift
    assert "& 63u" in block                        # 64-byte phase on Blackhole
