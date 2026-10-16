"""The tuner's estimate is evaluated at tiles a tree leaves room for.

`Tuner::pickTiles` stands in for the solver's tile choice: each tile symbol
starts at the largest multiple of 32 that divides its extent and leaves every
core of a split loop a tile (tile * cores <= extent), or, on a split loop, at
the largest tile whose padded work per core (whole waves) is within 25% of
the least; while the L1 buffers
(loads twice for double buffering, stores, compute temporaries) exceed one
core's L1, the tile holding the most bytes steps down. More cores therefore
mean smaller tiles and more traffic, which the estimate now sees.
"""

from __future__ import annotations

import re

from test_mapping_tune import MATMUL, tune

HW12x10 = "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x12y10.mlir"


def fixed(schedule: str) -> tuple[str, float]:
    """Picked tiles and DRAM traffic (bytes) of the tree after `schedule`."""
    result = tune(MATMUL, "policy=fixed", f"options={schedule}", f"hw-spec={HW12x10}", "dump-tree")
    assert result.returncode == 0, result.stderr[-2000:]
    tiles = re.search(r"^picked tiles: (.*)$", result.stderr, re.M).group(1)
    dram = float(re.search(r"^cost terms: .*dram_bytes=([0-9.e+]+)", result.stderr, re.M).group(1))
    return tiles, dram


def test_more_cores_leave_room_for_smaller_tiles_only() -> None:
    # 2048 x 16384 @ 16384 x 2048. On 8 x 8 cores m and n may reach 256; k
    # shrinks until A, B (double buffered), C and the accumulator fit L1.
    tiles64, dram64 = fixed("split(m,8);split(n,8)")
    assert tiles64 == "tile_k=512 tile_m=256 tile_n=256", tiles64
    # On 12 x 10 cores m and n stop at 128 (2048 / 12 and 2048 / 10). On m,
    # 16 tiles over 12 cores pad every core to 2 x 128 rows where 64-row
    # tiles need 3 x 64, so m takes 64; on n, 128 and 64 pad alike.
    tiles120, dram120 = fixed("split(m,12);split(n,10)")
    assert tiles120 == "tile_k=1024 tile_m=64 tile_n=128", tiles120
    # Half the tile edge doubles the refetches of A and B.
    assert dram64 < dram120, (dram64, dram120)


def test_bounds_option_restores_the_two_point_estimate() -> None:
    result = tune(MATMUL, "policy=maxpar", "options=tiles=bounds", f"hw-spec={HW12x10}", "dump-tree")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "picked tiles" not in result.stderr
