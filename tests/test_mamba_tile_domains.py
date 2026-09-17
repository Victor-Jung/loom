"""A tile consumed as `.begin` must have its block size pinned to 1.

`hl.tile(...)` without a block_size leaves the symbol free, and the frontend
advertises the dimension's full extent as its upper bound. That is correct when
the tile is used as a *range* -- the emitted slice then has extent `tile_X`, so
striding by `tile_X` covers the dimension.

It is wrong when the tile is used as `.begin`, a scalar index: the slice has
extent 1 while the loop still strides by `tile_X` and runs `ceil(extent/tile_X)`
times. The solver is then free to pick the whole dimension, which it prefers
because fewer iterations model as cheaper -- and the loop runs once, at offset 0,
reading one element. mamba_chunk_scan computed 1/64 of its output that way
(tile_b=2, tile_h=32), silently and with no error anywhere.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("helion")

# Tiles this kernel indexes with `.begin`; each produces an extent-1 slice.
SCALAR_INDEXED = {"tile_b", "tile_c", "tile_h"}


@pytest.fixture(scope="module")
def tile_upper_bounds() -> dict[str, int]:
    from kernels.mamba_chunk_scan import MambaChunkScan

    kernel = MambaChunkScan({"B": 2, "L": 384, "N": 32, "H": 32, "G": 4, "D": 32, "C": 192})
    mlir = kernel.generate_mlir()
    return {
        name: int(bound)
        for name, bound in re.findall(
            r"symbol_ref = @(tile_\w+), upper_bound = (\d+)", mlir
        )
    }


@pytest.mark.parametrize("tile", sorted(SCALAR_INDEXED))
def test_scalar_indexed_tiles_are_pinned(tile_upper_bounds: dict[str, int], tile: str) -> None:
    assert tile in tile_upper_bounds, f"{tile} missing from {sorted(tile_upper_bounds)}"
    assert tile_upper_bounds[tile] == 1, (
        f"{tile} is indexed as .begin (extent-1 slice) but advertises "
        f"upper_bound={tile_upper_bounds[tile]}; the solver may pick a block "
        f"size >1, and the loop would then skip elements"
    )


def test_range_indexed_tiles_keep_their_full_domain(tile_upper_bounds: dict[str, int]) -> None:
    """tile_m/tile_n/tile_k slice by range, so blocking them is legal and the
    solver should still be free to explore."""
    for tile in ("tile_m", "tile_n", "tile_k"):
        assert tile_upper_bounds.get(tile, 0) > 1, (
            f"{tile} should keep a real search domain, got {tile_upper_bounds.get(tile)}"
        )
