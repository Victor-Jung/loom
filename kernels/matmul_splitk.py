"""Matmul, variant splitk: K made partly spatial and reduced across tiles.

The baseline keeps K as a sequential in-core reduction, so at large K each core
streams the whole K extent for one output tile. Here K is tiled spatially and
the partials are combined with the gather + tile.id == 0 idiom from mqa_decode,
Helion having no atomic_add.

CLI as kernels/matmul.py.
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import gather


def _matmul_splitk(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out_ = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile_m, tile_n, tile_s in hl.tile([m, n, k]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
        for tile_k in hl.tile(tile_s.begin, tile_s.end):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        gathered = gather(tile_s, acc)
        if tile_s.id == 0:
            out_[tile_m, tile_n] = torch.sum(gathered, 0)
    return out_


class MatmulSplitK(LoomKernel):
    kernel_name = "matmul_splitk"

    M: int = 4096
    K: int = 256
    N: int = 4096
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_matmul_splitk)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        """Return concrete input tensors that define M, K, N at MLIR-gen time."""
        x = torch.empty([cls.M, cls.K], device="cpu", dtype=torch.float16)
        y = torch.empty([cls.K, cls.N], device="cpu", dtype=torch.float16)
        return (x, y)


if __name__ == "__main__":
    try:
        shape, normalized_argv = resolve_kernel_shape_args(sys.argv)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    sys.argv = normalized_argv
    kernel = MatmulSplitK(shape)
    kernel.run()
