"""Outer scaling: out[m, n] = x[m, n] * a[m] * b[n].

Memory-bound benchmark program for the reuse-aware tuner. a[m] is invariant
in n and b[n] in m, so no loop order serves both vectors; the tuner has to
weigh the two against the x and out traffic.

    python kernels/outer_scale.py --config kernels/config_files/outer_scale.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast


def _outer_scale(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out_ = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            col = a[tile_m, :]
            row = b[:, tile_n]
            out_[tile_m, tile_n] = (
                x[tile_m, tile_n]
                * broadcast(col, 1, [tile_m, tile_n])
                * broadcast(row, 0, [tile_m, tile_n])
            )
    return out_


class OuterScale(LoomKernel):
    kernel_name = "outer_scale"

    M: int = 256
    N: int = 512
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_outer_scale)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        x = torch.empty([cls.M, cls.N], device="cpu", dtype=torch.float16)
        a = torch.empty([cls.M, 1], device="cpu", dtype=torch.float16)
        b = torch.empty([1, cls.N], device="cpu", dtype=torch.float16)
        return (x, a, b)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    OuterScale(shape).run()
