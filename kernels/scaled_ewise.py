"""Batched elementwise scale with one shared operand: out[b] = x[b] * s.

Benchmark program for loop-order search. s[m, n] does not depend on b, so
with b innermost the s tile is reused across the batch; with b outermost it is
re-fetched for every (b, m, n) tile.

    python kernels/scaled_ewise.py --config kernels/config_files/scaled_ewise.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args


def _scaled_ewise(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    b, m, n = x.size()
    out_ = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_b in hl.tile(b, block_size=1):
            for tile_n in hl.tile(n):
                out_[tile_b.begin, tile_m, tile_n] = (
                    x[tile_b.begin, tile_m, tile_n] * s[tile_m, tile_n]
                )
    return out_


class ScaledEwise(LoomKernel):
    kernel_name = "scaled_ewise"

    B: int = 8
    M: int = 256
    N: int = 256
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_scaled_ewise)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        x = torch.empty([cls.B, cls.M, cls.N], device="cpu", dtype=torch.float16)
        s = torch.empty([cls.M, cls.N], device="cpu", dtype=torch.float16)
        return (x, s)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    ScaledEwise(shape).run()
