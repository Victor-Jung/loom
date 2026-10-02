"""Elementwise add over a rank-3 tensor: out[b, m, n] = x[b, m, n] + y[b, m, n].

Toy program for the mapping-program tuner with no reduction loop: one spatial
loop (b, one batch per iteration) over two temporal loops (m, n) without
loop-carried values, so the two temporal loops can be interchanged freely.

    python kernels/toy_ewise3d.py --config kernels/config_files/toy_ewise3d.json --debug
    python kernels/toy_ewise3d.py --config kernels/config_files/toy_ewise3d.json --debug \
        --tune fixed --tune-options "interchange(m,n)"
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args


def _ewise3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, n = x.size()
    out_ = torch.empty([b, m, n], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(b, block_size=1):
        for tile_m in hl.tile(m):
            for tile_n in hl.tile(n):
                out_[tile_b.begin, tile_m, tile_n] = (
                    x[tile_b.begin, tile_m, tile_n] + y[tile_b.begin, tile_m, tile_n]
                )
    return out_


class ToyEwise3d(LoomKernel):
    kernel_name = "toy_ewise3d"

    B: int = 2
    M: int = 256
    N: int = 512
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_ewise3d)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        x = torch.empty([cls.B, cls.M, cls.N], device="cpu", dtype=torch.float16)
        y = torch.empty([cls.B, cls.M, cls.N], device="cpu", dtype=torch.float16)
        return (x, y)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    ToyEwise3d(shape).run()
