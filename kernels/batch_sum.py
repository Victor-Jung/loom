"""Sum over the batch: out[m, n] = sum_b x[b, m, n].

Benchmark program whose only loop-carried loop accumulates with an
elementwise add instead of a matmul.

    python kernels/batch_sum.py --config kernels/config_files/batch_sum.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args


def _batch_sum(x: torch.Tensor) -> torch.Tensor:
    b, m, n = x.size()
    out_ = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
            for tile_b in hl.tile(b, block_size=1):
                acc = acc + x[tile_b.begin, tile_m, tile_n]
            out_[tile_m, tile_n] = acc
    return out_

class BatchSum(LoomKernel):
    kernel_name = "batch_sum"

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
    )(_batch_sum)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        return (torch.empty([cls.B, cls.M, cls.N], device="cpu", dtype=torch.float16),)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    BatchSum(shape).run()
