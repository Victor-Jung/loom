"""Subtract the row mean: out[m, :] = x[m, :] - mean(x[m, :]).

Benchmark program combining a row reduction, a scalar scale and a column
broadcast with no loop-carried values.

    python kernels/row_center.py --config kernels/config_files/row_center.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast


def _row_center(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    n_s = hl.specialize(n)
    out_ = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :]
        inv_n = hl.full([], 1.0 / n_s, dtype=torch.float16)
        mean = torch.sum(row, -1, keepdim=True) * inv_n
        out_[tile_m, :] = row - broadcast(mean, 1, [tile_m, n_s])
    return out_

class RowCenter(LoomKernel):
    kernel_name = "row_center"

    M: int = 512
    N: int = 256
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_row_center)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        return (torch.empty([cls.M, cls.N], device="cpu", dtype=torch.float16),)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    RowCenter(shape).run()
