"""Row softmax over full rows: out[m, :] = softmax(x[m, :]).

Benchmark program with no reduction loop: each row block is reduced inside
the tile (amax, sum), so the program exercises the reduction, exp, divide and
column-broadcast primitives rather than loop-carried accumulation.

    python kernels/row_softmax.py --config kernels/config_files/row_softmax.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast


def _row_softmax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    n_s = hl.specialize(n)
    out_ = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :]
        mx = torch.amax(row, -1, keepdim=True)
        p = torch.exp(row - broadcast(mx, 1, [mx.size(0), n_s]))
        s = torch.sum(p, -1, keepdim=True)
        out_[tile_m, :] = p / broadcast(s, 1, [s.size(0), n_s])
    return out_


class RowSoftmax(LoomKernel):
    kernel_name = "row_softmax"

    M: int = 512
    N: int = 256
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_row_softmax)

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
    RowSoftmax(shape).run()
