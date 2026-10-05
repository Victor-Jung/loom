"""Column softmax over full columns: out[:, n] = softmax(x[:, n]).

Benchmark program for column reductions: amax and sum reduce along the
rows of a tile (REDUCE_COL) and the [1, n] results are broadcast back over
the rows.

    python kernels/col_softmax.py --config kernels/config_files/col_softmax.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast


def _col_softmax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    m_s = hl.specialize(m)
    out_ = torch.empty_like(x)
    for tile_n in hl.tile(n):
        col = x[:, tile_n]
        mx = torch.amax(col, 0, keepdim=True)
        p = torch.exp(col - broadcast(mx, 0, [m_s, mx.size(1)]))
        s = torch.sum(p, 0, keepdim=True)
        out_[:, tile_n] = p / broadcast(s, 0, [m_s, s.size(1)])
    return out_


class ColSoftmax(LoomKernel):
    kernel_name = "col_softmax"

    M: int = 256
    N: int = 512
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_col_softmax)

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
    ColSoftmax(shape).run()
