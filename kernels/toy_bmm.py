"""Batched matmul as B independent 2D matmuls: out[b] = x[b] @ y[b].

Toy program for the mapping-program tuner. The spatial loop is m; b and n
are temporal loops without loop-carried values, so they can be interchanged;
k is the single accumulation loop the downstream stack requires.

    python kernels/toy_bmm.py --config kernels/config_files/toy_bmm.json --debug
    python kernels/toy_bmm.py --config kernels/config_files/toy_bmm.json --debug \
        --tune fixed --tune-options "interchange(b,n)"
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args


def _bmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.size()
    b2, k2, n = y.size()
    assert b == b2 and k == k2
    out_ = torch.empty([b, m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile_m in hl.tile(m):
        for tile_b in hl.tile(b, block_size=1):
            for tile_n in hl.tile(n):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
                for tile_k in hl.tile(k):
                    acc = hl.dot(
                        x[tile_b.begin, tile_m, tile_k],
                        y[tile_b.begin, tile_k, tile_n],
                        acc=acc,
                    )
                out_[tile_b.begin, tile_m, tile_n] = acc
    return out_


class ToyBmm(LoomKernel):
    kernel_name = "toy_bmm"

    B: int = 2
    M: int = 256
    K: int = 256
    N: int = 256
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_bmm)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        x = torch.empty([cls.B, cls.M, cls.K], device="cpu", dtype=torch.float16)
        y = torch.empty([cls.B, cls.K, cls.N], device="cpu", dtype=torch.float16)
        return (x, y)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    ToyBmm(shape).run()
