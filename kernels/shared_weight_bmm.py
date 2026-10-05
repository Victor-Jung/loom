"""Batched matmul with one shared weight: out[b] = x[b] @ w.

Benchmark program for loop-order search. w[k, n] does not depend on b, so
the order of the temporal loops b and n decides how often the weight tile is
re-fetched: b innermost reloads w on every step, b outermost loads it once
per (n, k) tile.

    python kernels/shared_weight_bmm.py --config kernels/config_files/shared_weight_bmm.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args


def _shared_weight_bmm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    b, m, k = x.size()
    k2, n = w.size()
    assert k == k2
    out_ = torch.empty([b, m, n], dtype=torch.promote_types(x.dtype, w.dtype), device=x.device)
    for tile_m in hl.tile(m):
        for tile_b in hl.tile(b, block_size=1):
            for tile_n in hl.tile(n):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
                for tile_k in hl.tile(k):
                    acc = hl.dot(x[tile_b.begin, tile_m, tile_k], w[tile_k, tile_n], acc=acc)
                out_[tile_b.begin, tile_m, tile_n] = acc
    return out_


class SharedWeightBmm(LoomKernel):
    kernel_name = "shared_weight_bmm"

    B: int = 4
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
    )(_shared_weight_bmm)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        x = torch.empty([cls.B, cls.M, cls.K], device="cpu", dtype=torch.float16)
        w = torch.empty([cls.K, cls.N], device="cpu", dtype=torch.float16)
        return (x, w)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    SharedWeightBmm(shape).run()
