"""Matmul with a transposed left operand: out = x.T @ y.

Benchmark program for the transpose primitive inside the accumulation loop.

    python kernels/transpose_matmul.py --config kernels/config_files/transpose_matmul.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args


def _transpose_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    k, m = x.size()
    k2, n = y.size()
    assert k == k2
    out_ = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
            for tile_k in hl.tile(k):
                acc = hl.dot(x[tile_k, tile_m].T, y[tile_k, tile_n], acc=acc)
            out_[tile_m, tile_n] = acc
    return out_

class TransposeMatmul(LoomKernel):
    kernel_name = "transpose_matmul"

    K: int = 256
    M: int = 256
    N: int = 256
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_transpose_matmul)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        x = torch.empty([cls.K, cls.M], device="cpu", dtype=torch.float16)
        y = torch.empty([cls.K, cls.N], device="cpu", dtype=torch.float16)
        return (x, y)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    TransposeMatmul(shape).run()
