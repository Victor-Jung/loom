"""GEMM with an elementwise epilogue: out = exp((x @ w + bias) / 64).

Benchmark program for statement placement: the epilogue (row-broadcast bias
add, exp) follows the accumulation loop in the same nest.

    python kernels/gemm_bias_exp.py --config kernels/config_files/gemm_bias_exp.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast


def _gemm_bias_exp(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = w.size()
    assert k == k2
    out_ = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
            for tile_k in hl.tile(k):
                acc = hl.dot(x[tile_m, tile_k], w[tile_k, tile_n], acc=acc)
            b = bias[:, tile_n]
            scale = hl.full([], 1.0 / 64.0, dtype=torch.float16)
            out_[tile_m, tile_n] = torch.exp((acc + broadcast(b, 0, [tile_m, tile_n])) * scale)
    return out_

class GemmBiasExp(LoomKernel):
    kernel_name = "gemm_bias_exp"

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
    )(_gemm_bias_exp)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        x = torch.empty([cls.M, cls.K], device="cpu", dtype=torch.float16)
        w = torch.empty([cls.K, cls.N], device="cpu", dtype=torch.float16)
        bias = torch.empty([1, cls.N], device="cpu", dtype=torch.float16)
        return (x, w, bias)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    GemmBiasExp(shape).run()
