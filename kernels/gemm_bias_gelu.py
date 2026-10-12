"""GEMM with a bias and GELU epilogue: out = gelu(x @ w + bias).

GELU in its sigmoid form, y * sigmoid(1.702 y) (PCC > 0.999 against the exact
GELU on these inputs). Benchmark program against
`ttnn.linear(x, w, bias=bias, activation="gelu")`, which fuses the epilogue.

    python kernels/gemm_bias_gelu.py -M4096_K1024_N4096 --config kernels/config_files/gemm_bias_gelu.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast


def _gemm_bias_gelu(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
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
            y = acc + broadcast(b, 0, [tile_m, tile_n])
            # GELU, sigmoid form: y * sigmoid(1.702 y) = y / (1 + exp(-1.702 y))
            c = hl.full([], -1.702, dtype=torch.float16)
            one = hl.full([], 1.0, dtype=torch.float16)
            out_[tile_m, tile_n] = y / (torch.exp(y * c) + one)
    return out_

class GemmBiasGelu(LoomKernel):
    kernel_name = "gemm_bias_gelu"

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
    )(_gemm_bias_gelu)

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
    GemmBiasGelu(shape).run()
