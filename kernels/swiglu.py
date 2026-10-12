"""SwiGLU feed-forward gate: out = silu(x @ wg) * (x @ wu).

Two GEMMs sharing x in one k loop (two accumulators), then the gate in the
epilogue, silu(g) = g / (1 + exp(-g)). Benchmark program against ttnn's
`linear(x, wg, activation="silu")` times `linear(x, wu)`.

    python kernels/swiglu.py -M2048_K1024_N2048 --config kernels/config_files/swiglu.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast


def _swiglu(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = wg.size()
    assert k == k2
    out_ = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            g = hl.zeros([tile_m, tile_n], dtype=torch.float16)
            u = hl.zeros([tile_m, tile_n], dtype=torch.float16)
            for tile_k in hl.tile(k):
                xk = x[tile_m, tile_k]
                g = hl.dot(xk, wg[tile_k, tile_n], acc=g)
                u = hl.dot(xk, wu[tile_k, tile_n], acc=u)
            neg = hl.full([], -1.0, dtype=torch.float16)
            one = hl.full([], 1.0, dtype=torch.float16)
            out_[tile_m, tile_n] = g / (torch.exp(g * neg) + one) * u
    return out_


class SwiGLU(LoomKernel):
    kernel_name = "swiglu"

    M: int = 2048
    K: int = 1024
    N: int = 2048
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_swiglu)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        x = torch.empty([cls.M, cls.K], device="cpu", dtype=torch.float16)
        wg = torch.empty([cls.K, cls.N], device="cpu", dtype=torch.float16)
        wu = torch.empty([cls.K, cls.N], device="cpu", dtype=torch.float16)
        return (x, wg, wu)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    SwiGLU(shape).run()
