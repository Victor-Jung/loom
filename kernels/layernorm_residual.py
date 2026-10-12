"""LayerNorm with a fused residual add.
    h = x + res
    out = (h - mean(h)) * rsqrt(var(h) + eps) * w + b
Two row reductions (mean, variance) over the hidden dimension; the [1, N]
weight and bias rows are shared by every row block.
Benchmark program against
`ttnn.layer_norm(x, weight=w, bias=b, residual_input_tensor=res)`.
    python kernels/layernorm_residual.py -M8192_N1024 --config kernels/config_files/layernorm_residual.json --debug
"""
from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast


def _layernorm_residual(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor,
                        b: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    n_s = hl.specialize(n)
    out_ = torch.empty_like(x)
    for tile_m in hl.tile(m):
        h = x[tile_m, :] + res[tile_m, :]
        inv_n = hl.full([], 1.0 / n_s, dtype=torch.float16)
        mean = torch.sum(h, -1, keepdim=True) * inv_n
        d = h - broadcast(mean, 1, [tile_m, n_s])
        d_b = h - broadcast(mean, 1, [tile_m, n_s])  # a second copy: `d * d` does not import
        var = torch.sum(d * d_b, -1, keepdim=True) * inv_n
        eps_dev = hl.full([], 1e-5, dtype=torch.float16)
        inv = torch.rsqrt(var + eps_dev)
        y = d * broadcast(inv, 1, [tile_m, n_s])
        out_[tile_m, :] = y * broadcast(w[:, :], 0, [tile_m, n_s]) + broadcast(b[:, :], 0, [tile_m, n_s])
    return out_


class LayerNormResidual(LoomKernel):
    kernel_name = "layernorm_residual"

    M: int = 8192
    N: int = 1024
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_layernorm_residual)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        x = torch.empty([cls.M, cls.N], dtype=torch.float16)
        res = torch.empty([cls.M, cls.N], dtype=torch.float16)
        w = torch.empty([1, cls.N], dtype=torch.float16)
        b = torch.empty([1, cls.N], dtype=torch.float16)
        return (x, res, w, b)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    LayerNormResidual(shape).run()
