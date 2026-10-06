"""RMSNorm with a fused residual add.

    h = x + res
    out = h * rsqrt(mean(h^2) + eps) * w

Row reduction over the hidden dimension; the [1, N] weight row is shared by
every row block.
Benchmark program against `ttnn.rms_norm(x, weight=w, residual_input_tensor=res)`.

    python kernels/rmsnorm_residual.py -M8192_N4096 --config kernels/config_files/rmsnorm_residual.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast


def _rmsnorm_residual(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    n_s = hl.specialize(n)
    out_ = torch.empty_like(x)
    for tile_m in hl.tile(m):
        h = x[tile_m, :] + res[tile_m, :]
        h_b = res[tile_m, :] + x[tile_m, :]  # a second copy: `h * h` does not import, pow lowers to fpowi
        inv_n = hl.full([], 1.0 / n_s, dtype=torch.float16)
        ms = torch.sum(h * h_b, -1, keepdim=True) * inv_n
        eps_dev = hl.full([], 1e-5, dtype=torch.float16)
        inv = torch.rsqrt(ms + eps_dev)
        y = h * broadcast(inv, 1, [tile_m, n_s])
        out_[tile_m, :] = y * broadcast(w[:, :], 0, [tile_m, n_s])
    return out_


class RMSNormResidual(LoomKernel):
    kernel_name = "rmsnorm_residual"

    M: int = 8192
    N: int = 4096
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_rmsnorm_residual)

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
        return (x, res, w)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    RMSNormResidual(shape).run()
