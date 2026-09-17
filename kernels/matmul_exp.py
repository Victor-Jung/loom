"""Matmul with a fused exp epilogue: out = exp(A@B).

One loop-carried reduction, epilogue applied to the accumulator before the
store, so C = A@B never reaches DRAM. ttnn cannot fuse exp into matmul (only
relu/gelu/silu), so the same math there costs two kernels plus a DRAM
round-trip of C; at small K and large M,N that traffic dominates.

CLI as kernels/matmul.py.
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args


def _matmul_exp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out_ = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out_[tile_m, tile_n] = torch.exp(acc)
    return out_


class MatmulExp(LoomKernel):
    kernel_name = "matmul_exp"

    M: int = 4096
    K: int = 256
    N: int = 4096
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_matmul_exp)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        """Return concrete input tensors that define M, K, N at MLIR-gen time."""
        x = torch.empty([cls.M, cls.K], device="cpu", dtype=torch.float16)
        y = torch.empty([cls.K, cls.N], device="cpu", dtype=torch.float16)
        return (x, y)


if __name__ == "__main__":
    try:
        shape, normalized_argv = resolve_kernel_shape_args(sys.argv)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    sys.argv = normalized_argv
    kernel = MatmulExp(shape)
    kernel.run()
