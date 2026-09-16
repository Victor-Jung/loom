"""Minimal reproducer: read a rank-2 vector slice [tile_m, 1] and broadcast it.

This is the smallest kernel that exercises the subview -> memref.cast ->
to_tensor path which previously produced rank-1 L1 allocations and unbound
loads. Used to generate regression fixtures for loom-dataflow.
"""
from __future__ import annotations
import sys
import torch, helion, helion.language as hl
from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast


def _vecslice(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    out_ = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        # Loom requires exactly one loop-carried scf.for, so keep a reduction.
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        # [tile_m, 1] slice: rank 2 with a unit minor dim -- the pattern under test
        s = v[tile_m, :]
        s = broadcast(s, 1, [s.size(0), tile_n])
        out_[tile_m, tile_n] = acc * s
    return out_


class RegressVecSlice(LoomKernel):
    kernel_name = "regress_vecslice"
    M: int = 256
    N: int = 256
    K: int = 256
    assume_divisible: bool = True
    kernel = helion.kernel(static_shapes=False,
        autotune_config_overrides={"range_unroll_factors": [0, 0], "range_num_stages": [0, 0]},
    )(_vecslice)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for k_, v_ in shape.items():
                setattr(cls, k_, v_)

    @classmethod
    def bind_args(cls) -> tuple:
        return (torch.empty([cls.M, cls.K], device="cpu", dtype=torch.float16),
                torch.empty([cls.K, cls.N], device="cpu", dtype=torch.float16),
                torch.empty([cls.M, 1], device="cpu", dtype=torch.float16))


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    RegressVecSlice(shape).run()
