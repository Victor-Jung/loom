"""Back-to-back GEMM, UNFUSED: D = (A@B)@E as two loop nests.

The intermediate C = A@B is materialised to DRAM by the first nest and read
back by the second. This is what a user writes naively, and what ttnn must do
(two matmul ops), so it is the baseline for the fusion-depth experiment.
"""
from __future__ import annotations
import sys
import torch, helion, helion.language as hl
from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args


def _chain_unfused(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    n2, p = z.size()
    dt = torch.promote_types(x.dtype, y.dtype)
    c = torch.empty([m, n], dtype=dt, device=x.device)
    out_ = torch.empty([m, p], dtype=dt, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        c[tile_m, tile_n] = acc
    hl.barrier()
    for tile_m2, tile_p in hl.tile([m, p]):
        acc2 = hl.zeros([tile_m2, tile_p], dtype=torch.float16)
        for tile_n2 in hl.tile(n):
            acc2 = hl.dot(c[tile_m2, tile_n2], z[tile_n2, tile_p], acc=acc2)
        out_[tile_m2, tile_p] = acc2
    return out_


class ChainUnfused(LoomKernel):
    kernel_name = "chain_unfused"
    M: int = 4096
    K: int = 128
    N: int = 4096
    P: int = 128
    assume_divisible: bool = True
    kernel = helion.kernel(static_shapes=False,
        autotune_config_overrides={"range_unroll_factors": [0, 0], "range_num_stages": [0, 0]},
    )(_chain_unfused)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for k_, v in shape.items():
                setattr(cls, k_, v)

    @classmethod
    def bind_args(cls) -> tuple:
        return (torch.empty([cls.M, cls.K], device="cpu", dtype=torch.float16),
                torch.empty([cls.K, cls.N], device="cpu", dtype=torch.float16),
                torch.empty([cls.N, cls.P], device="cpu", dtype=torch.float16))


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    ChainUnfused(shape).run()
