"""Back-to-back GEMM, FUSED: D = (A@B)@E in ONE loop nest.

Identical math to chain_unfused, but the C = A@B tile is consumed by the second
dot immediately, so it never reaches DRAM. Only the statement placement and
loop nesting differ - the fusion depth is the whole change.
"""
from __future__ import annotations
import sys
import torch, helion, helion.language as hl
from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args


def _chain_fused(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    n2, p = z.size()
    dt = torch.promote_types(x.dtype, y.dtype)
    p_s = hl.specialize(p)
    out_ = torch.empty([m, p], dtype=dt, device=x.device)
    for tile_m in hl.tile(m):
        acc2 = hl.zeros([tile_m, p_s], dtype=torch.float16)
        for tile_n in hl.tile(n):
            acc1 = hl.zeros([tile_m, tile_n], dtype=torch.float16)
            for tile_k in hl.tile(k):
                acc1 = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc1)
            acc2 = hl.dot(acc1, z[tile_n, :], acc=acc2)
        out_[tile_m, :] = acc2
    return out_


class ChainFused(LoomKernel):
    kernel_name = "chain_fused"
    M: int = 4096
    K: int = 128
    N: int = 4096
    P: int = 128
    assume_divisible: bool = True
    kernel = helion.kernel(static_shapes=False,
        autotune_config_overrides={"range_unroll_factors": [0, 0], "range_num_stages": [0, 0]},
    )(_chain_fused)

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
    ChainFused(shape).run()
