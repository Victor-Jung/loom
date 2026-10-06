"""Fused scale, mask and softmax over attention scores.

    out[b, h, i, :] = softmax(x[b, h, i, :] / sqrt(N) + mask[i, :])

The mask is indexed by query row and key only (a causal or padding mask), so
it is shared by every batch and head of a row block. Benchmark program against
`ttnn.scale_mask_softmax_in_place(..., is_causal_mask=True)`, which takes the
same [1, 1, S, N] mask.

    python kernels/mask_softmax.py -B2_H32_S1024_N1024 --config kernels/config_files/mask_softmax.json --debug
"""

from __future__ import annotations

import math
import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast


def _mask_softmax(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch, heads, rows, n = x.size()
    n_s = hl.specialize(n)
    scale = 1.0 / math.sqrt(n_s)  # ttnn head_size = N: the only scale the frontend keeps constant
    out_ = torch.empty_like(x)
    for tile_m in hl.tile(rows):
        for tile_b in hl.tile(batch, block_size=1):
            for tile_h in hl.tile(heads, block_size=1):
                scale_dev = hl.full([], scale, dtype=torch.float16)
                row = x[tile_b.begin, tile_h.begin, tile_m, :]
                msk = mask[tile_m, :]
                s = row * scale_dev + msk
                mx = torch.amax(s, -1, keepdim=True)
                p = torch.exp(s - broadcast(mx, 1, [mx.size(0), n_s]))
                l = torch.sum(p, -1, keepdim=True)
                out_[tile_b.begin, tile_h.begin, tile_m, :] = p / broadcast(l, 1, [l.size(0), n_s])
    return out_


class MaskSoftmax(LoomKernel):
    kernel_name = "mask_softmax"

    B: int = 2
    H: int = 32
    S: int = 1024
    N: int = 1024
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_mask_softmax)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        x = torch.empty([cls.B, cls.H, cls.S, cls.N], dtype=torch.float16)
        mask = torch.empty([cls.S, cls.N], dtype=torch.float16)
        return (x, mask)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    MaskSoftmax(shape).run()
