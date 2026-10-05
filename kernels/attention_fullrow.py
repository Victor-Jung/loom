"""Attention with a full-row softmax in one loop nest: out = softmax(q @ k.T) @ v.

Benchmark program chaining two matmuls through a row softmax inside one
nest: the score tile covers the whole key dimension, so the softmax needs no
loop-carried state and the second matmul consumes the normalised tile.

    python kernels/attention_fullrow.py --config kernels/config_files/attention_fullrow.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast


def _attention_fullrow(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    m, d = q.size()
    n, d2 = k.size()
    assert d == d2
    n_s = hl.specialize(n)
    d_s = hl.specialize(d)
    out_ = torch.empty([m, d], dtype=q.dtype, device=q.device)
    for tile_m in hl.tile(m):
        s = torch.matmul(q[tile_m, :], k[:, :].T)
        mx = torch.amax(s, -1, keepdim=True)
        p = torch.exp(s - broadcast(mx, 1, [tile_m, n_s]))
        l = torch.sum(p, -1, keepdim=True)
        p = p / broadcast(l, 1, [tile_m, n_s])
        out_[tile_m, :] = torch.matmul(p, v[:, :])
    return out_


class AttentionFullRow(LoomKernel):
    kernel_name = "attention_fullrow"

    M: int = 512
    N: int = 256
    D: int = 64
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_attention_fullrow)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        q = torch.empty([cls.M, cls.D], device="cpu", dtype=torch.float16)
        k = torch.empty([cls.N, cls.D], device="cpu", dtype=torch.float16)
        v = torch.empty([cls.N, cls.D], device="cpu", dtype=torch.float16)
        return (q, k, v)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    AttentionFullRow(shape).run()
