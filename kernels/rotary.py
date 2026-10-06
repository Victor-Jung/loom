"""Rotary position embedding over all heads of a prefill.

    out[h, t, :] = x[h, t, :] * cos[t, :] + (x[h, t, :] @ rot) * sin[t, :]

`rot` is the [D, D] pair-rotation matrix (the same trick ttnn uses with a
32x32 transformation matrix per tile), so the rotation is a matmul by a
constant and the cos/sin rows are shared by every head. Benchmark program
against `ttnn.experimental.rotary_embedding_llama`.

    python kernels/rotary.py -H32_T4096_D128 --config kernels/config_files/rotary.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args


def _rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    heads, tokens, d = x.size()
    out_ = torch.empty_like(x)
    for tile_t in hl.tile(tokens):
        for tile_h in hl.tile(heads, block_size=1):
            xb = x[tile_h.begin, tile_t, :]
            c = cos[tile_t, :]
            s = sin[tile_t, :]
            r = torch.matmul(xb, rot[:, :])
            out_[tile_h.begin, tile_t, :] = xb * c + r * s
    return out_


def pair_rotation_matrix(d: int) -> torch.Tensor:
    """x @ M = (-x1, x0, -x3, x2, ...): the interleaved rotate-half."""
    m = torch.zeros(d, d)
    for i in range(0, d, 2):
        m[i, i + 1] = 1.0
        m[i + 1, i] = -1.0
    return m


class Rotary(LoomKernel):
    kernel_name = "rotary"

    H: int = 32
    T: int = 4096
    D: int = 128
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_rotary)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        x = torch.empty([cls.H, cls.T, cls.D], dtype=torch.float16)
        cos = torch.empty([cls.T, cls.D], dtype=torch.float16)
        sin = torch.empty([cls.T, cls.D], dtype=torch.float16)
        rot = torch.empty([cls.D, cls.D], dtype=torch.float16)
        return (x, cos, sin, rot)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    Rotary(shape).run()
