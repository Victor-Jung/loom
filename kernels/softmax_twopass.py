"""Row softmax over tiled rows in three passes: running max, sum of exp, normalise.

Benchmark program with two sequential loop-carried loops at the same level
followed by a loop without carried values.

    python kernels/softmax_twopass.py --config kernels/config_files/softmax_twopass.json --debug
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast


def _softmax_twopass(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out_ = torch.empty_like(x)
    for tile_m in hl.tile(m):
        mx = hl.full([tile_m, 1], float("-inf"), dtype=torch.float16)
        for tile_n in hl.tile(n):
            mx = torch.maximum(mx, torch.amax(x[tile_m, tile_n], -1, keepdim=True))
        s = hl.zeros([tile_m, 1], dtype=torch.float16)
        for tile_n in hl.tile(n):
            p = torch.exp(x[tile_m, tile_n] - broadcast(mx, 1, [tile_m, tile_n]))
            s = s + torch.sum(p, -1, keepdim=True)
        for tile_n in hl.tile(n):
            p = torch.exp(x[tile_m, tile_n] - broadcast(mx, 1, [tile_m, tile_n]))
            out_[tile_m, tile_n] = p / broadcast(s, 1, [tile_m, tile_n])
    return out_

class SoftmaxTwoPass(LoomKernel):
    kernel_name = "softmax_twopass"

    M: int = 256
    N: int = 512
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_softmax_twopass)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        return (torch.empty([cls.M, cls.N], device="cpu", dtype=torch.float16),)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    SoftmaxTwoPass(shape).run()
