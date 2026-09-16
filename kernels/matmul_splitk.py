"""Matmul, VARIANT splitk: K is made partly SPATIAL and reduced across tiles.

The baseline keeps K as a sequential in-core reduction, so at large K each core
streams the whole K extent for one output tile (worst reuse). Here K is tiled
spatially and the partial accumulators are combined with the gather + tile.id==0
idiom that mqa_decode uses for its split-KV reduction (Helion here has no
atomic_add, so that is the supported way to express a cross-tile reduction).

Original header follows.

Matmul kernel for the Loom pipeline.

Standalone CLI script. Run from the repo root:

    python kernels/matmul.py --config kernels/config_files/matmul.json --njobs 16 --debug --topk-candidates 1 --topk-block-size 3

This script inherits the full Loom CLI and pipeline from LoomKernel.
To write your own kernel, copy this file, replace the kernel body
and bind_args tensors, and keep the __main__ block unchanged.
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import gather


def _matmul_splitk(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Matrix-multiply x @ y using helion tiling.

    Kernel dimensions are determined at bind_args() time (M=4096, K=512, N=4096).
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2
    out_ = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile_m, tile_n, tile_s in hl.tile([m, n, k]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
        for tile_k in hl.tile(tile_s.begin, tile_s.end):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        gathered = gather(tile_s, acc)
        if tile_s.id == 0:
            out_[tile_m, tile_n] = torch.sum(gathered, 0)
    return out_


class MatmulSplitK(LoomKernel):
    """Matmul kernel: computes C = A @ B for fixed (M, K, N) shapes.

    Kernel dimensions (class-level constants, can be overridden in subclasses):
        M=4096, K=512, N=4096
    """

    kernel_name = "matmul_splitk"

    M: int = 4096
    K: int = 256
    N: int = 4096
    assume_divisible: bool = True

    # Assign the helion-decorated function as a class attribute.
    # We cannot stack @staticmethod with @helion.kernel because the helion
    # decorator returns a custom object, not a plain callable.
    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_matmul_splitk)

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
    kernel = MatmulSplitK(shape)
    kernel.run()
