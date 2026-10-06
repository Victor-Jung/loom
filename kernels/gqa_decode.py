"""Grouped-query attention decode: one query token per sequence against a
long KV cache, the heads of one kv group padded to a 32-row query block.

    q[b, g]: [32, D]  (rows 0..ratio-1 hold the group's query heads)
    K[b, g]: [S, D]   V[b, g]: [S, D]
    out[b, g] = softmax(q K^T / sqrt(D)) V

Benchmark program against `ttnn.transformer.scaled_dot_product_attention_decode`,
which pads the group's heads to a tile the same way. The (batch, group) pairs
are the parallel loop; the KV cache is walked once with an online softmax.

    python kernels/gqa_decode.py -B8_G8_S4096_D128 --config kernels/config_files/gqa_decode.json --debug
"""

from __future__ import annotations

import math
import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args
from helion_mlir.custom_op import broadcast, set_memory_space


def _gqa_decode(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out_ = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        qk_scale_dev = hl.full([], sm_scale, dtype=torch.float16)
        m_i = hl.full([tile_b, tile_m, 1], float("-inf"), dtype=torch.float16)
        l_i = torch.full_like(m_i, 1.0)
        q = q_view[tile_b, tile_m, :]
        # zeros_like keeps the loaded block's width: a specialized non-power-of-two
        # width in hl.zeros is rounded up to the next power of two by Helion.
        acc = torch.zeros_like(q)
        for tile_n in hl.tile(v_view.size(1)):
            k = set_memory_space(k_view[tile_b, :, tile_n], local_mem_kind=1)
            qk = torch.bmm(q, k)
            qk = qk * qk_scale_dev
            m_ij = torch.maximum(m_i, torch.amax(qk, -1, keepdim=True))
            m_ij_broad = broadcast(m_ij, 2, [m_ij.size(0), m_ij.size(1), tile_n])
            qk = qk - m_ij_broad
            p = torch.exp(qk)
            alpha = torch.exp(m_i - m_ij)
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = acc * alpha
            acc = acc + torch.bmm(p, v)
            l_ij = torch.sum(p, -1, keepdim=True)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        m_i += torch.log(l_i)
        l_i_broadcast = broadcast(l_i, 2, [l_i.size(0), l_i.size(1), q.size(2)])
        acc = acc / l_i_broadcast
        out_[tile_b, tile_m, :] = acc.to(out_.dtype)
    return out_.view(q_in.size())


class GQADecode(LoomKernel):
    kernel_name = "gqa_decode"

    B: int = 8      # sequences
    G: int = 8      # kv heads (query groups)
    S: int = 4096   # cached positions
    D: int = 128    # head dim
    ROWS: int = 32  # query heads of a group, padded to one tile
    assume_divisible: bool = True
    tile_upper_bounds = {"tile_b": 1}  # one (batch, group) pair per block: the matmuls are 2-D

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_gqa_decode)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        bg = cls.B * cls.G
        q = torch.empty([bg, cls.ROWS, cls.D], dtype=torch.float16)
        k = torch.empty([bg, cls.S, cls.D], dtype=torch.float16)
        v = torch.empty([bg, cls.S, cls.D], dtype=torch.float16)
        return (q, k, v)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    GQADecode(shape).run()
