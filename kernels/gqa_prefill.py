"""Grouped-query attention prefill, non-causal.

    q[b*Hq + h]: [S, D]     k[b*G + g], v[b*G + g]: [S, D]     g = h // (Hq/G)
    out[b*Hq + h] = softmax(q k^T / sqrt(D)) v

K and V are indexed by the kv group, so the RATIO query heads of a group read
the same cache: a reuse the tuner can turn into a multicast across head
cores. Benchmark program against
`ttnn.transformer.scaled_dot_product_attention(is_causal=False)`.

    python kernels/gqa_prefill.py -BH32_BG8_S2048_D128 --config kernels/config_files/gqa_prefill.json --debug
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

RATIO = 4  # query heads per kv head


def _gqa_prefill(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
    bh, seq, d = q_in.size()
    head_dim = hl.specialize(d)
    k_t = k_in.transpose(1, 2)
    out_ = torch.empty_like(q_in)
    sm_scale = 1.0 / math.sqrt(head_dim)
    for tile_m in hl.tile(seq):
        for tile_h in hl.tile(bh, block_size=1):
            qk_scale_dev = hl.full([], sm_scale, dtype=torch.float16)
            m_i = hl.full([tile_m, 1], float("-inf"), dtype=torch.float16)
            l_i = hl.full([tile_m, 1], 1.0, dtype=torch.float16)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float16)
            q = q_in[tile_h.begin, tile_m, :]
            for tile_n in hl.tile(seq):
                k = k_t[tile_h.begin // RATIO, :, tile_n]
                qk = torch.matmul(q, k)
                qk = qk * qk_scale_dev
                m_ij = torch.maximum(m_i, torch.amax(qk, -1, keepdim=True))
                qk = qk - broadcast(m_ij, 1, [m_ij.size(0), tile_n])
                p = torch.exp(qk)
                alpha = torch.exp(m_i - m_ij)
                v = v_in[tile_h.begin // RATIO, tile_n, :]
                acc = acc * broadcast(alpha, 1, [alpha.size(0), head_dim])
                acc = acc + torch.matmul(p, v)
                l_i = l_i * alpha + torch.sum(p, -1, keepdim=True)
                m_i = m_ij
            out_[tile_h.begin, tile_m, :] = acc / broadcast(l_i, 1, [l_i.size(0), head_dim])
    return out_


class GQAPrefill(LoomKernel):
    kernel_name = "gqa_prefill"

    BH: int = 32    # batch x query heads
    BG: int = 8     # batch x kv heads (BH / BG = RATIO)
    S: int = 2048
    D: int = 128
    assume_divisible: bool = True

    kernel = helion.kernel(
        static_shapes=False,
        autotune_config_overrides={
            "range_unroll_factors": [0, 0],
            "range_num_stages": [0, 0],
        },
    )(_gqa_prefill)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            for key, value in shape.items():
                setattr(cls, key, value)

    @classmethod
    def bind_args(cls) -> tuple:
        q = torch.empty([cls.BH, cls.S, cls.D], dtype=torch.float16)
        k = torch.empty([cls.BG, cls.S, cls.D], dtype=torch.float16)
        v = torch.empty([cls.BG, cls.S, cls.D], dtype=torch.float16)
        return (q, k, v)


if __name__ == "__main__":
    shape, argv = resolve_kernel_shape_args(sys.argv)
    sys.argv = argv
    GQAPrefill(shape).run()
