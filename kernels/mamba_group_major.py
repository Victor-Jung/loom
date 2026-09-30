"""Mamba chunk-scan, GROUP-MAJOR head loop.

Same math as mamba_chunk_scan, with the head loop split into group
(sequential) x head-in-group (spatial). cb and C are indexed by the group, not
the head, so making head-in-group spatial leaves them invariant along a mesh
axis, where the mapper can multicast them instead of re-reading them once per
head. The baseline refetches both 8x (nheads/ngroups): cb 5.9 -> 94 MB and
C 3.9 -> 63 MB, out of 535 MB total off-chip traffic.

The head dimension is split in the LAYOUT rather than with arithmetic: an index
of the form `tile_g.begin * 8 + tile_hh.begin` combines two tile symnodes and
the frontend cannot resolve it ("Cannot resolve symbol 8*u16 + u17"). Passing
tensors already shaped [..., ngroups, heads_per_group, ...] keeps every index a
plain `.begin`. Tensors also arrive pre-transposed, so the kernel body does no
layout work.

CLI as kernels/mamba_chunk_scan.py.
"""

from __future__ import annotations

import sys

import torch
import helion
import helion.language as hl

from loom import LoomKernel
from loom.loom_utils.kernel_size import resolve_kernel_shape_args

from helion_mlir.custom_op import broadcast


def _mamba_group_major(
    cb: torch.Tensor,
    x: torch.Tensor,
    dt_k: torch.Tensor,
    dA_cumsum_m: torch.Tensor,
    dA_cumsum_k: torch.Tensor,
    C: torch.Tensor,
    prev_states_T: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Argument (head dim pre-split into (ngroups, heads_per_group); pre-transposed):
        cb:            (batch, nchunks, ngroups, chunk_size, chunk_size)
        x:             (batch, ngroups, hpg, seqlen, headdim)
        dt_k:          (batch, ngroups, hpg, nchunks, 1, chunk_size)
        dA_cumsum_m:   (batch, ngroups, hpg, nchunks, chunk_size, 1)
        dA_cumsum_k:   (batch, ngroups, hpg, nchunks, 1, chunk_size)
        C:             (batch, ngroups, seqlen, dstate)
        prev_states_T: (batch, nchunks, ngroups, hpg, dstate, headdim)
        D:             (batch, ngroups, hpg, chunk_size, headdim)  -- pre-replicated
    Return:
        out: (batch, ngroups, hpg, seqlen, headdim)
    """

    batch, nchunks, ngroups, chunk_size, _ = cb.shape
    _, _, hpg, seqlen, headdim = x.shape
    _, _, _, dstate = C.shape
    assert nchunks == (seqlen + chunk_size - 1) // chunk_size

    block_m = hl.register_block_size(chunk_size)
    block_n = hl.register_block_size(headdim)
    block_k = hl.register_block_size(64, 64)

    assert x.shape == (batch, ngroups, hpg, seqlen, headdim)
    # L1 allocations must be rank >= 2 and there is no transpose kernel, so the
    # per-chunk vectors arrive as two views: [chunk_size, 1] and [1, chunk_size].
    assert dt_k.shape == (batch, ngroups, hpg, nchunks, 1, chunk_size)
    assert dA_cumsum_m.shape == (batch, ngroups, hpg, nchunks, chunk_size, 1)
    assert dA_cumsum_k.shape == (batch, ngroups, hpg, nchunks, 1, chunk_size)
    assert C.shape == (batch, ngroups, seqlen, dstate)
    assert prev_states_T.shape == (batch, nchunks, ngroups, hpg, dstate, headdim)
    assert D.shape == (batch, ngroups, hpg, chunk_size, headdim)

    dtype = cb.dtype
    accum_dtype = torch.float16

    out_ = torch.empty_like(x)

    # tile_hh (head within group) is spatial; cb and C do not depend on it.
    for tile_m, tile_n, tile_c, tile_hh in hl.tile(
        [chunk_size, headdim, nchunks, hpg],
        block_size=[block_m, block_n, 1, 1],
    ):
        # block_size=1 is required: indexed by `.begin`, so the slices have
        # extent 1 while the loop strides by the block size.
        for tile_b in hl.tile(batch, block_size=1):
            for tile_g in hl.tile(ngroups, block_size=1):
                acc_o = hl.zeros([tile_m, tile_n], dtype=accum_dtype)
                # dA_cumsum_local_m: [tile_m, 1]  (rank 2 for the L1 estimator)
                dA_cumsum_local_m = dA_cumsum_m[
                    tile_b.begin, tile_g.begin, tile_hh.begin, tile_c.begin, tile_m, :
                ]
                dA_cumsum_local_m_bc_n = broadcast(
                    dA_cumsum_local_m, 1, [dA_cumsum_local_m.size(0), tile_n]
                )
                scale_m_local = torch.exp(dA_cumsum_local_m_bc_n)

                # C_local: [tile_m, dstate] -- group-indexed, invariant in tile_hh
                C_local = C[
                    tile_b.begin,
                    tile_g.begin,
                    tile_m.index + tile_c.begin * chunk_size,
                    :,
                ]
                prev_states_local = prev_states_T[
                    tile_b.begin, tile_c.begin, tile_g.begin, tile_hh.begin, :, tile_n
                ]
                acc_o = hl.dot(C_local, prev_states_local, acc=acc_o)
                acc_o *= scale_m_local

                # Static bound: the original (tile_m.id + 1) * block_m is a trip
                # count Loom cannot trace. Makes the intra-chunk term non-causal.
                # FUSED residual, added BEFORE the k-loop: addition commutes
                # with the loop's accumulation so the value is identical, but
                # the add is not the final op, so its destination cannot be
                # coalesced with the output CB (which made compute a second
                # consumer of the writer's buffer). D is pre-replicated to the
                # full tile shape because a [1, tile_n] slice + row-broadcast
                # populates only row 0 of each tile on device.
                D_local = D[
                    tile_b.begin, tile_g.begin, tile_hh.begin, tile_m, tile_n
                ]
                x_residual = x[
                    tile_b.begin,
                    tile_g.begin,
                    tile_hh.begin,
                    tile_c.begin * chunk_size + tile_m.index,
                    tile_n,
                ]
                acc_o += x_residual * D_local

                for tile_k in hl.tile(chunk_size, block_size=block_k):
                    # cb_local: [tile_m, tile_k] -- invariant in tile_hh
                    cb_local = cb[
                        tile_b.begin, tile_c.begin, tile_g.begin, tile_m, tile_k
                    ]
                    dA_cumsum_local_k = dA_cumsum_k[
                        tile_b.begin, tile_g.begin, tile_hh.begin, tile_c.begin, :, tile_k
                    ]
                    dA_cumsum_local_m_bc_k = broadcast(
                        dA_cumsum_local_m, 1, [dA_cumsum_local_m.size(0), tile_k]
                    )
                    dA_cumsum_local_k = broadcast(
                        dA_cumsum_local_k, 0, [tile_m, dA_cumsum_local_k.size(1)]
                    )
                    cb_local *= torch.exp(dA_cumsum_local_m_bc_k - dA_cumsum_local_k)
                    # dt_local: [1, tile_k], broadcast over the tile_m axis
                    dt_local = dt_k[
                        tile_b.begin, tile_g.begin, tile_hh.begin, tile_c.begin, :, tile_k
                    ]
                    dt_local = broadcast(dt_local, 0, [tile_m, dt_local.size(1)])
                    cb_local *= dt_local
                    x_local = x[
                        tile_b.begin,
                        tile_g.begin,
                        tile_hh.begin,
                        tile_c.begin * chunk_size + tile_k.index,
                        tile_n,
                    ]
                    acc_o = torch.addmm(acc_o, cb_local, x_local)

                out_[
                    tile_b.begin,
                    tile_g.begin,
                    tile_hh.begin,
                    tile_c.begin * chunk_size + tile_m.index,
                    tile_n,
                ] = acc_o.to(dtype=dtype)

    return out_


class MambaGroupMajor(LoomKernel):

    kernel_name = "mamba_group_major"

    BATCH: int = 2
    SEQLEN: int = 1920
    NHEADS: int = 32
    HEADDIM: int = 128
    NGROUPS: int = 4
    DSTATE: int = 128
    CHUNK_SIZE: int = 192

    assume_divisible: bool = True

    kernel = helion.kernel(static_shapes=False)(_mamba_group_major)

    def __init__(self, shape: dict[str, int] | None = None) -> None:
        if shape:
            cls = type(self)
            key_to_attr = {"B": "BATCH", "L": "SEQLEN", "N": "NHEADS", "H": "HEADDIM",
                           "G": "NGROUPS", "D": "DSTATE", "C": "CHUNK_SIZE"}
            for key, value in shape.items():
                setattr(cls, key_to_attr.get(key, key), value)

    @classmethod
    def bind_args(cls) -> tuple:
        """Concrete input tensors defining kernel shapes at MLIR-gen time."""
        nc = (cls.SEQLEN + cls.CHUNK_SIZE - 1) // cls.CHUNK_SIZE
        g, hpg = cls.NGROUPS, cls.NHEADS // cls.NGROUPS
        f16 = torch.float16
        e = lambda *s: torch.empty(list(s), dtype=f16)
        return (
            e(cls.BATCH, nc, g, cls.CHUNK_SIZE, cls.CHUNK_SIZE),          # cb
            e(cls.BATCH, g, hpg, cls.SEQLEN, cls.HEADDIM),                # x
            e(cls.BATCH, g, hpg, nc, 1, cls.CHUNK_SIZE),                  # dt_k
            e(cls.BATCH, g, hpg, nc, cls.CHUNK_SIZE, 1),                  # dA_cumsum_m
            e(cls.BATCH, g, hpg, nc, 1, cls.CHUNK_SIZE),                  # dA_cumsum_k
            e(cls.BATCH, g, cls.SEQLEN, cls.DSTATE),                      # C
            e(cls.BATCH, nc, g, hpg, cls.DSTATE, cls.HEADDIM),            # prev_states_T
            e(cls.BATCH, g, hpg, cls.CHUNK_SIZE, cls.HEADDIM),            # D
        )


if __name__ == "__main__":
    try:
        shape, normalized_argv = resolve_kernel_shape_args(sys.argv)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    sys.argv = normalized_argv
    kernel = MambaGroupMajor(shape)
    kernel.run()
