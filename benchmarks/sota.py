"""Compare a benchmark program's best Loom kernel with the state of the art.

For every program the yardsticks are:

  ttnn fused     the hand-written fused ttnn operator for this computation
  ttnn composite the same computation from standard ttnn ops, what a user or
                 an op-by-op compiler gets without kernel work
  loom           the best passing candidate of each tuning policy already run
                 by benchmarks/run.py (test/bench/<program>/<tag>/results.json)
  roofline       bytes that must cross DRAM once, over the p150a's 512 GB/s

Baselines run on the same random inputs as the Loom kernel (seed 0) and are
checked against the program's golden reference, each in its own process.

Both sides run the same PE kernel (per-core compute settings): every ttnn op
that takes a compute_kernel_config gets one built from --fidelity and
--no-approx, with fp32 destination accumulation off and packer L1
accumulation on, as in the generated Loom kernels; the Loom rows listed are
the ones run with the same settings (benchmarks/run.py --fidelity). The
fused ttnn operators are swept over their program configs (chunk sizes) and
every point is reported, so ttnn is not judged at an untuned default.

    python benchmarks/sota.py gqa_decode --param groups=8 --fidelity HiFi4
    python benchmarks/sota.py mask_softmax --fidelity HiFi2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.programs import PROGRAMS  # noqa: E402
from benchmarks.run import TT_METAL_HOME, arg_shapes, reset_board  # noqa: E402,F401

DRAM_BYTES_PER_S = 512e9  # Blackhole p150a GDDR6 peak, from the product spec


# --------------------------------------------------------------------------
# Baselines: name -> fn(ttnn, device, inputs by arg index (torch, float32),
#                       shapes, params) -> (output torch tensor, callable to time)
# --------------------------------------------------------------------------

def _to_dev(ttnn, d, t):
    import torch
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _ckc(ttnn, d, params):
    """The compute kernel config both sides run with (see the module docstring)."""
    return ttnn.init_device_compute_kernel_config(
        d.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, params["fidelity"]),
        math_approx_mode=params["approx"],
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _sdpa_cfg(ttnn, d, params):
    """Program config of the SDPA family from the sweep point, or None for ttnn's default."""
    if "q_chunk" not in params and "k_chunk" not in params:
        return None
    grid = d.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                                  q_chunk_size=int(params.get("q_chunk", 32)),
                                  k_chunk_size=int(params.get("k_chunk", 32)),
                                  exp_approx_mode=params["approx"])


def gqa_decode_fused(ttnn, d, ins, shapes, params):
    """ttnn.transformer.scaled_dot_product_attention_decode on the real heads.

    Loom's layout pads each kv group's heads to 32 query rows; ttnn takes the
    real heads [1, b, nh, dh] and pads to a tile internally. The output is
    compared on the real rows only."""
    import torch
    kt, v, q_pad = ins
    bg, rows, dh = q_pad.shape
    groups = int(params.get("groups", 8))
    ratio = int(params.get("ratio", 4))
    b = bg // groups
    q_real = q_pad.view(b, groups, rows, dh)[:, :, :ratio, :].reshape(b, groups * ratio, dh)
    q_tt = _to_dev(ttnn, d, q_real.unsqueeze(0))                    # [1, b, nh, dh]
    k_tt = _to_dev(ttnn, d, kt.transpose(1, 2).reshape(b, groups, -1, dh))  # [b, nkv, s, dh]
    v_tt = _to_dev(ttnn, d, v.reshape(b, groups, -1, dh))
    s = k_tt.shape[2]
    cur_pos = [s - 1] * b

    cfg, ckc = _sdpa_cfg(ttnn, d, params), _ckc(ttnn, d, params)

    def run():
        return ttnn.transformer.scaled_dot_product_attention_decode(q_tt, k_tt, v_tt, is_causal=True, cur_pos=cur_pos,
                                                                     program_config=cfg, compute_kernel_config=ckc)

    out = ttnn.to_torch(run()).float()[0]                             # [b, nh, dh]
    full = torch.zeros(b, groups, rows, dh)
    full[:, :, :ratio, :] = out.view(b, groups, ratio, dh)
    return full.view(bg, rows, dh), run, lambda ref: _pcc(out.reshape(-1), ref.view(b, groups, rows, dh)[:, :, :ratio, :].reshape(-1))


def gqa_decode_composite(ttnn, d, ins, shapes, params):
    kt, v, q_pad = ins
    q_tt, kt_tt, v_tt = (_to_dev(ttnn, d, t) for t in (q_pad, kt, v))
    scale = 1.0 / math.sqrt(q_pad.shape[-1])

    ckc = _ckc(ttnn, d, params)

    def run():
        s = ttnn.matmul(q_tt, kt_tt, compute_kernel_config=ckc)
        s = ttnn.multiply(s, scale)
        p = ttnn.softmax(s, dim=-1, compute_kernel_config=ckc, numeric_stable=True)
        return ttnn.matmul(p, v_tt, compute_kernel_config=ckc)

    return ttnn.to_torch(run()).float(), run, None


def mask_softmax_fused(ttnn, d, ins, shapes, params):
    """ttnn.scale_mask_softmax_in_place with a causal-style [1, 1, S, N] mask shared by batch and heads."""
    x, mask = ins
    x_tt = _to_dev(ttnn, d, x)
    # ttnn wants the mask's batch dim to match the input's: [B, 1, S, N].
    m_tt = _to_dev(ttnn, d, mask.reshape(1, 1, *mask.shape).expand(x.shape[0], 1, *mask.shape).contiguous())
    scale = 1.0 / math.sqrt(x.shape[-1])

    ckc = _ckc(ttnn, d, params)

    def run():
        return ttnn.scale_mask_softmax_in_place(x_tt, scale, m_tt, is_causal_mask=True, compute_kernel_config=ckc,
                                                numeric_stable=True)

    out = ttnn.to_torch(run()).float()
    # In place: the timed runs re-normalize an already normalized tensor, which
    # moves the same bytes, so the timing stands; the PCC uses the first call.
    return out, run, None


def mask_softmax_composite(ttnn, d, ins, shapes, params):
    x, mask = ins
    x_tt = _to_dev(ttnn, d, x)
    m_tt = _to_dev(ttnn, d, mask.reshape(1, 1, *mask.shape))
    scale = 1.0 / math.sqrt(x.shape[-1])

    ckc = _ckc(ttnn, d, params)

    def run():
        s = ttnn.multiply(x_tt, scale)
        s = ttnn.add(s, m_tt)
        return ttnn.softmax(s, dim=-1, compute_kernel_config=ckc, numeric_stable=True)

    return ttnn.to_torch(run()).float(), run, None


def _rmsnorm_parts(ins):
    import torch
    w = next(t for t in ins if t.shape[0] == 1)
    x, res = [t for t in ins if t.shape[0] != 1]
    h = x.float() + res.float()
    ref = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) * w.float()
    return x, res, w, ref


def rmsnorm_fused(ttnn, d, ins, shapes, params):
    x, res, w, ref = _rmsnorm_parts(ins)
    x_tt, r_tt, w_tt = (_to_dev(ttnn, d, t) for t in (x, res, w))

    ckc = _ckc(ttnn, d, params)

    def run():
        return ttnn.rms_norm(x_tt, epsilon=1e-5, weight=w_tt, residual_input_tensor=r_tt, compute_kernel_config=ckc)

    out = ttnn.to_torch(run()).float()
    return out, run, lambda _golden: _pcc(out, ref)


def rmsnorm_composite(ttnn, d, ins, shapes, params):
    x, res, w, ref = _rmsnorm_parts(ins)
    x_tt, r_tt, w_tt = (_to_dev(ttnn, d, t) for t in (x, res, w))

    ckc = _ckc(ttnn, d, params)

    def run():
        h = ttnn.add(x_tt, r_tt)
        ms = ttnn.mean(ttnn.multiply(h, h), dim=-1, keepdim=True)
        inv = ttnn.rsqrt(ttnn.add(ms, 1e-5))
        return ttnn.multiply(ttnn.multiply(h, inv), w_tt)

    out = ttnn.to_torch(run()).float()
    return out, run, lambda _golden: _pcc(out, ref)


def mla_decode_fused(ttnn, d, ins, shapes, params):
    """ttnn.transformer.flash_multi_latent_attention_decode: q [1, b, nh, d],
    latent kv [b, 1, s, d], head_dim_v columns kept. Compared on those columns."""
    import torch
    q, kv, _kvt = _mla_parts(ins)
    b, nh, dh = q.shape
    dv = int(params.get("head_dim_v", 512))
    q_tt = _to_dev(ttnn, d, q.unsqueeze(0))
    k_tt = _to_dev(ttnn, d, kv.unsqueeze(1))
    cur_pos = [kv.shape[1] - 1] * b

    cfg, ckc = _sdpa_cfg(ttnn, d, params), _ckc(ttnn, d, params)

    def run():
        return ttnn.transformer.flash_multi_latent_attention_decode(q_tt, k_tt, dv, is_causal=True, cur_pos=cur_pos,
                                                                    program_config=cfg, compute_kernel_config=ckc)

    out = ttnn.to_torch(run()).float()[0]
    # Loom's program reads independent random tensors for the two views, so the
    # check is against the single-cache formula, first dv columns.
    ref = (torch.softmax((q.float() @ kv.float().transpose(1, 2)) / math.sqrt(dh), dim=-1) @ kv.float())[:, :, :dv]
    return out, run, lambda _golden: _pcc(out, ref)


def _mla_parts(ins):
    q = min(ins, key=lambda t: t.shape[1] * t.shape[2])
    kvt = next(t for t in ins if t is not q and t.shape[1] == q.shape[-1])
    kv = next(t for t in ins if t is not q and t is not kvt)
    return q, kv, kvt


def mla_decode_composite(ttnn, d, ins, shapes, params):
    import torch
    q, kv, kvt = _mla_parts(ins)
    q_tt, kv_tt = _to_dev(ttnn, d, q), _to_dev(ttnn, d, kv)
    kvt_tt = _to_dev(ttnn, d, kvt)
    scale = 1.0 / math.sqrt(q.shape[-1])

    ckc = _ckc(ttnn, d, params)

    def run():
        s = ttnn.matmul(q_tt, kvt_tt, compute_kernel_config=ckc)
        s = ttnn.multiply(s, scale)
        p = ttnn.softmax(s, dim=-1, compute_kernel_config=ckc, numeric_stable=True)
        return ttnn.matmul(p, kv_tt, compute_kernel_config=ckc)

    return ttnn.to_torch(run()).float(), run, None


def gqa_prefill_fused(ttnn, d, ins, shapes, params):
    """ttnn.transformer.scaled_dot_product_attention, non-causal, q [b, nqh, s, dh], k/v [b, nkv, s, dh] with b = 1."""
    q = max(ins, key=lambda t: t.shape[0])
    kt = next(t for t in ins if t is not q and t.shape[1] < t.shape[2])
    v = next(t for t in ins if t is not q and t is not kt)
    q_tt = _to_dev(ttnn, d, q.unsqueeze(0))
    k_tt = _to_dev(ttnn, d, kt.transpose(1, 2).contiguous().unsqueeze(0))
    v_tt = _to_dev(ttnn, d, v.unsqueeze(0))

    cfg, ckc = _sdpa_cfg(ttnn, d, params), _ckc(ttnn, d, params)

    def run():
        return ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, is_causal=False,
                                                             program_config=cfg, compute_kernel_config=ckc)

    return ttnn.to_torch(run()).float()[0], run, None


def gqa_prefill_composite(ttnn, d, ins, shapes, params):
    import torch
    q = max(ins, key=lambda t: t.shape[0])
    kt = next(t for t in ins if t is not q and t.shape[1] < t.shape[2])
    v = next(t for t in ins if t is not q and t is not kt)
    ratio = q.shape[0] // v.shape[0]
    g = torch.arange(q.shape[0]) // ratio
    q_tt, kt_tt, v_tt = (_to_dev(ttnn, d, t) for t in (q, kt[g].contiguous(), v[g].contiguous()))
    scale = 1.0 / math.sqrt(q.shape[-1])

    ckc = _ckc(ttnn, d, params)

    def run():
        s = ttnn.matmul(q_tt, kt_tt, compute_kernel_config=ckc)
        s = ttnn.multiply(s, scale)
        p = ttnn.softmax(s, dim=-1, compute_kernel_config=ckc, numeric_stable=True)
        return ttnn.matmul(p, v_tt, compute_kernel_config=ckc)

    return ttnn.to_torch(run()).float(), run, None


def pair_rotation_matrix(d: int):
    """x @ M = (-x1, x0, -x3, x2, ...): the interleaved rotate-half (same as kernels/rotary.py)."""
    import torch
    m = torch.zeros(d, d)
    for i in range(0, d, 2):
        m[i, i + 1] = 1.0
        m[i + 1, i] = -1.0
    return m


def _rotary_parts(ins):
    x = next(t for t in ins if t.dim() == 3)
    rot = next(t for t in ins if t.dim() == 2 and t.shape[0] == t.shape[1] and t.shape[0] == x.shape[-1])
    cos, sin = [t for t in ins if t is not x and t is not rot]
    return x, cos, sin, rot


def rotary_fused(ttnn, d, ins, shapes, params):
    """ttnn.experimental.rotary_embedding_llama with its 32x32 pair matrix. Loom's
    program takes a random [D, D] matrix, so this is checked against the same
    formula with the pair-rotation matrix instead of Loom's golden."""
    x, cos, sin, _ = _rotary_parts(ins)
    h, t, dh = x.shape
    x_tt = _to_dev(ttnn, d, x.unsqueeze(0))
    cos_tt = _to_dev(ttnn, d, cos.reshape(1, 1, t, dh))
    sin_tt = _to_dev(ttnn, d, sin.reshape(1, 1, t, dh))
    trans_tt = _to_dev(ttnn, d, pair_rotation_matrix(32).reshape(1, 1, 32, 32))

    ckc = _ckc(ttnn, d, params)

    def run():
        return ttnn.experimental.rotary_embedding_llama(x_tt, cos_tt, sin_tt, trans_tt, is_decode_mode=False,
                                                        compute_kernel_config=ckc)

    out = ttnn.to_torch(run()).float()[0]
    ref = x.float() * cos.float() + (x.float() @ pair_rotation_matrix(dh)) * sin.float()
    return out, run, lambda _golden: _pcc(out, ref)


def rotary_composite(ttnn, d, ins, shapes, params):
    x, cos, sin, rot = _rotary_parts(ins)
    x_tt, cos_tt, sin_tt, rot_tt = (_to_dev(ttnn, d, t) for t in (x, cos.unsqueeze(0), sin.unsqueeze(0), rot))

    ckc = _ckc(ttnn, d, params)

    def run():
        r = ttnn.matmul(x_tt, rot_tt, compute_kernel_config=ckc)
        return ttnn.add(ttnn.multiply(x_tt, cos_tt), ttnn.multiply(r, sin_tt))

    return ttnn.to_torch(run()).float(), run, None


def gemm_bias_exp_fused(ttnn, d, ins, shapes, params):
    """ttnn.linear with the bias fused; the scale and exp epilogue have no fused form in ttnn."""
    a, b, bias = ins
    a_tt, b_tt, bias_tt = (_to_dev(ttnn, d, t) for t in (a, b, bias))

    ckc = _ckc(ttnn, d, params)

    def run():
        y = ttnn.linear(a_tt, b_tt, bias=bias_tt, compute_kernel_config=ckc)
        return ttnn.exp(ttnn.multiply(y, 1.0 / 64.0))

    return ttnn.to_torch(run()).float(), run, None


def gemm_bias_exp_composite(ttnn, d, ins, shapes, params):
    a, b, bias = ins
    a_tt, b_tt, bias_tt = (_to_dev(ttnn, d, t) for t in (a, b, bias))

    ckc = _ckc(ttnn, d, params)

    def run():
        y = ttnn.add(ttnn.matmul(a_tt, b_tt, compute_kernel_config=ckc), bias_tt)
        return ttnn.exp(ttnn.multiply(y, 1.0 / 64.0))

    return ttnn.to_torch(run()).float(), run, None


def gemm_linear_only(ttnn, d, ins, shapes, params):
    """Reference, not the same computation: ttnn.linear with the bias and no scale/exp
    epilogue, to split the fused baseline's time into matmul and epilogue."""
    import torch
    a, b, bias = ins
    a_tt, b_tt, bias_tt = (_to_dev(ttnn, d, t) for t in (a, b, bias))
    ckc = _ckc(ttnn, d, params)

    def run():
        return ttnn.linear(a_tt, b_tt, bias=bias_tt, compute_kernel_config=ckc)

    out = ttnn.to_torch(run()).float()
    ref = a.float() @ b.float() + bias.float()
    return out, run, lambda _golden: _pcc(out, ref)


# ---- Library programs and the new operators (2026-10-12) -------------------
# One function per ttnn expression; `run` is what the timing replays. Where
# ttnn has no fused operator for the computation, only the op-by-op
# composite is listed; a single ttnn op is listed as "ttnn op".

def _simple(fn, ttnn, d, ins, params, golden=None):
    """Upload `ins`, time `fn(ttnn, tensors, ckc)`; PCC against `golden`
    (or the program's reference) on the first result."""
    tts = [_to_dev(ttnn, d, t) for t in ins]
    ckc = _ckc(ttnn, d, params)

    def run():
        return fn(ttnn, tts, ckc)

    out = ttnn.to_torch(run()).float()
    return out, run, (lambda _ref: _pcc(out, golden)) if golden is not None else None


def _softmax_composite_ops(ttnn, x, dim):
    m = ttnn.max(x, dim=dim, keepdim=True)
    e = ttnn.exp(ttnn.subtract(x, m))
    return ttnn.multiply(e, ttnn.reciprocal(ttnn.sum(e, dim=dim, keepdim=True)))


def matmul_op(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.matmul(a[0], a[1], compute_kernel_config=c), ttnn, d, ins, params)


def transpose_matmul_op(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.matmul(a[0], a[1], transpose_a=True, compute_kernel_config=c),
                   ttnn, d, ins, params)


def chain_matmul_composite(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.matmul(t.matmul(a[0], a[1], compute_kernel_config=c), a[2],
                                            compute_kernel_config=c), ttnn, d, ins, params)


def multiply_op(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.multiply(a[0], a[1]), ttnn, d, ins, params)


def add_op(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.add(a[0], a[1]), ttnn, d, ins, params)


def outer_scale_composite(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.multiply(t.multiply(a[0], a[1]), a[2]), ttnn, d, ins, params)


def batch_sum_op(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.sum(a[0], dim=0), ttnn, d, ins, params)


def weighted_batch_sum_composite(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.sum(t.multiply(a[0], a[1]), dim=0), ttnn, d, ins, params)


def softmax_fused(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.softmax(a[0], dim=-1, compute_kernel_config=c, numeric_stable=True),
                   ttnn, d, ins, params)


def softmax_composite(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: _softmax_composite_ops(t, a[0], -1), ttnn, d, ins, params)


def col_softmax_composite(ttnn, d, ins, shapes, params):
    """Softmax over dim 0: transpose, the fused row softmax, transpose back."""
    return _simple(lambda t, a, c: t.transpose(
        t.softmax(t.transpose(a[0], 0, 1), dim=-1, compute_kernel_config=c, numeric_stable=True), 0, 1),
        ttnn, d, ins, params)


def row_center_composite(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.subtract(a[0], t.mean(a[0], dim=-1, keepdim=True)), ttnn, d, ins, params)


def attention_fullrow_fused(ttnn, d, ins, shapes, params):
    """SDPA on [1, 1, S, D] views; the program has no 1/sqrt(D) scale."""
    import torch
    q, k, v = ins
    golden = torch.softmax(q.float() @ k.float().T, dim=-1) @ v.float()
    qt, kt, vt = (_to_dev(ttnn, d, t.reshape(1, 1, *t.shape)) for t in (q, k, v))
    ckc = _ckc(ttnn, d, params)

    def run():
        return ttnn.transformer.scaled_dot_product_attention(qt, kt, vt, is_causal=False, scale=1.0,
                                                             compute_kernel_config=ckc)

    out = ttnn.to_torch(run()).float().reshape(golden.shape)
    return out, run, lambda _ref: _pcc(out, golden)


def attention_fullrow_composite(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.matmul(
        t.softmax(t.matmul(a[0], a[1], transpose_b=True, compute_kernel_config=c), dim=-1,
                  compute_kernel_config=c, numeric_stable=True), a[2], compute_kernel_config=c),
        ttnn, d, ins, params)


def layernorm_fused(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.layer_norm(a[0], epsilon=1e-5, weight=a[2], bias=a[3],
                                                residual_input_tensor=a[1], compute_kernel_config=c),
                   ttnn, d, ins, params)


def layernorm_composite(ttnn, d, ins, shapes, params):
    def fn(t, a, c):
        h = t.add(a[0], a[1])
        dlt = t.subtract(h, t.mean(h, dim=-1, keepdim=True))
        var = t.mean(t.multiply(dlt, dlt), dim=-1, keepdim=True)
        y = t.multiply(dlt, t.rsqrt(t.add(var, 1e-5)))
        return t.add(t.multiply(y, a[2]), a[3])
    return _simple(fn, ttnn, d, ins, params)


def gemm_bias_gelu_fused(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.linear(a[0], a[1], bias=a[2], activation="gelu", compute_kernel_config=c),
                   ttnn, d, ins, params)


def gemm_bias_gelu_composite(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.gelu(t.add(t.matmul(a[0], a[1], compute_kernel_config=c), a[2])),
                   ttnn, d, ins, params)


def swiglu_fused(ttnn, d, ins, shapes, params):
    """The SiLU fused into the gate GEMM, the up GEMM, then the product."""
    return _simple(lambda t, a, c: t.multiply(t.linear(a[0], a[1], activation="silu", compute_kernel_config=c),
                                              t.matmul(a[0], a[2], compute_kernel_config=c)),
                   ttnn, d, ins, params)


def swiglu_composite(ttnn, d, ins, shapes, params):
    return _simple(lambda t, a, c: t.multiply(t.silu(t.matmul(a[0], a[1], compute_kernel_config=c)),
                                              t.matmul(a[0], a[2], compute_kernel_config=c)),
                   ttnn, d, ins, params)


def _mm2d_cfg(ttnn, m: int, n: int, k: int, params: dict, act=None, fuse_batch: bool = False):
    """2-D multicast matmul program config for an [m, k] x [k, n] matmul on a
    gx x gy grid (params gx, gy, bw = in0_block_w), the largest output
    subblock (h * w <= 8) dividing the per-core block, and an optional fused
    activation (applied in the matmul kernel together with the bias)."""
    mt, nt, kt = m // 32, n // 32, k // 32
    gx, gy, bw = params["gx"], params["gy"], params["bw"]
    pm, pn = -(-mt // gy), -(-nt // gx)
    sub = max(((h, w) for h in range(1, pm + 1) for w in range(1, pn + 1)
               if pm % h == 0 and pn % w == 0 and h * w <= 8), key=lambda s: (s[0] * s[1], s[1]))
    if kt % bw:
        raise ValueError(f"in0_block_w {bw} does not divide {kt} k tiles")
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=bw, out_subblock_h=sub[0],
        out_subblock_w=sub[1], per_core_M=pm, per_core_N=pn, transpose_mcast=False,
        fused_activation=act, fuse_batch=fuse_batch)


def matmul_2d_tuned(ttnn, d, ins, shapes, params):
    """ttnn.matmul with a 2-D multicast program config (batch folded into M
    when only the activation is batched)."""
    a, b = ins
    m = math.prod(a.shape[:-1])
    cfg = _mm2d_cfg(ttnn, m, b.shape[-1], a.shape[-1], params, fuse_batch=a.dim() > 2)
    return _simple(lambda t, x, c: t.matmul(x[0], x[1], program_config=cfg, compute_kernel_config=c),
                   ttnn, d, ins, params)


def matmul_1d_tuned(ttnn, d, ins, shapes, params):
    """ttnn.matmul with a 1-D config: the activation multicast to every core,
    each core a block of `pn` output tile columns (decode-shaped GEMMs)."""
    a, b = ins
    mt, nt, kt = a.shape[0] // 32, b.shape[1] // 32, a.shape[1] // 32
    pn, bw = params["pn"], params["bw"]
    sub_w = max(w for w in range(1, min(pn, 8) + 1) if pn % w == 0)
    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(12, 10), in0_block_w=bw, out_subblock_h=1, out_subblock_w=sub_w,
        per_core_M=mt, per_core_N=pn, fuse_batch=True, fused_activation=None, mcast_in0=True)
    if kt % bw or -(-nt // pn) > 120:
        raise ValueError("bad 1-D config")
    return _simple(lambda t, x, c: t.matmul(x[0], x[1], program_config=cfg, compute_kernel_config=c),
                   ttnn, d, ins, params)


def gemm_bias_gelu_mcast(ttnn, d, ins, shapes, params):
    """ttnn.linear with bias and GELU fused into the matmul kernel."""
    x, w, bias = ins
    cfg = _mm2d_cfg(ttnn, x.shape[0], w.shape[1], x.shape[1], params, (ttnn.UnaryOpType.GELU, True))
    return _simple(lambda t, a, c: t.linear(a[0], a[1], bias=a[2], program_config=cfg, compute_kernel_config=c),
                   ttnn, d, ins, params)


def gemm_bias_exp_mcast(ttnn, d, ins, shapes, params):
    """exp((x @ w + b) / 64) as one matmul: the constant scale folded into w and
    b offline (not timed), bias and exp fused into the matmul kernel."""
    import torch
    x, w, bias = ins
    golden = torch.exp((x.float() @ w.float() + bias.float()) / 64.0)
    cfg = _mm2d_cfg(ttnn, x.shape[0], w.shape[1], x.shape[1], params, (ttnn.UnaryOpType.EXP, False))
    return _simple(lambda t, a, c: t.linear(a[0], a[1], bias=a[2], program_config=cfg, compute_kernel_config=c),
                   ttnn, d, [x, w / 64.0, bias / 64.0], params, golden=golden)


def swiglu_mcast(ttnn, d, ins, shapes, params):
    """The best ttnn has: the gate matmul with SiLU fused, the up matmul, the
    product (no op computes two matmuls of one input)."""
    x, wg, wu = ins
    cfg = _mm2d_cfg(ttnn, x.shape[0], wg.shape[1], x.shape[1], params, ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU))
    cfg_up = _mm2d_cfg(ttnn, x.shape[0], wu.shape[1], x.shape[1], params)
    return _simple(lambda t, a, c: t.multiply(t.matmul(a[0], a[1], program_config=cfg, compute_kernel_config=c),
                                              t.matmul(a[0], a[2], program_config=cfg_up, compute_kernel_config=c)),
                   ttnn, d, ins, params)


BASELINES = {
    "gemm_longk": {"ttnn op": matmul_op, "ttnn op (tuned)": matmul_2d_tuned},
    "gemv_decode": {"ttnn op": matmul_op, "ttnn op (tuned 1d)": matmul_1d_tuned},
    "moe_expert_bmm": {"ttnn op": matmul_op},
    "toy_bmm_full": {"ttnn op": matmul_op},
    "shared_weight_bmm_full": {"ttnn op": matmul_op, "ttnn op (tuned)": matmul_2d_tuned},
    "transpose_matmul_full": {"ttnn composite": transpose_matmul_op},
    "chain_fused_full": {"ttnn composite": chain_matmul_composite},
    "scaled_ewise_full": {"ttnn op": multiply_op},
    "bias_add_full": {"ttnn op": add_op},
    "outer_scale_full": {"ttnn composite": outer_scale_composite},
    "batch_sum_full": {"ttnn op": batch_sum_op},
    "weighted_batch_sum_full": {"ttnn composite": weighted_batch_sum_composite},
    "row_softmax_full": {"ttnn fused": softmax_fused, "ttnn composite": softmax_composite},
    "softmax_twopass_full": {"ttnn fused": softmax_fused, "ttnn composite": softmax_composite},
    "col_softmax_full": {"ttnn composite": col_softmax_composite},
    "row_center_full": {"ttnn composite": row_center_composite},
    "attention_fullrow_full": {"ttnn fused": attention_fullrow_fused, "ttnn composite": attention_fullrow_composite},
    "layernorm_residual": {"ttnn fused": layernorm_fused, "ttnn composite": layernorm_composite},
    "gemm_bias_gelu": {"ttnn fused": gemm_bias_gelu_mcast, "ttnn composite": gemm_bias_gelu_composite},
    "swiglu": {"ttnn partly fused": swiglu_mcast, "ttnn composite": swiglu_composite},
    "gqa_decode": {"ttnn fused": gqa_decode_fused, "ttnn composite": gqa_decode_composite},
    "mask_softmax": {"ttnn fused": mask_softmax_fused, "ttnn composite": mask_softmax_composite},
    "rmsnorm_residual": {"ttnn fused": rmsnorm_fused, "ttnn composite": rmsnorm_composite},
    "mla_decode": {"ttnn fused": mla_decode_fused, "ttnn composite": mla_decode_composite},
    "gqa_prefill": {"ttnn fused": gqa_prefill_fused, "ttnn composite": gqa_prefill_composite},
    "rotary": {"ttnn fused": rotary_fused, "ttnn composite": rotary_composite},
    # ttnn.linear cannot fuse exp as an activation ("Unsupported activation
    # function"), so the epilogue runs as separate ops.
    "gemm_bias_exp_full": {"ttnn fused (scale folded)": gemm_bias_exp_mcast,
                           "ttnn composite": gemm_bias_exp_composite,
                           "ttnn linear+bias only (reference)": gemm_linear_only},
}


# Program-config sweep per fused operator: every point is run and reported.
_MM_POINTS = [{"gx": gx, "gy": gy, "bw": bw} for gx, gy in ((12, 10), (8, 8), (8, 10), (12, 8))
              for bw in (2, 4, 8, 16)]

SWEEPS = {
    ("gemm_bias_gelu", "ttnn fused"): _MM_POINTS,
    ("gemm_bias_exp_full", "ttnn fused (scale folded)"): _MM_POINTS,
    ("swiglu", "ttnn partly fused"): _MM_POINTS,
    ("gemm_longk", "ttnn op (tuned)"): [{"gx": gx, "gy": gy, "bw": bw} for gx, gy in ((8, 8), (4, 8), (8, 4))
                                        for bw in (4, 8, 16)],
    ("shared_weight_bmm_full", "ttnn op (tuned)"): _MM_POINTS,
    ("gemv_decode", "ttnn op (tuned 1d)"): [{"pn": pn, "bw": bw} for pn in (2, 4) for bw in (4, 8, 16)],
    ("gqa_decode", "ttnn fused"): [{}] + [{"k_chunk": k} for k in (64, 128, 256, 512)],
    ("mla_decode", "ttnn fused"): [{}] + [{"k_chunk": k} for k in (64, 128, 256)],
    ("gqa_prefill", "ttnn fused"): [{}] + [{"q_chunk": q, "k_chunk": k} for q in (64, 128, 256) for k in (64, 128, 256, 512)],
}


def _point_name(name: str, point: dict) -> str:
    if not point:
        return f"{name} (default cfg)"
    if "gx" in point:
        return f"{name} {point['gx']}x{point['gy']} bw{point['bw']}"
    if "pn" in point:
        return f"{name} pn{point['pn']} bw{point['bw']}"
    return name + " " + " ".join(f"{k.split('_')[0]}{v}" for k, v in point.items())


def roofline_bytes(program: str, shapes: list[list[int]], params: dict) -> int:
    """Compulsory DRAM traffic in bytes: every argument read or written once."""
    n = sum(math.prod(s) for s in shapes)
    return 2 * n  # bf16


# --------------------------------------------------------------------------

def _pcc(a, b) -> float:
    import torch
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


PER = 20  # launches per timing sample


def device_ops(ttnn, run) -> list[str]:
    """The device operations one call of `run` launches, in order (ttnn graph
    capture of an untraced call). A baseline is fused only if this is one."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        run()
    finally:
        graph = ttnn.graph.end_graph_capture()
    names = [n.get("params", {}).get("name", "") for n in graph
             if n.get("node_type") == "function_start"]
    # Every device-op launch goes through a ttnn::prim::* call; the call
    # after it names the operation (e.g. Matmul, BinaryNgDeviceOperation).
    ops = []
    for i, name in enumerate(names):
        if name.startswith("ttnn::prim::"):
            nxt = names[i + 1] if i + 1 < len(names) else name
            ops.append(nxt if not nxt.startswith("ttnn::") else name)
    return ops


def worker(program: str, baseline: str, shapes: list[list[int]], params: dict, iters: int, warmup: int) -> None:
    import torch
    import ttnn
    prog = PROGRAMS[program]
    torch.manual_seed(0)
    # Same inputs as benchmarks/run.py: randn per input in argument order.
    ins = [torch.randn(s) for s in shapes[:-1]]
    ref = prog.reference([t.float() for t in ins]).float()
    d = ttnn.open_device(device_id=0, trace_region_size=64 << 20)
    try:
        out, run, pcc_fn = BASELINES[program][baseline](ttnn, d, ins, shapes, params)
        ttnn.synchronize_device(d)
        pcc = pcc_fn(ref) if pcc_fn else _pcc(out, ref)
        ops = device_ops(ttnn, run)
        ttnn.synchronize_device(d)
        # Same timing as benchmarks/run.py: replay a captured trace PER times
        # per synchronization (no host dispatch in the samples); ops that
        # cannot be traced fall back to PER plain launches per synchronization.
        timing, launch = "trace", None
        try:
            tid = ttnn.begin_trace_capture(d, cq_id=0)
            run()
            ttnn.end_trace_capture(d, tid, cq_id=0)
            launch = lambda: ttnn.execute_trace(d, tid, cq_id=0, blocking=False)
        except Exception:
            timing, launch = "batch", run
        for _ in range(warmup):
            for _ in range(PER):
                launch()
            ttnn.synchronize_device(d)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            for _ in range(PER):
                launch()
            ttnn.synchronize_device(d)
            ts.append((time.perf_counter() - t0) / PER)
        g = d.compute_with_storage_grid_size()
        print("RESULT " + json.dumps({"pcc": pcc, "device_ms": statistics.median(ts) * 1e3,
                                      "grid": [g.x, g.y], "timing": timing, "device_ops": ops}),
              flush=True)
    finally:
        ttnn.close_device(d)


def run_baseline(program, baseline, shapes, params, iters, warmup, timeout_s) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", program, baseline,
           json.dumps(shapes), json.dumps(params), str(iters), str(warmup)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=ROOT)
    except subprocess.TimeoutExpired:
        reset_board()
        return {"error": f"hang: no result within {timeout_s} s; board reset"}
    line = next((l for l in reversed(res.stdout.splitlines()) if l.startswith("RESULT ")), None)
    if res.returncode != 0 or line is None:
        reset_board()
        tail = (res.stderr or res.stdout)[-800:]
        return {"error": f"failed (exit {res.returncode}): {tail.strip()}"}
    return json.loads(line[len("RESULT "):])


def loom_results(program: str) -> list[dict]:
    """Best passing candidate per tag from benchmarks/run.py, with the PE kernel it ran
    (results written before the PE kernel was recorded have none and are skipped)."""
    rows = []
    for res in sorted((ROOT / "test/bench" / program).glob("*/results.json")):
        data = json.loads(res.read_text())
        ran = [c for c in data["candidates"] if c.get("stage") in ("ok", "pcc")]
        if not ran:
            continue
        ok = [c for c in ran if c["stage"] == "ok"]
        best = min(ok or ran, key=lambda c: c["device_ms"])
        rows.append({"tag": res.parent.name, "function": best["function"], "device_ms": best["device_ms"],
                     "pcc": best["pcc"], "out": str(res.parent.relative_to(ROOT)),
                     "pe_kernel": data["summary"].get("pe_kernel", {})})
    return rows


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _, _, program, baseline, shapes, params, iters, warmup = sys.argv
        worker(program, baseline, json.loads(shapes), json.loads(params), int(iters), int(warmup))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("program", choices=sorted(BASELINES))
    ap.add_argument("--param", action="append", default=[], help="key=value for the baselines")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--reset-each", action="store_true")
    ap.add_argument("--fidelity", default="HiFi4", choices=["LoFi", "HiFi2", "HiFi3", "HiFi4"])
    ap.add_argument("--no-approx", action="store_true")
    ap.add_argument("--tag", default=None,
                    help="Loom result tag (test/bench/<program>/<tag>) whose shapes the baselines use; "
                         "only Loom rows of that shape are listed")
    ap.add_argument("--grid", default=None, metavar="XxY",
                    help="restrict ttnn to an X x Y compute grid (e.g. 12x10, Loom's hardware spec); "
                         "results go to a file suffixed _g<XxY>")
    a = ap.parse_args()
    if a.grid:
        gx, gy = (int(v) for v in a.grid.lower().split("x"))
        # Shrinks compute_with_storage_grid_size(), which every ttnn op sizes its work by.
        os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = f"{gx - 1},{gy - 1}"
    params = dict(kv.split("=", 1) for kv in a.param)
    params["fidelity"], params["approx"] = a.fidelity, not a.no_approx
    pe = {"math_fidelity": a.fidelity, "math_approx_mode": not a.no_approx,
          "fp32_dest_acc_en": False, "packer_l1_acc": True, "data_format": "bf16"}

    loom = [l for l in loom_results(a.program) if l.get("pe_kernel") == {k: pe[k] for k in l.get("pe_kernel", {})}
            and l.get("pe_kernel", {}).get("math_fidelity") == a.fidelity]
    if a.tag:
        loom = [l for l in loom if l["tag"] == a.tag] + [l for l in loom if l["tag"] != a.tag]
    if not loom:
        sys.exit(f"no Loom result that ran under test/bench/{a.program} at {a.fidelity}; "
                 f"run benchmarks/run.py {a.program} --fidelity {a.fidelity} first")
    shapes = arg_shapes(Path(loom[0]["out"]), loom[0]["function"])
    loom = [l for l in loom if arg_shapes(Path(l["out"]), l["function"]) == shapes]
    bytes_once = roofline_bytes(a.program, shapes, params)
    roof_ms = bytes_once / DRAM_BYTES_PER_S * 1e3

    rows = []
    for name in BASELINES[a.program]:
        for point in SWEEPS.get((a.program, name), [{}]):
            if a.reset_each:
                reset_board()
            r = run_baseline(a.program, name, shapes, {**params, **point}, a.iters, a.warmup, a.timeout)
            if a.grid and "grid" in r and r["grid"] != [gx, gy]:
                r = {"error": f"ran on a {r['grid'][0]}x{r['grid'][1]} grid, not {a.grid}"}
            label = _point_name(name, point) if (a.program, name) in SWEEPS else name
            rows.append({"name": label, "baseline": name, "program_config": point, **r})
    for l in loom:
        rows.append({"name": f"loom {l['tag']}", "pcc": l["pcc"], "device_ms": l["device_ms"], "function": l["function"]})
    rows.append({"name": "roofline (512 GB/s)", "device_ms": roof_ms, "bytes": bytes_once})

    print(f"== {a.program}  shapes={shapes}  params={params}")
    print(f"   PE kernel (both sides): {pe}")
    print(f"{'yardstick':<36} {'ms':>8} {'x roof':>7} {'PCC':>9}")
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<36} {'FAIL':>8}          {r['error'].splitlines()[0][:80] if r['error'] else ''}")
            continue
        pcc = f"{r['pcc']:.6f}" if "pcc" in r else ""
        ops = r.get("device_ops")
        opstr = f"  {len(ops)} op{'s' if len(ops) != 1 else ''}: {', '.join(ops)}" if ops is not None else ""
        print(f"{r['name']:<36} {r['device_ms']:8.3f} {r['device_ms'] / roof_ms:7.2f} {pcc:>9}{opstr}")
    for base in BASELINES[a.program]:
        ok = [r for r in rows if r.get("baseline") == base and "device_ms" in r and r.get("pcc", 0) >= 0.99]
        if ok:
            best = min(ok, key=lambda r: r["device_ms"])
            print(f"best {base}: {best['name']} {best['device_ms']:.3f} ms")
    stem = f"{a.program}_{a.tag}" if a.tag else f"{a.program}_{a.fidelity}"
    out = ROOT / "tmp_output/sota" / (f"{stem}_g{a.grid}.json" if a.grid else f"{stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"program": a.program, "shapes": shapes, "params": params, "pe_kernel": pe,
                               "grid": a.grid,
                               "rows": rows}, indent=2))
    print(f"results: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
