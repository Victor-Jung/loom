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

    python benchmarks/sota.py gqa_decode --param groups=8
    python benchmarks/sota.py mask_softmax
"""

from __future__ import annotations

import argparse
import json
import math
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

    def run():
        return ttnn.transformer.scaled_dot_product_attention_decode(q_tt, k_tt, v_tt, is_causal=True, cur_pos=cur_pos)

    out = ttnn.to_torch(run()).float()[0]                             # [b, nh, dh]
    full = torch.zeros(b, groups, rows, dh)
    full[:, :, :ratio, :] = out.view(b, groups, ratio, dh)
    return full.view(bg, rows, dh), run, lambda ref: _pcc(out.reshape(-1), ref.view(b, groups, rows, dh)[:, :, :ratio, :].reshape(-1))


def gqa_decode_composite(ttnn, d, ins, shapes, params):
    kt, v, q_pad = ins
    q_tt, kt_tt, v_tt = (_to_dev(ttnn, d, t) for t in (q_pad, kt, v))
    scale = 1.0 / math.sqrt(q_pad.shape[-1])

    def run():
        s = ttnn.matmul(q_tt, kt_tt)
        s = ttnn.multiply(s, scale)
        p = ttnn.softmax(s, dim=-1)
        return ttnn.matmul(p, v_tt)

    return ttnn.to_torch(run()).float(), run, None


def mask_softmax_fused(ttnn, d, ins, shapes, params):
    """ttnn.scale_mask_softmax_in_place with a causal-style [1, 1, S, N] mask shared by batch and heads."""
    x, mask = ins
    x_tt = _to_dev(ttnn, d, x)
    # ttnn wants the mask's batch dim to match the input's: [B, 1, S, N].
    m_tt = _to_dev(ttnn, d, mask.reshape(1, 1, *mask.shape).expand(x.shape[0], 1, *mask.shape).contiguous())
    scale = 1.0 / math.sqrt(x.shape[-1])

    def run():
        return ttnn.scale_mask_softmax_in_place(x_tt, scale, m_tt, is_causal_mask=True)

    out = ttnn.to_torch(run()).float()
    # In place: the timed runs re-normalize an already normalized tensor, which
    # moves the same bytes, so the timing stands; the PCC uses the first call.
    return out, run, None


def mask_softmax_composite(ttnn, d, ins, shapes, params):
    x, mask = ins
    x_tt = _to_dev(ttnn, d, x)
    m_tt = _to_dev(ttnn, d, mask.reshape(1, 1, *mask.shape))
    scale = 1.0 / math.sqrt(x.shape[-1])

    def run():
        s = ttnn.multiply(x_tt, scale)
        s = ttnn.add(s, m_tt)
        return ttnn.softmax(s, dim=-1)

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

    def run():
        return ttnn.rms_norm(x_tt, epsilon=1e-5, weight=w_tt, residual_input_tensor=r_tt)

    out = ttnn.to_torch(run()).float()
    return out, run, lambda _golden: _pcc(out, ref)


def rmsnorm_composite(ttnn, d, ins, shapes, params):
    x, res, w, ref = _rmsnorm_parts(ins)
    x_tt, r_tt, w_tt = (_to_dev(ttnn, d, t) for t in (x, res, w))

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

    cfg = None
    if hasattr(ttnn, "SDPAProgramConfig"):
        # The default config allocates 5.7 MB of L1 for 128 heads x 512 at 4096
        # positions; smaller chunks fit.
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(12, 10), q_chunk_size=32,
                                     k_chunk_size=int(params.get("k_chunk", 128)), exp_approx_mode=False)

    def run():
        return ttnn.transformer.flash_multi_latent_attention_decode(q_tt, k_tt, dv, is_causal=True, cur_pos=cur_pos,
                                                                    program_config=cfg)

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

    def run():
        s = ttnn.matmul(q_tt, kvt_tt)
        s = ttnn.multiply(s, scale)
        p = ttnn.softmax(s, dim=-1)
        return ttnn.matmul(p, kv_tt)

    return ttnn.to_torch(run()).float(), run, None


def gqa_prefill_fused(ttnn, d, ins, shapes, params):
    """ttnn.transformer.scaled_dot_product_attention, non-causal, q [b, nqh, s, dh], k/v [b, nkv, s, dh] with b = 1."""
    q = max(ins, key=lambda t: t.shape[0])
    kt = next(t for t in ins if t is not q and t.shape[1] < t.shape[2])
    v = next(t for t in ins if t is not q and t is not kt)
    q_tt = _to_dev(ttnn, d, q.unsqueeze(0))
    k_tt = _to_dev(ttnn, d, kt.transpose(1, 2).contiguous().unsqueeze(0))
    v_tt = _to_dev(ttnn, d, v.unsqueeze(0))

    def run():
        return ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, is_causal=False)

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

    def run():
        s = ttnn.matmul(q_tt, kt_tt)
        s = ttnn.multiply(s, scale)
        p = ttnn.softmax(s, dim=-1)
        return ttnn.matmul(p, v_tt)

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

    def run():
        return ttnn.experimental.rotary_embedding_llama(x_tt, cos_tt, sin_tt, trans_tt, is_decode_mode=False)

    out = ttnn.to_torch(run()).float()[0]
    ref = x.float() * cos.float() + (x.float() @ pair_rotation_matrix(dh)) * sin.float()
    return out, run, lambda _golden: _pcc(out, ref)


def rotary_composite(ttnn, d, ins, shapes, params):
    x, cos, sin, rot = _rotary_parts(ins)
    x_tt, cos_tt, sin_tt, rot_tt = (_to_dev(ttnn, d, t) for t in (x, cos.unsqueeze(0), sin.unsqueeze(0), rot))

    def run():
        r = ttnn.matmul(x_tt, rot_tt)
        return ttnn.add(ttnn.multiply(x_tt, cos_tt), ttnn.multiply(r, sin_tt))

    return ttnn.to_torch(run()).float(), run, None


def gemm_bias_exp_fused(ttnn, d, ins, shapes, params):
    """ttnn.linear with the bias fused; the scale and exp epilogue have no fused form in ttnn."""
    a, b, bias = ins
    a_tt, b_tt, bias_tt = (_to_dev(ttnn, d, t) for t in (a, b, bias))

    def run():
        y = ttnn.linear(a_tt, b_tt, bias=bias_tt)
        return ttnn.exp(ttnn.multiply(y, 1.0 / 64.0))

    return ttnn.to_torch(run()).float(), run, None


def gemm_bias_exp_composite(ttnn, d, ins, shapes, params):
    a, b, bias = ins
    a_tt, b_tt, bias_tt = (_to_dev(ttnn, d, t) for t in (a, b, bias))

    def run():
        y = ttnn.add(ttnn.matmul(a_tt, b_tt), bias_tt)
        return ttnn.exp(ttnn.multiply(y, 1.0 / 64.0))

    return ttnn.to_torch(run()).float(), run, None


BASELINES = {
    "gqa_decode": {"ttnn fused": gqa_decode_fused, "ttnn composite": gqa_decode_composite},
    "mask_softmax": {"ttnn fused": mask_softmax_fused, "ttnn composite": mask_softmax_composite},
    "rmsnorm_residual": {"ttnn fused": rmsnorm_fused, "ttnn composite": rmsnorm_composite},
    "mla_decode": {"ttnn fused": mla_decode_fused, "ttnn composite": mla_decode_composite},
    "gqa_prefill": {"ttnn fused": gqa_prefill_fused, "ttnn composite": gqa_prefill_composite},
    "rotary": {"ttnn fused": rotary_fused, "ttnn composite": rotary_composite},
    "gemm_bias_exp_full": {"ttnn linear+bias, exp": gemm_bias_exp_fused, "ttnn composite": gemm_bias_exp_composite},
}


def roofline_bytes(program: str, shapes: list[list[int]], params: dict) -> int:
    """Compulsory DRAM traffic in bytes: every argument read or written once."""
    n = sum(math.prod(s) for s in shapes)
    return 2 * n  # bf16


# --------------------------------------------------------------------------

def _pcc(a, b) -> float:
    import torch
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def worker(program: str, baseline: str, shapes: list[list[int]], params: dict, iters: int, warmup: int) -> None:
    import torch
    import ttnn
    prog = PROGRAMS[program]
    torch.manual_seed(0)
    # Same inputs as benchmarks/run.py: randn per input in argument order.
    ins = [torch.randn(s) for s in shapes[:-1]]
    ref = prog.reference([t.float() for t in ins]).float()
    d = ttnn.open_device(device_id=0)
    try:
        out, run, pcc_fn = BASELINES[program][baseline](ttnn, d, ins, shapes, params)
        ttnn.synchronize_device(d)
        pcc = pcc_fn(ref) if pcc_fn else _pcc(out, ref)
        for _ in range(warmup):
            run(); ttnn.synchronize_device(d)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); run(); ttnn.synchronize_device(d)
            ts.append(time.perf_counter() - t0)
        print("RESULT " + json.dumps({"pcc": pcc, "device_ms": statistics.median(ts) * 1e3}), flush=True)
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
    """Best passing candidate per tag from benchmarks/run.py."""
    rows = []
    for res in sorted((ROOT / "test/bench" / program).glob("*/results.json")):
        data = json.loads(res.read_text())
        ok = [c for c in data["candidates"] if c.get("stage") == "ok"]
        if not ok:
            continue
        best = min(ok, key=lambda c: c["device_ms"])
        rows.append({"tag": res.parent.name, "function": best["function"], "device_ms": best["device_ms"],
                     "pcc": best["pcc"], "out": str(res.parent.relative_to(ROOT))})
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
    ap.add_argument("--tag", default=None,
                    help="Loom result tag (test/bench/<program>/<tag>) whose shapes the baselines use; "
                         "only Loom rows of that shape are listed")
    a = ap.parse_args()
    params = dict(kv.split("=", 1) for kv in a.param)

    loom = loom_results(a.program)
    if a.tag:
        loom = [l for l in loom if l["tag"] == a.tag] + [l for l in loom if l["tag"] != a.tag]
    if not loom:
        sys.exit(f"no passing Loom result under test/bench/{a.program}; run benchmarks/run.py first")
    shapes = arg_shapes(Path(loom[0]["out"]), loom[0]["function"])
    loom = [l for l in loom if arg_shapes(Path(l["out"]), l["function"]) == shapes]
    bytes_once = roofline_bytes(a.program, shapes, params)
    roof_ms = bytes_once / DRAM_BYTES_PER_S * 1e3

    rows = []
    for name in BASELINES[a.program]:
        if a.reset_each:
            reset_board()
        r = run_baseline(a.program, name, shapes, params, a.iters, a.warmup, a.timeout)
        rows.append({"name": name, **r})
    for l in loom:
        rows.append({"name": f"loom {l['tag']}", "pcc": l["pcc"], "device_ms": l["device_ms"], "function": l["function"]})
    rows.append({"name": "roofline (512 GB/s)", "device_ms": roof_ms, "bytes": bytes_once})

    print(f"== {a.program}  shapes={shapes}  params={params}")
    print(f"{'yardstick':<28} {'ms':>8} {'x roof':>7} {'PCC':>9}")
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<28} {'FAIL':>8}          {r['error'][:90]}")
            continue
        pcc = f"{r['pcc']:.6f}" if "pcc" in r else ""
        print(f"{r['name']:<28} {r['device_ms']:8.3f} {r['device_ms'] / roof_ms:7.2f} {pcc:>9}")
    out = ROOT / "tmp_output/sota" / (f"{a.program}_{a.tag}.json" if a.tag else f"{a.program}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"program": a.program, "shapes": shapes, "params": params, "rows": rows}, indent=2))
    print(f"results: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
