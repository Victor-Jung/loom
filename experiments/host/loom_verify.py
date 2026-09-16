#!/usr/bin/env python
"""Functional-correctness harness for Loom-generated kernels.

Semantics are taken from the Helion definitions in kernels/*.py:

  flash_attention : out[b,m,:] = softmax(q[b,m,:] @ K[b]^T * 1/sqrt(d)) @ V[b]
                    (non-causal; B folds logical_batch*heads; d = 65536/H)
  mqa_decode      : out[b,h,:] = softmax(q[b,h,:] @ K[b]^T * 1/sqrt(D)) @ V[b]
                    (single query token; K/V shared across heads; the split-LSE
                     merge in the kernel is a reduction, so plain attention is
                     the reference)

Operand roles are recovered from memref shapes, and the Q/V ambiguity (identical
shapes) is resolved empirically: softmax rows sum to 1, so feeding a constant
tensor as V makes the output that same constant.

Large shapes are verified by sampling output positions and computing an exact
per-row reference, which costs O(L*d) instead of O(L^2*d).
"""
import argparse, importlib.util, math, re, sys
from pathlib import Path
import torch, ttnn

ap = argparse.ArgumentParser()
ap.add_argument("--kernel", required=True, choices=["flash_attention", "mqa_decode"])
ap.add_argument("--ir", required=True)
ap.add_argument("--kernels", required=True)
ap.add_argument("--samples", type=int, default=64, help="output rows to spot-check")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

hp = Path(a.kernels) / "host_ttnn.py"; text = hp.read_text()
func = re.search(r"^run = (\w+)_ttnn$", text, re.M).group(1)
sig = re.search(rf"func\.func @{re.escape(func)}\(([^)]*)\)", Path(a.ir).read_text()).group(1)
shapes = [[int(d) for d in m.group(1).split("x")]
          for m in re.finditer(r"memref<([0-9x]+)x(?:f16|bf16|f32)>", sig)]
nx = int(re.search(r"^\s*end_core_x = (\d+)$", text, re.M).group(1)) + 1
ny = int(re.search(r"^\s*end_core_y = (\d+)$", text, re.M).group(1)) + 1
spec = importlib.util.spec_from_file_location("lh", hp)
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
order = h._PARAM_ORDER
out_names = set(h._OUTPUT_PARAM_ORDER)
n_in = len(order) - len(out_names)

# ---- identify operand roles from shapes -------------------------------------
# Kt is [B, d, L]; Q/V are [B, L, d] (FA) or Q is [B,H,D], V is [B,L,D] (decode).
out_shape = shapes[-1]
B = out_shape[0]
if a.kernel == "flash_attention":
    L, d = out_shape[1], out_shape[2]
    kt_idx = next(i for i, s in enumerate(shapes[:n_in]) if s == [B, d, L])
    cand = [i for i in range(n_in) if i != kt_idx]          # Q and V, same shape
else:
    H, D = out_shape[1], out_shape[2]
    kt_idx = next(i for i, s in enumerate(shapes[:n_in]) if s[1] == D and s[2] != D)
    L = shapes[kt_idx][2]
    q_idx = next(i for i, s in enumerate(shapes[:n_in]) if s == [B, H, D])
    v_idx = next(i for i, s in enumerate(shapes[:n_in]) if s == [B, L, D])
    cand = None
scale = 1.0 / math.sqrt(d if a.kernel == "flash_attention" else D)

dev = ttnn.open_device(device_id=0)
try:
    g = dev.compute_with_storage_grid_size()
    if nx > g.x or ny > g.y:
        sys.exit(f"SKIP: needs {nx}x{ny}, device has {g.x}x{g.y}")
    mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    torch.manual_seed(a.seed)

    def run(tensors):
        args = [mk(t) for t in tensors]
        o = h.run(*args)
        o = o if not isinstance(o, tuple) else o[0]
        res = ttnn.to_torch(o).float()
        # these are multi-GB for flash attention; free before the next run
        for t in args:
            try: ttnn.deallocate(t)
            except Exception: pass
        return res

    if a.kernel == "flash_attention":
        # probe: constant candidate -> if it is V, every output element equals it
        probe_const = 0.375
        # Comparative, not thresholded: with a constant V the exact output is that
        # constant, so the true V is whichever candidate gets closest. An absolute
        # bound fails at long L, where bf16 softmax accumulation is itself lossy.
        scores_ = {}
        for guess in cand:
            ts = []
            for i, s in enumerate(shapes):
                if i == guess:       ts.append(torch.full(s, probe_const, dtype=torch.float16))
                elif i < n_in:       ts.append(torch.randn(s, dtype=torch.float16))
                else:                ts.append(torch.zeros(s, dtype=torch.float16))
            o = run(ts)
            e_max = (o - probe_const).abs().max().item()
            e_mean = (o - probe_const).abs().mean().item()
            scores_[guess] = e_mean
            print(f"  probe arg{guess} as V: mean|out-{probe_const}|={e_mean:.4f} max={e_max:.4f}")
        v_idx = min(scores_, key=scores_.get)
        best, worst = sorted(scores_.values())[0], sorted(scores_.values())[-1]
        if worst < 2 * best:
            sys.exit(f"  AMBIGUOUS probe ({scores_}); cannot assign V")
        print(f"  constant-V closed-form check: mean abs err {best:.4f} "
              f"({best/probe_const*100:.1f}% of exact {probe_const})")
        q_idx = [i for i in cand if i != v_idx][0]
        print(f"  roles: Kt=arg{kt_idx}  Q=arg{q_idx}  V=arg{v_idx}  out=arg{len(shapes)-1}")

    # ---- real run -----------------------------------------------------------
    torch.manual_seed(a.seed)
    ts = [torch.randn(s, dtype=torch.float16) if i < n_in else torch.zeros(s, dtype=torch.float16)
          for i, s in enumerate(shapes)]
    got = run(ts)
    Kt, Q, V = ts[kt_idx], ts[q_idx], ts[v_idx]

    # ---- sampled exact reference -------------------------------------------
    gen = torch.Generator().manual_seed(a.seed + 1)
    rows = out_shape[1]
    n = min(a.samples, B * rows)
    bs = torch.randint(0, B, (n,), generator=gen)
    ms = torch.randint(0, rows, (n,), generator=gen)
    ref = torch.empty(n, out_shape[2], dtype=torch.float32)
    obs = torch.empty_like(ref)
    for i in range(n):
        b, m = int(bs[i]), int(ms[i])
        qr = Q[b, m, :].float()                      # [d]
        sc = (qr @ Kt[b].float()) * scale            # [L]
        p = torch.softmax(sc, dim=-1)
        ref[i] = p @ V[b].float()                    # [d]
        obs[i] = got[b, m, :]
    o_f, r_f = obs.flatten(), ref.flatten()
    pcc = torch.corrcoef(torch.stack([o_f, r_f]))[0, 1].item()
    denom = ref.abs().max().clamp_min(1e-6)
    rel = ((obs - ref).abs().max() / denom).item()
    err = (o_f - r_f)
    mean_rel = (err.abs().mean() / denom).item()
    # systematic scale/bias: obs ~= slope*ref + bias  (slope!=1 => real bug, not noise)
    slope = float((o_f @ r_f) / (r_f @ r_f).clamp_min(1e-12))
    bias = float(err.mean())
    print(f"  shape {out_shape}  grid {nx}x{ny}  samples={n}")
    print(f"  PCC {pcc:.6f}")
    print(f"  err/max|ref|:  max {rel:.4f}   mean {mean_rel:.5f}")
    scale_off = abs(slope - 1.0)
    print(f"  fit obs=a*ref+b:  a={slope:.5f}  b={bias:+.5f}")
    # PCC alone is insensitive to a uniform scale error: a systematically inflated
    # output still correlates ~1.0 with the reference. Gate on the slope too.
    if pcc <= 0.99:
        print("  FAIL (low PCC)")
    elif scale_off > 0.05:
        print(f"  FAIL (systematic scale error: output is {slope:.3f}x reference)")
    else:
        print("  PASS")
finally:
    ttnn.close_device(dev)
