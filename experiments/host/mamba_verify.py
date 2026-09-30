#!/usr/bin/env python
"""Functional verification of the Loom-generated mamba_chunk_scan kernel.

Argument layouts are taken from the frontend MLIR's named arguments; all
transposes are hoisted into the arguments, so the driver must pass the
already-transposed tensors:

  0 cb            [B, NC, G, CS, CS]
  1 x             [B, H, S, Dh]        x.transpose(1,2)
  2 dt_k          [B, H, NC, 1, CS]
  3 dA_cumsum_m   [B, H, NC, CS, 1]
  4 dA_cumsum_k   [B, H, NC, 1, CS]
  5 C             [B, G, S, ds]        C.transpose(1,2)
  6 xD            [B, H, S, Dh]        (x * D[h]).transpose(1,2)
  7 prev_states_T [B, NC, H, ds, Dh]   prev_states.transpose(3,4)
  8 out           [B, H, S, Dh]

Reference mode is `none`: the kernel's dynamic trip count (tile_m.id+1)*block_m
was replaced by a static chunk_size, so every k in the chunk contributes.
"""
import argparse, importlib.util, sys
from pathlib import Path
import torch, ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mamba_ref import mamba_ref

ap = argparse.ArgumentParser()
ap.add_argument("--kernels", default="/home/vicjung/loom/tmp_output/k_mamba")
ap.add_argument("--ir", required=True, help="p03 MLIR; shapes are derived from its memref args")
ap.add_argument("--causal", default="none", choices=["none", "block", "element"])
ap.add_argument("--const-vectors", action="store_true",
                help="make dt/dA constant so any misread of the [1,CS]/[CS,1] vector "
                     "loads is harmless; isolates vector loading from the rest")
ap.add_argument("--zero-matmul", action="store_true",
                help="zero cb and prev_states so the output IS the D residual "
                     "(x*D); isolates the residual path from the matmuls")
ap.add_argument("--index-x", action="store_true",
                help="set x[b,h,:,:] = h so the value identifies which head was read")
ap.add_argument("--index-d", action="store_true",
                help="set D[h] = h so the value identifies which head was read")
ap.add_argument("--unit-d", action="store_true",
                help="set D=1 so the residual is plain x; separates a bad D read "
                     "from a bad x read in the fused-D kernel")
ap.add_argument("--prev-states-transposed", action="store_true", default=True)
ap.add_argument("--no-prev-states-transposed", dest="prev_states_transposed", action="store_false")
a = ap.parse_args()

# Derive shapes from the func signature rather than hardcoding:
#   arg0 cb [B,NC,G,CS,CS]   arg1 x [B,H,S,Dh]   arg5 C [B,G,S,ds]
import re as _re
_sig = _re.search(r"func\.func @[^(]*\(([^)]*)\)", Path(a.ir).read_text()).group(1)
_sh = [[int(d) for d in m.group(1).split("x")]
       for m in _re.finditer(r"memref<([0-9x]+)x(?:f16|bf16|f32)>", _sig)]
B, NC, G, CS = _sh[0][0], _sh[0][1], _sh[0][2], _sh[0][3]
# arg1 is x: rank 4 [B,H,S,Dh] flat-head, rank 5 [B,G,hpg,S,Dh] group-major.
SPLIT = len(_sh[1]) == 5
if SPLIT:
    HPG, S, Dh = _sh[1][2], _sh[1][3], _sh[1][4]
    H = _sh[1][1] * HPG
else:
    HPG = None
    H, S, Dh = _sh[1][1], _sh[1][2], _sh[1][3]
ds = _sh[5][3]
print(f"derived B={B} H={H} S={S} Dh={Dh} G={G} ds={ds} CS={CS} NC={NC} "
      f"layout={'group-major' if SPLIT else 'flat-head'}")

hp = Path(a.kernels) / "host_ttnn.py"
spec = importlib.util.spec_from_file_location("mamba_host", hp)
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
import re
text = hp.read_text()
nx = int(re.search(r"^\s*end_core_x = (\d+)$", text, re.M).group(1)) + 1
ny = int(re.search(r"^\s*end_core_y = (\d+)$", text, re.M).group(1)) + 1

torch.manual_seed(0)
# modest magnitudes: the kernel exponentiates dA differences
cb = torch.randn(B, NC, G, CS, CS, dtype=torch.float16) * 0.1
x = torch.randn(B, S, H, Dh, dtype=torch.float16) * 0.5
if a.index_x:
    x = torch.arange(H, dtype=torch.float16).view(1, 1, H, 1).expand(B, S, H, Dh).contiguous()
dt = torch.randn(B, H, NC, CS, dtype=torch.float16) * 0.1
dA = torch.randn(B, H, NC, CS, dtype=torch.float16) * 0.1
if a.const_vectors:
    dt = torch.full_like(dt, 0.3)
    dA = torch.full_like(dA, 0.2)
Cm = torch.randn(B, S, G, ds, dtype=torch.float16) * 0.1
ps = torch.randn(B, NC, H, Dh, ds, dtype=torch.float16) * 0.1
D = (torch.arange(H, dtype=torch.float16) if a.index_d else
     torch.ones(H, dtype=torch.float16) if a.unit_d else torch.randn(H, dtype=torch.float16))
if a.zero_matmul:
    cb = torch.zeros_like(cb)
    ps = torch.zeros_like(ps)

dev = ttnn.open_device(device_id=0)
try:
    g = dev.compute_with_storage_grid_size()
    print(f"grid {g.x}x{g.y}  kernel needs {nx}x{ny}")
    if nx > g.x or ny > g.y:
        sys.exit("SKIP: grid too small")
    mk = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    xD = x * D.view(1, 1, H, 1)
    psT = ps.transpose(3, 4) if a.prev_states_transposed else ps
    # group-major splits the head axis in the layout: h = g*hpg + hh, so the
    # split is a plain contiguous reshape at whichever position H sits.
    def sp(t, dim):
        if not SPLIT:
            return t
        sh = list(t.shape)
        assert sh[dim] == H, f"expected head axis {H} at dim {dim}, got {sh}"
        return t.reshape(*sh[:dim], G, HPG, *sh[dim + 1:])
    args = [
        mk(cb),
        mk(sp(x.transpose(1, 2), 1)),
        mk(sp(dt.reshape(B, H, NC, 1, CS), 1)),
        mk(sp(dA.reshape(B, H, NC, CS, 1), 1)),
        mk(sp(dA.reshape(B, H, NC, 1, CS), 1)),
        mk(Cm.transpose(1, 2)),
    ]
    # args 6/7 are prev_states_T plus either xD or, when D is folded into the
    # kernel, the per-head scale D itself ([G, hpg, 1, Dh], the only rank-4 arg
    # whose leading dim is G). prev_states_T is the one carrying NC on axis 1.
    # 8 memrefs means neither xD nor D is passed: the residual is plain x, so
    # the reference must be taken with D == 1 (pass --unit-d).
    NO_SCALE = len(_sh) == 8
    # D is [B,G,hpg,NC,1,Dh] (rank 6, unit second-to-last) -- the shape family
    # dt_k uses, which takes the vector read path rather than whole-tile reads.
    FOLD_D = (not NO_SCALE) and _sh[7][-2] == 1 and _sh[7][-1] == Dh and len(_sh[7]) in (5, 6)
    # Flat fused kernel puts D at arg6 as [H, 1, Dh]: a bare [H] slices to a
    # rank-0 memref, which binds neither data-movement symbol and fails to
    # compile, so the unit dim is required.
    # D at arg6: [H,1,Dh] (broadcast form) or [H,CS,Dh] (pre-replicated form)
    # D at arg6: [H,1,Dh], [H,CS,Dh], or [B,H,CS,Dh] (pre-replicated w/ batch)
    FOLD_D_AT6 = (not NO_SCALE) and len(_sh[6]) in (3, 4) \
        and _sh[6][-1] == Dh and _sh[6][-2] in (1, CS) \
        and (_sh[6][0] == H or (len(_sh[6]) == 4 and _sh[6][0] == B and _sh[6][1] == H))
    # group-major fused: D at arg7 as [B,G,hpg,CS,Dh], pre-replicated
    FOLD_D_GM = (not NO_SCALE) and SPLIT and len(_sh[7]) == 5 \
        and _sh[7][-1] == Dh and _sh[7][-2] == CS
    _ps_t = mk(sp(psT, 2))
    if FOLD_D_GM:
        _d = D.reshape(1, G, H // G, 1, 1).expand(B, G, H // G, CS, Dh)
        args += [_ps_t, mk(_d)]
    elif FOLD_D_AT6:
        _shape = tuple(_sh[6])
        if _shape[0] == B and len(_shape) == 4:      # [B,H,CS,Dh]
            _d = D.reshape(1, H, 1, 1).expand(*_shape)
        else:                                        # [H,...,Dh]
            _d = D.reshape(H, *([1] * (len(_shape) - 1))).expand(*_shape)
        args += [mk(_d), _ps_t]
    elif NO_SCALE:
        args += [_ps_t]
    elif FOLD_D:
        if SPLIT:   # [B,G,hpg,NC,1,Dh]
            _d = D.reshape(1, G, H // G, 1, 1, 1).expand(B, G, H // G, NC, 1, Dh)
        else:       # [B,H,NC,1,Dh]
            _d = D.reshape(1, H, 1, 1, 1).expand(B, H, NC, 1, Dh)
        args += [_ps_t, mk(_d)]
    else:
        _xD_t = mk(sp(xD.transpose(1, 2), 1))
        args += [_ps_t, _xD_t] if _sh[6][1] == NC else [_xD_t, _ps_t]
    args.append(mk(sp(torch.zeros(B, H, S, Dh, dtype=torch.float16), 1)))
    out = h.run(*args)
    out = out if not isinstance(out, tuple) else out[0]
    got = ttnn.to_torch(out).float()
    if SPLIT:                      # [B,G,hpg,S,Dh] -> [B,H,S,Dh] for comparison
        got = got.reshape(B, H, S, Dh)

    ref = mamba_ref(cb, x, dt, dA, Cm, ps, D, block_m=CS, causal=a.causal)
    msk = torch.isfinite(got) & torch.isfinite(ref)
    pcc = torch.corrcoef(torch.stack([got[msk].flatten(), ref[msk].flatten()]))[0, 1].item()
    denom = ref.abs().max().clamp_min(1e-6)
    print(f"  causal={a.causal}  prev_states_transposed={a.prev_states_transposed}")
    print(f"  shapes got={tuple(got.shape)} ref={tuple(ref.shape)}")
    def _st(t, nm):
        f = t.flatten().float()
        print(f"  {nm}: mean {f.mean():+.5f}  std {f.std():.5f}  absmax {f.abs().max():.5f}  "
              f"zeros {(f==0).float().mean()*100:.1f}%  nonfinite {(~torch.isfinite(f)).sum().item()}")
    _st(got, "got"); _st(ref, "ref")
    if a.index_x:
        print("  got[0, h, 0, 0] all h:",
              [int(got[0, h, 0, 0]) for h in range(H)])
        print("  got[0, h, 0, 64] all h:",
              [int(got[0, h, 0, 64]) for h in range(H)])
        print("  got[0, h, 192, 0] all h:",
              [int(got[0, h, 192, 0]) for h in range(H)])
        print("  got[0, 0, s, 0] for s=0,192,384,..:",
              [round(float(got[0, 0, s, 0]), 1) for s in range(0, S, CS)][:10])
    # per-(b,h) correlation: a subset being right localises the fault
    gg = got.reshape(got.shape[0]*got.shape[1], -1).float()
    rr = ref.reshape(ref.shape[0]*ref.shape[1], -1).float()
    cc = [torch.corrcoef(torch.stack([gg[i], rr[i]]))[0,1].item() for i in range(gg.shape[0])]
    import statistics as _s
    good = [i for i,c in enumerate(cc) if c > 0.9]
    print(f"  per-(b,h) PCC: min {min(cc):.3f} median {_s.median(cc):.3f} max {max(cc):.3f}; "
          f"{len(good)}/{len(cc)} rows >0.9")
    weak = sorted(((c, i) for i, c in enumerate(cc) if c <= 0.9))[:6]
    if weak:
        H_ = got.shape[1]
        print("  weakest rows (b,h): " + ", ".join(
            f"({i // H_},{i % H_})={c:.3f}" for c, i in weak))
    print(f"  PCC {pcc:.6f}   max|err|/max|ref| {((got-ref).abs().max()/denom).item():.4f}")
    print("  PASS" if pcc > 0.99 else "  FAIL")
finally:
    ttnn.close_device(dev)
