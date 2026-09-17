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
H, S, Dh = _sh[1][1], _sh[1][2], _sh[1][3]
ds = _sh[5][3]
print(f"derived B={B} H={H} S={S} Dh={Dh} G={G} ds={ds} CS={CS} NC={NC}")

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
dt = torch.randn(B, H, NC, CS, dtype=torch.float16) * 0.1
dA = torch.randn(B, H, NC, CS, dtype=torch.float16) * 0.1
if a.const_vectors:
    dt = torch.full_like(dt, 0.3)
    dA = torch.full_like(dA, 0.2)
Cm = torch.randn(B, S, G, ds, dtype=torch.float16) * 0.1
ps = torch.randn(B, NC, H, Dh, ds, dtype=torch.float16) * 0.1
D = torch.randn(H, dtype=torch.float16)

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
    args = [
        mk(cb),
        mk(x.transpose(1, 2)),
        mk(dt.reshape(B, H, NC, 1, CS)),
        mk(dA.reshape(B, H, NC, CS, 1)),
        mk(dA.reshape(B, H, NC, 1, CS)),
        mk(Cm.transpose(1, 2)),
        mk(xD.transpose(1, 2)),
        mk(psT),
        mk(torch.zeros(B, H, S, Dh, dtype=torch.float16)),
    ]
    out = h.run(*args)
    out = out if not isinstance(out, tuple) else out[0]
    got = ttnn.to_torch(out).float()

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
