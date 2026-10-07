#!/usr/bin/env python
"""Run and verify the hand-written TTNN mamba chunk-scan baseline.

kernels/mamba_baseline.py had no driver and had never been executed. It is the
reference definition of the computation -- notably it applies the D residual
(x * D[h]) on device, inside the scan, which the Loom kernels must match.

  python -u experiments/host/mamba_baseline_run.py --bench
"""
import argparse, sys, time, statistics as st
from pathlib import Path
import torch, ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mamba_ref import mamba_ref
from kernels.mamba_baseline import (  # noqa: E402
    prepare_mamba2_chunk_scan_inputs, ttnn_mamba2_chunk_scan_compute,
)

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=2);   ap.add_argument("--L", type=int, default=1920)
ap.add_argument("--H", type=int, default=32);  ap.add_argument("--Dh", type=int, default=128)
ap.add_argument("--G", type=int, default=4);   ap.add_argument("--ds", type=int, default=128)
ap.add_argument("--CS", type=int, default=192)
ap.add_argument("--causal", default="element", choices=["none", "block", "element"])
ap.add_argument("--bench", action="store_true")
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--fidelity", default="HiFi4", choices=["LoFi", "HiFi2", "HiFi3", "HiFi4"])
ap.add_argument("--no-approx", action="store_true")
a = ap.parse_args()
B, L, H, Dh, G, ds, CS = a.B, a.L, a.H, a.Dh, a.G, a.ds, a.CS
NC = L // CS
print(f"B={B} L={L} H={H} Dh={Dh} G={G} ds={ds} CS={CS} NC={NC}")

torch.manual_seed(0)
cb = torch.randn(B, NC, G, CS, CS, dtype=torch.float16) * 0.1
x  = torch.randn(B, L, H, Dh, dtype=torch.float16) * 0.5
dt = torch.randn(B, H, NC, CS, dtype=torch.float16) * 0.1
dA = torch.randn(B, H, NC, CS, dtype=torch.float16) * 0.1
Cm = torch.randn(B, L, G, ds, dtype=torch.float16) * 0.1
ps = torch.randn(B, NC, H, Dh, ds, dtype=torch.float16) * 0.1
D  = torch.randn(H, dtype=torch.float16)

dev = ttnn.open_device(device_id=0)
try:
    prepped = prepare_mamba2_chunk_scan_inputs(cb, x, dt, dA, Cm, ps, D)
    mk = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cb_tt, x_tt, dt_tt, dA_tt, C_tt, prev_tt, D_host = (
        mk(prepped[0]), mk(prepped[1]), mk(prepped[2]), mk(prepped[3]),
        mk(prepped[4]), mk(prepped[5]), prepped[6])
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, a.fidelity),
        math_approx_mode=not a.no_approx, fp32_dest_acc_en=False, packer_l1_acc=True)
    print(f"PE kernel: fidelity {a.fidelity} approx {not a.no_approx} fp32_acc False packer_l1_acc True")
    kw = dict(batch=B, nchunks=NC, ngroups=G, chunk_size=CS, seqlen=L,
              nheads=H, headdim=Dh, dstate=ds, compute_kernel_config=ckc)

    t0 = time.perf_counter()
    outs = ttnn_mamba2_chunk_scan_compute(cb_tt, x_tt, dt_tt, dA_tt, C_tt,
                                          prev_tt, D_host, **kw)
    ttnn.synchronize_device(dev)
    print(f"ran: {len(outs)} output tiles in {(time.perf_counter()-t0)*1e3:.1f} ms")

    got = torch.zeros(B, H, L, Dh)
    i = 0
    for b in range(B):
        for h in range(H):
            for c in range(NC):
                got[b, h, c*CS:(c+1)*CS, :] = ttnn.to_torch(outs[i]).reshape(CS, Dh).float()
                i += 1
    ref = mamba_ref(cb, x, dt, dA, Cm, ps, D, block_m=CS, causal=a.causal)
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    print(f"  got std {got.std():.5f}  ref std {ref.std():.5f}")
    print(f"  PCC {pcc:.6f}   {'PASS' if pcc > 0.99 else 'FAIL'}")

    if a.bench:
        def once():
            ttnn_mamba2_chunk_scan_compute(cb_tt, x_tt, dt_tt, dA_tt, C_tt,
                                           prev_tt, D_host, **kw)
            ttnn.synchronize_device(dev)
        once()
        ts = []
        for _ in range(a.iters):
            t0 = time.perf_counter(); once(); ts.append(time.perf_counter() - t0)
        print(f"  ttnn baseline: median {st.median(ts)*1e3:.3f} ms  (n={a.iters})")
finally:
    ttnn.close_device(dev)
