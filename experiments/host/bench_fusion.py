#!/usr/bin/env python
"""Fusion-depth demo: D = exp(A@B).

  Loom  : ONE kernel, epilogue fused into the matmul accumulator (C never hits DRAM)
  ttnn  : TWO kernels - matmul then exp - because ttnn cannot fuse exp into matmul
          (only relu/gelu/silu are accepted), so C round-trips through DRAM.

ttnn is run at HiFi4 + packer_l1_acc to match Loom's hardcoded HiFi4.
"""
import argparse, importlib.util, re, statistics as st, time
from pathlib import Path
import torch, ttnn

ap=argparse.ArgumentParser()
ap.add_argument("--ir",required=True); ap.add_argument("--kernels",required=True)
ap.add_argument("--iters",type=int,default=30); ap.add_argument("--warmup",type=int,default=8)
a=ap.parse_args()
hp=Path(a.kernels)/"host_ttnn.py"; text=hp.read_text()
func=re.search(r"^run = (\w+)_ttnn$",text,re.M).group(1)
sig=re.search(rf"func\.func @{re.escape(func)}\(([^)]*)\)",Path(a.ir).read_text()).group(1)
shapes=[[int(d) for d in m.group(1).split("x")] for m in re.finditer(r"memref<([0-9x]+)x(?:f16|bf16|f32)>",sig)]
nx=int(re.search(r"^\s*end_core_x = (\d+)$",text,re.M).group(1))+1
ny=int(re.search(r"^\s*end_core_y = (\d+)$",text,re.M).group(1))+1
spec=importlib.util.spec_from_file_location("lh",hp); h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
(M,K),(K2,N)=shapes[0],shapes[1]; flops=2*M*N*K
inter_mb = M*N*2/1e6

d=ttnn.open_device(device_id=0)
try:
    g=d.compute_with_storage_grid_size()
    print(f"grid {g.x}x{g.y}  D=exp(A@B)  {M}x{N}x{K}  ({flops/1e9:.2f} GFLOP, "
          f"intermediate C = {inter_mb:.0f} MB)  kernel needs {nx}x{ny}")
    if nx>g.x or ny>g.y: raise SystemExit("SKIP: grid too small")
    torch.manual_seed(0)
    mk=lambda t: ttnn.from_torch(t,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,
                                 device=d,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # scale inputs so exp() does not overflow bf16
    At=torch.randn(M,K,dtype=torch.float16)*0.1
    Bt=torch.randn(K,N,dtype=torch.float16)*0.1
    A,B=mk(At),mk(Bt); OUT=mk(torch.zeros(M,N,dtype=torch.float16))
    ck=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,packer_l1_acc=True)

    def loom(): h.run(A,B,OUT)
    def ttnn_two():
        c=ttnn.matmul(A,B,compute_kernel_config=ck); return ttnn.exp(c)
    def ttnn_mm_only():
        return ttnn.matmul(A,B,compute_kernel_config=ck)

    # correctness
    h.run(A,B,OUT); ttnn.synchronize_device(d)
    got=ttnn.to_torch(OUT).float(); ref=torch.exp((At.float()@Bt.float()))
    pcc=torch.corrcoef(torch.stack([got.flatten(),ref.flatten()]))[0,1].item()
    print(f"  Loom correctness vs torch exp(A@B): PCC {pcc:.6f}  {'PASS' if pcc>0.99 else 'FAIL'}")

    def bench(fn):
        for _ in range(a.warmup): fn(); ttnn.synchronize_device(d)
        ts=[]
        for _ in range(a.iters):
            t0=time.perf_counter(); fn(); ttnn.synchronize_device(d); ts.append(time.perf_counter()-t0)
        return st.median(ts), st.pstdev(ts)
    for name,fn in [("Loom fused (1 kernel)",loom),
                    ("ttnn matmul+exp (2 kernels)",ttnn_two),
                    ("ttnn matmul only (reference)",ttnn_mm_only)]:
        t,sd=bench(fn)
        print(f"  {name:<30} {t*1e3:8.3f} ms  (sd {sd*1e3:.3f})")
finally:
    ttnn.close_device(d)
