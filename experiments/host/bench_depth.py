#!/usr/bin/env python
"""Fusion-depth crossover: D = (exp/log)^d (A@B).

Loom fuses the whole epilogue chain into the matmul accumulator (one kernel).
ttnn must run matmul + d separate elementwise kernels, each a full DRAM
round-trip of the M*N intermediate.
"""
import argparse, importlib.util, re, statistics as st, time
from pathlib import Path
import torch, ttnn
ap=argparse.ArgumentParser()
ap.add_argument("--ir",required=True); ap.add_argument("--kernels",required=True)
ap.add_argument("--depth",type=int,required=True)
ap.add_argument("--iters",type=int,default=20); ap.add_argument("--warmup",type=int,default=5)
a=ap.parse_args()
hp=Path(a.kernels)/"host_ttnn.py"; text=hp.read_text()
func=re.search(r"^run = (\w+)_ttnn$",text,re.M).group(1)
sig=re.search(rf"func\.func @{re.escape(func)}\(([^)]*)\)",Path(a.ir).read_text()).group(1)
shapes=[[int(d) for d in m.group(1).split("x")] for m in re.finditer(r"memref<([0-9x]+)x(?:f16|bf16|f32)>",sig)]
spec=importlib.util.spec_from_file_location("lh",hp); h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
(M,K),(K2,N)=shapes[0],shapes[1]
d=ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    mk=lambda t: ttnn.from_torch(t,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,
                                 device=d,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    At=torch.randn(M,K,dtype=torch.float16)*0.1; Bt=torch.randn(K,N,dtype=torch.float16)*0.1
    A,B=mk(At),mk(Bt); OUT=mk(torch.zeros(M,N,dtype=torch.float16))
    ck=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,packer_l1_acc=True)
    def loom(): h.run(A,B,OUT)
    def ttnn_chain():
        c=ttnn.matmul(A,B,compute_kernel_config=ck)
        for i in range(a.depth):
            c = ttnn.exp(c) if i%2==0 else ttnn.log(c)
        return c
    # correctness
    h.run(A,B,OUT); ttnn.synchronize_device(d)
    ref=(At.float()@Bt.float())
    for i in range(a.depth): ref = torch.exp(ref) if i%2==0 else torch.log(ref)
    got=ttnn.to_torch(OUT).float()
    msk=torch.isfinite(ref)&torch.isfinite(got)
    pcc=torch.corrcoef(torch.stack([got[msk].flatten(),ref[msk].flatten()]))[0,1].item()
    def bench(fn):
        for _ in range(a.warmup): fn(); ttnn.synchronize_device(d)
        ts=[]
        for _ in range(a.iters):
            t0=time.perf_counter(); fn(); ttnn.synchronize_device(d); ts.append(time.perf_counter()-t0)
        return st.median(ts)
    tl=bench(loom); tt=bench(ttnn_chain)
    print(f"depth={a.depth:2d}  loom {tl*1e3:8.3f} ms   ttnn {tt*1e3:8.3f} ms   "
          f"loom/ttnn {tl/tt:5.2f}x   {'LOOM WINS' if tl<tt else ''}   PCC {pcc:.4f}")
finally:
    ttnn.close_device(d)
