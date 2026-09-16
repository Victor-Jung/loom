#!/usr/bin/env python
"""16384^3 GEMM: full timing distribution, Loom vs ttnn at several configs."""
import argparse, importlib.util, re, statistics as st, time
from pathlib import Path
import torch, ttnn
ap=argparse.ArgumentParser(); ap.add_argument("--ir",required=True); ap.add_argument("--kernels",required=True)
ap.add_argument("--iters",type=int,default=15); ap.add_argument("--warmup",type=int,default=3)
a=ap.parse_args()
hp=Path(a.kernels)/"host_ttnn.py"; text=hp.read_text()
func=re.search(r"^run = (\w+)_ttnn$",text,re.M).group(1)
sig=re.search(rf"func\.func @{re.escape(func)}\(([^)]*)\)",Path(a.ir).read_text()).group(1)
shapes=[[int(d) for d in m.group(1).split("x")] for m in re.finditer(r"memref<([0-9x]+)x(?:f16|bf16|f32)>",sig)]
spec=importlib.util.spec_from_file_location("lh",hp); h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
(M,K),(K2,N)=shapes[0],shapes[1]; flops=2*M*N*K
def stats(ts,label):
    tf=[flops/t/1e12 for t in ts]
    print(f"  {label:<26} mean {st.mean(tf):7.2f}  median {st.median(tf):7.2f}  "
          f"sd {st.pstdev(tf):5.2f}  min {min(tf):7.2f}  max {max(tf):7.2f}  "
          f"[{st.mean(ts)*1e3:.1f} ms]")
    return st.mean(tf)
d=ttnn.open_device(device_id=0)
try:
    g=d.compute_with_storage_grid_size(); print(f"grid {g.x}x{g.y}   shape {M}x{N}x{K}  ({flops/1e9:.0f} GFLOP)  iters={a.iters}")
    torch.manual_seed(0)
    mk=lambda s,z=False: ttnn.from_torch(torch.zeros(s) if z else torch.randn(s,dtype=torch.float16),
        dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=d,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    args=[mk(s,z=(n in set(h._OUTPUT_PARAM_ORDER))) for n,s in zip(h._PARAM_ORDER,shapes)]
    cap={}; real=ttnn.generic_op
    def cptr(io,prog): cap['io'],cap['prog']=io,prog; return real(io,prog)
    ttnn.generic_op=cptr; h.run(*args); ttnn.synchronize_device(d); ttnn.generic_op=real
    def timeit(fn):
        for _ in range(a.warmup): fn(); ttnn.synchronize_device(d)
        ts=[]
        for _ in range(a.iters):
            t0=time.perf_counter(); fn(); ttnn.synchronize_device(d); ts.append(time.perf_counter()-t0)
        return ts
    lm=stats(timeit(lambda: ttnn.generic_op(cap['io'],cap['prog'])),"loom (kernel)")
    cfgs=[("ttnn default",None),
          ("ttnn LoFi+l1acc",ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,packer_l1_acc=True)),
          ("ttnn HiFi2+l1acc",ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,packer_l1_acc=True)),
          ("ttnn HiFi4+l1acc",ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,packer_l1_acc=True))]
    for name,ck in cfgs:
        f=(lambda: ttnn.matmul(args[0],args[1])) if ck is None else (lambda ck=ck: ttnn.matmul(args[0],args[1],compute_kernel_config=ck))
        tm=stats(timeit(f),name)
        print(f"      -> loom/{name.replace('ttnn ','')} = {lm/tm:.3f}")
finally:
    ttnn.close_device(d)
