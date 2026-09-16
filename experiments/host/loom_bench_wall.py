#!/usr/bin/env python
"""Wall-clock benchmark: times h.run() which REBUILDS the program each call.
Slower and includes ~1ms python overhead, but valid for kernels whose captured
ProgramDescriptor cannot be safely re-dispatched (cross-core reductions)."""
import argparse, importlib.util, re, statistics as st, time
from pathlib import Path
import torch, ttnn
ap=argparse.ArgumentParser(); ap.add_argument("--ir",required=True); ap.add_argument("--kernels",required=True)
ap.add_argument("--iters",type=int,default=8); ap.add_argument("--warmup",type=int,default=2)
a=ap.parse_args()
hp=Path(a.kernels)/"host_ttnn.py"; text=hp.read_text()
func=re.search(r"^run = (\w+)_ttnn$",text,re.M).group(1)
sig=re.search(rf"func\.func @{re.escape(func)}\(([^)]*)\)",Path(a.ir).read_text()).group(1)
shapes=[[int(d) for d in m.group(1).split("x")] for m in re.finditer(r"memref<([0-9x]+)x(?:f16|bf16|f32)>",sig)]
nx=int(re.search(r"^\s*end_core_x = (\d+)$",text,re.M).group(1))+1
ny=int(re.search(r"^\s*end_core_y = (\d+)$",text,re.M).group(1))+1
spec=importlib.util.spec_from_file_location("lh",hp); h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
(M,K),(K2,N)=shapes[0],shapes[1]; flops=2*M*N*K
d=ttnn.open_device(device_id=0)
try:
    g=d.compute_with_storage_grid_size()
    if nx>g.x or ny>g.y: raise SystemExit(f"SKIP {nx}x{ny} > {g.x}x{g.y}")
    torch.manual_seed(0)
    mk=lambda s,z=False: ttnn.from_torch(torch.zeros(s) if z else torch.randn(s),
        dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=d,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    args=[mk(s,z=(n in set(h._OUTPUT_PARAM_ORDER))) for n,s in zip(h._PARAM_ORDER,shapes)]
    def once(): h.run(*args); ttnn.synchronize_device(d)
    for _ in range(a.warmup): once()
    ts=[]
    for _ in range(a.iters):
        t0=time.perf_counter(); once(); ts.append(time.perf_counter()-t0)
    t=st.median(ts)
    ck=ttnn.WormholeComputeKernelConfig(packer_l1_acc=True)
    def ref(): ttnn.matmul(args[0],args[1],compute_kernel_config=ck); ttnn.synchronize_device(d)
    for _ in range(a.warmup): ref()
    rs=[]
    for _ in range(a.iters):
        t0=time.perf_counter(); ref(); rs.append(time.perf_counter()-t0)
    tr=st.median(rs)
    print(f"WALL {M}x{N}x{K} loom_ms={t*1e3:.3f} loom_tfs={flops/t/1e12:.2f} ttnn_ms={tr*1e3:.3f} ttnn_tfs={flops/tr/1e12:.2f}")
finally:
    ttnn.close_device(d)
