#!/usr/bin/env python
"""Loom vs ttnn.matmul at MATCHED math fidelity.

Loom's generated host code uses a bare ttnn.ComputeConfigDescriptor(), which
defaults to HiFi4 (4 matrix-engine passes). ttnn.matmul picks its own default.
Comparing them unmatched measures fidelity, not kernel quality.
"""
import argparse, importlib.util, re, statistics as st, time
from pathlib import Path
import torch, ttnn

ap = argparse.ArgumentParser()
ap.add_argument("--ir", required=True); ap.add_argument("--kernels", required=True)
ap.add_argument("--iters", type=int, default=10); ap.add_argument("--warmup", type=int, default=3)
a = ap.parse_args()

host_py = Path(a.kernels)/"host_ttnn.py"; text = host_py.read_text()
func = re.search(r"^run = (\w+)_ttnn$", text, re.M).group(1)
sig = re.search(rf"func\.func @{re.escape(func)}\(([^)]*)\)", Path(a.ir).read_text()).group(1)
shapes=[[int(d) for d in m.group(1).split("x")] for m in
        re.finditer(r"memref<([0-9x]+)x(?:f16|bf16|f32)>", sig)]
need_x=int(re.search(r"^\s*end_core_x = (\d+)$",text,re.M).group(1))+1
need_y=int(re.search(r"^\s*end_core_y = (\d+)$",text,re.M).group(1))+1
spec=importlib.util.spec_from_file_location("loom_host_ttnn",host_py)
h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
(M,K),(K2,N)=shapes[0],shapes[1]; flops=2*M*N*K

_orig_ccd = ttnn.ComputeConfigDescriptor
def ccd_with(fid):
    def f(*x, **k):
        c=_orig_ccd(*x,**k); c.math_fidelity=fid; return c
    return f

def bench(fn,iters,warmup,dev):
    for _ in range(warmup): fn(); ttnn.synchronize_device(dev)
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter(); fn(); ttnn.synchronize_device(dev); ts.append(time.perf_counter()-t0)
    return st.median(ts)

d=ttnn.open_device(device_id=0)
try:
    g=d.compute_with_storage_grid_size()
    if need_x>g.x or need_y>g.y: raise SystemExit(f"SKIP {need_x}x{need_y} > {g.x}x{g.y}")
    torch.manual_seed(0)
    mk=lambda s,z=False: ttnn.from_torch(torch.zeros(s) if z else torch.randn(s),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    args=[mk(s,z=(n in set(h._OUTPUT_PARAM_ORDER))) for n,s in zip(h._PARAM_ORDER,shapes)]
    tf=lambda t:(flops/t/1e12) if t>0 else float('nan')

    print(f"shape {M}x{N}x{K}  grid {need_x}x{need_y}  ({flops/1e9:.1f} GFLOP)  [bf16]")
    print(f"  {'fidelity':<9}{'loom ms':>10}{'loomTF':>9}{'ttnn ms':>10}{'ttnnTF':>9}{'gap':>9}")
    for fname in ["LoFi","HiFi2","HiFi4"]:
        fid=getattr(ttnn.MathFidelity,fname)
        # Loom: inject fidelity, rebuild + capture descriptor
        ttnn.ComputeConfigDescriptor=ccd_with(fid)
        cap={}; real=ttnn.generic_op
        def capture(io,prog,_c=cap): _c['io'],_c['prog']=io,prog; return real(io,prog)
        ttnn.generic_op=capture
        h.run(*args); ttnn.synchronize_device(d)
        ttnn.generic_op=real; ttnn.ComputeConfigDescriptor=_orig_ccd
        t_loom=bench(lambda: ttnn.generic_op(cap['io'],cap['prog']), a.iters,a.warmup,d)
        # ttnn at the same fidelity
        ck=ttnn.WormholeComputeKernelConfig(math_fidelity=fid)
        t_ref=bench(lambda: ttnn.matmul(args[0],args[1],compute_kernel_config=ck), a.iters,a.warmup,d)
        gap=f"{t_loom/t_ref:.2f}x" if t_loom>=t_ref else f"{t_ref/t_loom:.2f}xF"
        print(f"  {fname:<9}{t_loom*1e3:>10.3f}{tf(t_loom):>9.1f}{t_ref*1e3:>10.3f}{tf(t_ref):>9.1f}{gap:>9}")
    t_def=bench(lambda: ttnn.matmul(args[0],args[1]), a.iters,a.warmup,d)
    print(f"  ttnn default (no config): {t_def*1e3:.3f} ms  {tf(t_def):.1f} TFLOP/s")
finally:
    ttnn.close_device(d)
