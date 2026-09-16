#!/usr/bin/env python
"""Benchmark a Loom-generated matmul against ttnn.matmul.

Device time is measured by capturing the ProgramDescriptor that h.run() builds,
then dispatching it directly in a loop -- no subtraction, so it stays valid even
when the Python program-build cost dominates wall clock.
"""
import argparse, importlib.util, re, statistics as st, time
from pathlib import Path
import torch, ttnn

ap = argparse.ArgumentParser()
ap.add_argument("--ir", required=True); ap.add_argument("--kernels", required=True)
ap.add_argument("--iters", type=int, default=20); ap.add_argument("--warmup", type=int, default=5)
a = ap.parse_args()

host_py = Path(a.kernels)/"host_ttnn.py"; text = host_py.read_text()
func = re.search(r"^run = (\w+)_ttnn$", text, re.M).group(1)
sig = re.search(rf"func\.func @{re.escape(func)}\(([^)]*)\)", Path(a.ir).read_text()).group(1)
shapes = [[int(d) for d in m.group(1).split("x")]
          for m in re.finditer(r"memref<([0-9x]+)x(?:f16|bf16|f32)>", sig)]
need_x = int(re.search(r"^\s*end_core_x = (\d+)$", text, re.M).group(1))+1
need_y = int(re.search(r"^\s*end_core_y = (\d+)$", text, re.M).group(1))+1
spec = importlib.util.spec_from_file_location("loom_host_ttnn", host_py)
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
(M,K),(K2,N) = shapes[0], shapes[1]; flops = 2*M*N*K

def bench(fn, iters, warmup, dev):
    for _ in range(warmup): fn(); ttnn.synchronize_device(dev)
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter(); fn(); ttnn.synchronize_device(dev); ts.append(time.perf_counter()-t0)
    return st.median(ts)

d = ttnn.open_device(device_id=0)
try:
    g = d.compute_with_storage_grid_size()
    if need_x>g.x or need_y>g.y: raise SystemExit(f"SKIP grid {need_x}x{need_y} > {g.x}x{g.y}")
    torch.manual_seed(0)
    mk = lambda s,z=False: ttnn.from_torch(torch.zeros(s) if z else torch.randn(s),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    args=[mk(s, z=(n in set(h._OUTPUT_PARAM_ORDER))) for n,s in zip(h._PARAM_ORDER, shapes)]

    cap={}
    real = ttnn.generic_op
    def capture(io, prog): cap['io'],cap['prog']=io,prog; return real(io,prog)
    ttnn.generic_op = capture
    h.run(*args); ttnn.synchronize_device(d)          # build once, capture descriptor
    ttnn.generic_op = real
    if 'prog' not in cap: raise SystemExit("failed to capture ProgramDescriptor")

    t_dev  = bench(lambda: ttnn.generic_op(cap['io'], cap['prog']), a.iters, a.warmup, d)
    t_wall = bench(lambda: h.run(*args), max(a.iters//2,3), 2, d)
    t_ref  = bench(lambda: ttnn.matmul(args[0], args[1]), a.iters, a.warmup, d)

    tf=lambda t:(flops/t/1e12) if t>0 else float('nan')
    print(f"shape {M}x{N}x{K}  grid {need_x}x{need_y}  ({flops/1e9:.1f} GFLOP)")
    print(f"  loom device   {t_dev*1e3:9.3f} ms   {tf(t_dev):8.2f} TFLOP/s   <- kernel (direct dispatch)")
    print(f"  loom wall     {t_wall*1e3:9.3f} ms   {tf(t_wall):8.2f} TFLOP/s   (incl. python rebuild)")
    print(f"  ttnn.matmul   {t_ref*1e3:9.3f} ms   {tf(t_ref):8.2f} TFLOP/s")
    print(f"  loom/ttnn     {t_dev/t_ref:.2f}x slower" if t_dev>t_ref else f"  loom/ttnn     {t_ref/t_dev:.2f}x FASTER")
finally:
    ttnn.close_device(d)
