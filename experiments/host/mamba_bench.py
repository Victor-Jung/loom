#!/usr/bin/env python
"""Device-time benchmark for a Loom-generated mamba_chunk_scan kernel.

Captures the ProgramDescriptor that h.run() builds and dispatches it directly,
so the python rebuild cost never enters the measurement. Prints the core grid
the kernel occupies, which is the point of comparison between variants.
"""
import argparse, importlib.util, re, statistics as st, time
from pathlib import Path
import torch, ttnn

ap = argparse.ArgumentParser()
ap.add_argument("--ir", required=True)
ap.add_argument("--kernels", required=True)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--warmup", type=int, default=5)
a = ap.parse_args()

host_py = Path(a.kernels) / "host_ttnn.py"
text = host_py.read_text()
func = re.search(r"^run = (\w+)_ttnn$", text, re.M).group(1)
sig = re.search(rf"func\.func @{re.escape(func)}\(([^)]*)\)", Path(a.ir).read_text()).group(1)
shapes = [[int(d) for d in m.group(1).split("x")]
          for m in re.finditer(r"memref<([0-9x]+)x(?:f16|bf16|f32)>", sig)]
need_x = int(re.search(r"^\s*end_core_x = (\d+)$", text, re.M).group(1)) + 1
need_y = int(re.search(r"^\s*end_core_y = (\d+)$", text, re.M).group(1)) + 1
spec = importlib.util.spec_from_file_location("loom_host_ttnn", host_py)
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)

d = ttnn.open_device(device_id=0)
try:
    g = d.compute_with_storage_grid_size()
    if need_x > g.x or need_y > g.y:
        raise SystemExit(f"SKIP grid {need_x}x{need_y} > device {g.x}x{g.y}")
    torch.manual_seed(0)
    mk = lambda s, z=False: ttnn.from_torch(
        torch.zeros(s) if z else torch.randn(s) * 0.1,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d,
        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    outs = set(h._OUTPUT_PARAM_ORDER)
    args = [mk(s, z=(n in outs)) for n, s in zip(h._PARAM_ORDER, shapes)]

    cap = {}
    real = ttnn.generic_op
    def capture(io, prog): cap['io'], cap['prog'] = io, prog; return real(io, prog)
    ttnn.generic_op = capture
    h.run(*args); ttnn.synchronize_device(d)
    ttnn.generic_op = real
    if 'prog' not in cap:
        raise SystemExit("failed to capture ProgramDescriptor")

    def bench(fn, iters, warmup):
        for _ in range(warmup): fn(); ttnn.synchronize_device(d)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); fn(); ttnn.synchronize_device(d)
            ts.append(time.perf_counter() - t0)
        return st.median(ts), st.pstdev(ts)

    t_dev, s_dev = bench(lambda: ttnn.generic_op(cap['io'], cap['prog']), a.iters, a.warmup)
    cores = need_x * need_y
    print(f"GRID {need_x}x{need_y} = {cores} cores   "
          f"device {t_dev*1e3:9.3f} ms  (sd {s_dev*1e3:.3f})   "
          f"ms*cores {t_dev*1e3*cores:9.1f}")
finally:
    ttnn.close_device(d)
