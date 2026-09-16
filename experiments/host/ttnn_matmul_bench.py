#!/usr/bin/env python
"""Standalone ttnn.matmul benchmark - no Loom required.

  python ttnn_matmul_bench.py --m 16384 --n 16384 --k 16384 --iters 15
"""
import argparse, statistics as st, time
import torch, ttnn

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--iters", type=int, default=15)
ap.add_argument("--warmup", type=int, default=3)
a = ap.parse_args()
flops = 2 * a.m * a.n * a.k

d = ttnn.open_device(device_id=0)
try:
    g = d.compute_with_storage_grid_size()
    print(f"grid {g.x}x{g.y}   {a.m}x{a.n}x{a.k}  ({flops/1e9:.1f} GFLOP)  "
          f"bf16 TILE DRAM  iters={a.iters}")
    torch.manual_seed(0)
    mk = lambda *s: ttnn.from_torch(torch.randn(*s, dtype=torch.float16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
    A, B = mk(a.m, a.k), mk(a.k, a.n)

    def bench(fn):
        for _ in range(a.warmup): fn(); ttnn.synchronize_device(d)
        ts = []
        for _ in range(a.iters):
            t0 = time.perf_counter(); fn(); ttnn.synchronize_device(d)
            ts.append(time.perf_counter() - t0)
        return ts

    cfgs = [("default (no compute_kernel_config)", None)]
    for f in ["LoFi", "HiFi2", "HiFi4"]:
        cfgs.append((f"{f} + packer_l1_acc",
                     ttnn.WormholeComputeKernelConfig(
                         math_fidelity=getattr(ttnn.MathFidelity, f), packer_l1_acc=True)))
    for name, ck in cfgs:
        fn = (lambda: ttnn.matmul(A, B)) if ck is None else (lambda ck=ck: ttnn.matmul(A, B, compute_kernel_config=ck))
        ts = bench(fn)
        tf = [flops / t / 1e12 for t in ts]
        print(f"  {name:<36} mean {st.mean(tf):7.2f} TFLOP/s  sd {st.pstdev(tf):5.2f}  "
              f"median {st.median(tf):7.2f}  [{st.mean(ts)*1e3:.3f} ms]")
finally:
    ttnn.close_device(d)
