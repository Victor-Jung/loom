#!/usr/bin/env python
"""Run any Loom-generated ttnn kernel on a Tenstorrent device.

Tensor shapes are read from the bufferized MLIR's func signature (authoritative),
not guessed. Numerical validation is only possible where we know the reference
semantics (matmul); otherwise we report that it ran and sanity-check the output.

  python loom_run_generic.py --ir <p03_bufferized.mlir> [--kernels DIR] [--ref matmul]
"""
import argparse, importlib.util, re, sys
from pathlib import Path
import torch, ttnn

ap = argparse.ArgumentParser()
ap.add_argument("--ir", required=True, help="p03_bufferized.mlir the kernels were lowered from")
ap.add_argument("--kernels", default="/home/vicjung/loom/tmp_output/kernels")
ap.add_argument("--ref", choices=["matmul", "none"], default="none")
a = ap.parse_args()

host_py = Path(a.kernels) / "host_ttnn.py"
text = host_py.read_text()

wrapper = re.search(r"^run = (\w+)_ttnn$", text, re.M)
if not wrapper:
    sys.exit("could not find 'run = ..._ttnn' in host_ttnn.py")
func_name = wrapper.group(1)

# Pull the memref arg shapes from the matching func.func in the MLIR.
ir = Path(a.ir).read_text()
sig = re.search(rf"func\.func @{re.escape(func_name)}\(([^)]*)\)", ir)
if not sig:
    sys.exit(f"func @{func_name} not found in {a.ir}")
shapes, dtypes = [], []
for m in re.finditer(r"memref<([0-9x]+)x(f16|bf16|f32)>", sig.group(1)):
    shapes.append([int(d) for d in m.group(1).split("x")])
    dtypes.append(m.group(2))
if not shapes:
    sys.exit("no memref args parsed from func signature")

need_x = (int(re.search(r"^\s*end_core_x = (\d+)$", text, re.M).group(1)) + 1)
need_y = (int(re.search(r"^\s*end_core_y = (\d+)$", text, re.M).group(1)) + 1)

spec = importlib.util.spec_from_file_location("loom_host_ttnn", host_py)
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
order   = h._PARAM_ORDER
outputs = set(h._OUTPUT_PARAM_ORDER)

print(f"func    : {func_name[:70]}")
print(f"tensors : {list(zip(order, shapes))}")
print(f"cores   : needs {need_x}x{need_y}")

d = ttnn.open_device(device_id=0)
try:
    g = d.compute_with_storage_grid_size()
    print(f"device  : {g.x}x{g.y}")
    if need_x > g.x or need_y > g.y:
        sys.exit(f"SKIP: needs {need_x}x{need_y}, device has {g.x}x{g.y}")

    torch.manual_seed(0)
    mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    torch_in, args = {}, {}
    for name, shp in zip(order, shapes):
        t = torch.zeros(shp) if name in outputs else torch.randn(shp)
        torch_in[name] = t
        args[name] = mk(t)

    out = h.run(*[args[n] for n in order])
    outs = out if isinstance(out, tuple) else (out,)
    got = [ttnn.to_torch(o).float() for o in outs]

    if a.ref == "matmul" and len(shapes) == 3:
        x = torch_in[order[0]]; y = torch_in[order[1]]
        ref = x @ y
        pcc = torch.corrcoef(torch.stack([got[0].flatten(), ref.flatten()]))[0, 1].item()
        print(f"PCC     : {pcc:.6f}")
        print("PASS" if pcc > 0.99 else "FAIL")
    else:
        o = got[0]
        finite = bool(torch.isfinite(o).all()); nz = int((o != 0).sum())
        print(f"output  : shape={tuple(o.shape)} finite={finite} nonzero={nz}/{o.numel()}")
        print("RAN (no reference checked)" if finite and nz else "SUSPECT (all-zero or non-finite)")
finally:
    ttnn.close_device(d)
