#!/usr/bin/env python
"""Compile, lower, run and check a benchmark program, one row per candidate.

Runs on the host. Compilation and lowering happen in the development
container; execution and the golden check happen here. Each candidate
function of the final p03 (one per tuner tree times spatial mapping, cut to
--topk) is lowered and run, and the row records the stage that failed if one
did, the solver's T_total, the PCC against the golden reference, and the
median device time.

    python -u benchmarks/run.py toy_bmm --tune enumerate --topk 2
    python -u benchmarks/run.py row_softmax
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TT_METAL_HOME = Path(os.environ.setdefault("TT_METAL_HOME", "/home/vicjung/tt-metal"))
if str(TT_METAL_HOME) not in sys.path:
    sys.path.insert(0, str(TT_METAL_HOME))
KERNEL_STAGE_DIR = TT_METAL_HOME / "tt_metal/programming_examples/mlir_matmul_simple/kernels"
CONTAINER = ["podman", "run", "--rm",
             "--mount", f"source={ROOT},target=/workspace/loom,type=bind",
             "--security-opt", "label=disable", "--security-opt=seccomp=unconfined",
             "--init", "-w", "/workspace/loom", "ftod/loom_dev:latest", "bash", "-lc"]

from benchmarks.programs import PROGRAMS, Program  # noqa: E402


def container(cmd: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(CONTAINER + [cmd], capture_output=True, text=True, timeout=timeout)


def compile_program(name: str, tag: str, tune: str | None, tune_options: str,
                    topk: int, njobs: int, hoist: bool = False,
                    prog: Program | None = None, per_order: int | None = None) -> tuple[Path, str]:
    prog = prog or PROGRAMS[name]
    out = Path("test/bench") / name / tag
    shape = f"-{prog.shape} " if prog.shape else ""
    tune_args = f"--tune {tune} " if tune else ""
    if tune and tune_options:
        tune_args += f"--tune-options '{tune_options}' "
    if tune and hoist:
        tune_args += "--tune-hoist "
    if per_order:
        tune_args += f"--topk-per-order {per_order} "
    cmd = (f"rm -rf {out} && env -u VIRTUAL_ENV uv run python -u {prog.kernel} {shape}"
           f"--config {prog.config} --output-path {out} --debug --njobs {njobs} "
           f"--topk-candidates {topk} {tune_args}")
    res = container(cmd)
    log = res.stdout + res.stderr
    (ROOT / out).mkdir(parents=True, exist_ok=True)
    (ROOT / out / "compile.log").write_text(log)
    return out, log


def parse_stage_times(log: str) -> dict[str, float]:
    """Pipeline step -> milliseconds, from the compile log's timing summary."""
    times = {}
    for m in re.finditer(r"^\s+(Step \d+[a-z]?: [^\n]+?)\s+([\d.]+)\s+[\d.]+%$", log, re.M):
        times[m.group(1).strip()] = float(m.group(2))
    return times


def parse_solver(out: Path) -> dict[str, int]:
    """Variant base name -> T_total in solver units."""
    text = (ROOT / out / "constraints" / "solver.log").read_text()
    ranks = {}
    for m in re.finditer(r"Variant \[\d+/\d+\]: (\S+)\s*\nOptimal T_total: ([\d,]+)", text):
        ranks[m.group(1)] = int(m.group(2).replace(",", ""))
    return ranks


def p03_functions(out: Path) -> list[str]:
    text = (ROOT / out / "IRs" / "p03_bufferized.mlir").read_text()
    return re.findall(r"func\.func @(\S+?)\(", text)


def tune_estimates(out: Path) -> dict[str, int]:
    """Function name -> the tuner's traffic estimate (elements), when tuned."""
    text = (ROOT / out / "IRs" / "p03_bufferized.mlir").read_text()
    est = {}
    for m in re.finditer(r"func\.func @(\S+?)\(.*?\{(?:[^{}]*?)loom\.tune\.estimate = (\d+)", text, re.S):
        est[m.group(1)] = int(m.group(2))
    return est


def lower(out: Path, index: int, kdir: Path) -> str:
    cmd = (f"rm -rf {kdir} tmp_output/mlir_{kdir.name} && "
           f"LOWER_MLIR_OUTPUT_DIR=$PWD/tmp_output/mlir_{kdir.name} SPLIT_KERNEL_OUTPUT_DIR=$PWD/{kdir} "
           f"./third_party/loom2ttkernel/lower.sh {out}/IRs/p03_bufferized.mlir {index}")
    res = container(cmd, timeout=1200)
    return res.stdout + res.stderr if res.returncode else ""


def arg_shapes(out: Path, func: str) -> list[list[int]]:
    ir = (ROOT / out / "IRs" / "p03_bufferized.mlir").read_text()
    sig = re.search(rf"func\.func @{re.escape(func)}\(([^)]*)\)", ir).group(1)
    return [[int(d) for d in m.group(1).split("x")]
            for m in re.finditer(r"memref<([0-9x]+)x(?:f16|bf16|f32)>", sig)]


def run_on_device(kdir: Path, shapes: list[list[int]], reference, iters: int, warmup: int):
    import torch
    import ttnn

    host_py = ROOT / kdir / "host_ttnn.py"
    spec = importlib.util.spec_from_file_location("loom_host_ttnn", host_py)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    order, outputs = h._PARAM_ORDER, set(h._OUTPUT_PARAM_ORDER)
    for f in ("reader.cpp", "compute.cpp", "writer.cpp"):
        (KERNEL_STAGE_DIR / f).write_bytes((ROOT / kdir / f).read_bytes())

    torch.manual_seed(0)
    torch_in = {n: (torch.zeros(s) if n in outputs else torch.randn(s))
                for n, s in zip(order, shapes)}
    d = ttnn.open_device(device_id=0)
    try:
        dev = {n: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
               for n, t in torch_in.items()}
        args = [dev[n] for n in order]

        cap = {}
        real = ttnn.generic_op
        def capture(io, prog):
            cap["io"], cap["prog"] = io, prog
            return real(io, prog)
        ttnn.generic_op = capture
        got = h.run(*args)
        ttnn.synchronize_device(d)
        ttnn.generic_op = real
        got = got if isinstance(got, tuple) else (got,)
        got = ttnn.to_torch(got[0]).float()

        ins = [torch_in[n].float() for n in order if n not in outputs]
        ref = reference(ins).float()
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()

        for _ in range(warmup):
            real(cap["io"], cap["prog"]); ttnn.synchronize_device(d)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); real(cap["io"], cap["prog"]); ttnn.synchronize_device(d)
            ts.append(time.perf_counter() - t0)
        return pcc, statistics.median(ts) * 1e3
    finally:
        ttnn.close_device(d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("program", choices=sorted(PROGRAMS))
    ap.add_argument("--tune", default=None, help="search policy (identity, fixed, enumerate, random)")
    ap.add_argument("--tune-options", default="")
    ap.add_argument("--tune-hoist", action="store_true", help="hoist loop-invariant loads in every candidate")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--per-order", type=int, default=None,
                    help="keep at most K mapping variants per tuner loop order before the top-k cut")
    ap.add_argument("--njobs", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tag", default=None, help="output sub-directory (default: policy name)")
    ap.add_argument("--shape", default=None, help="kernel-size override, e.g. K64_M256_N256")
    ap.add_argument("--no-device", action="store_true",
                    help="compile and lower only; report the stage each candidate reaches")
    ap.add_argument("--reuse", action="store_true",
                    help="skip compilation and lowering when their outputs already exist")
    a = ap.parse_args()

    prog = PROGRAMS[a.program]
    if a.shape:
        prog = Program(prog.kernel, prog.config, prog.reference, a.shape, prog.min_pcc)
    tag = a.tag or ((a.tune or "maxpar") + ("_hoist" if a.tune_hoist else "") + (f"_{a.shape}" if a.shape else ""))
    rows: list[dict] = []

    print(f"== {a.program}  policy={a.tune or 'maxpar (default)'} {a.tune_options}")
    t0 = time.perf_counter()
    out = Path("test/bench") / a.program / tag
    if a.reuse and (ROOT / out / "IRs" / "p03_bufferized.mlir").exists():
        log = (ROOT / out / "compile.log").read_text()
    else:
        out, log = compile_program(a.program, tag, a.tune, a.tune_options, a.topk, a.njobs, a.tune_hoist, prog,
                                   a.per_order)
    compile_s = time.perf_counter() - t0
    stage_ms = parse_stage_times(log)
    summary = {"compile_wall_s": round(compile_s, 1), "stage_ms": stage_ms}
    print("compile: " + f"{compile_s:.0f} s wall; " +
          ", ".join(f"{k.split(':', 1)[0]}={v / 1000:.1f}s" for k, v in stage_ms.items()))
    if not (ROOT / out / "IRs" / "p03_bufferized.mlir").exists():
        err = next((l for l in log.splitlines() if "rror" in l or "assert" in l), log[-300:])
        print(f"compile FAILED: {err.strip()[:200]}")
        rows.append({"stage": "compile", "error": err.strip()[:500]})
        (ROOT / out / "results.json").write_text(json.dumps({"summary": summary, "candidates": rows}, indent=2))
        return

    ranks = parse_solver(out)
    estimates = tune_estimates(out)
    funcs = p03_functions(out)
    print(f"compiled: {len(funcs)} candidate(s)")
    for i, func in enumerate(funcs, start=1):
        base = func.split("__is_double_buffer")[0]
        row = {"function": func, "T_total": ranks.get(base), "estimate": estimates.get(func)}
        kdir = Path("tmp_output/bench") / a.program / tag / f"f{i}"
        t1 = time.perf_counter()
        err = "" if a.reuse and (ROOT / kdir / "compute.cpp").exists() else lower(out, i, kdir)
        row["lower_s"] = round(time.perf_counter() - t1, 1)
        if err or not (ROOT / kdir / "compute.cpp").exists():
            msg = next((l for l in err.splitlines() if "error" in l), err[-300:])
            row.update(stage="lower", error=msg.strip()[:500])
            print(f"[{i}] {base[:60]}  lower FAILED: {msg.strip()[:120]}")
            rows.append(row)
            continue
        if a.no_device:
            row.update(stage="lowered")
            print(f"[{i}] {base[:70]}  T_total={row['T_total']}  lowered -> {kdir}")
            rows.append(row)
            continue
        try:
            pcc, ms = run_on_device(kdir, arg_shapes(out, func), prog.reference, a.iters, a.warmup)
        except Exception as e:  # noqa: BLE001 - report the stage, keep going
            row.update(stage="run", error=str(e)[:500])
            print(f"[{i}] {base[:60]}  run FAILED: {str(e)[:120]}")
            rows.append(row)
            break  # a wedged board makes further runs meaningless
        ok = pcc >= prog.min_pcc
        row.update(stage="ok" if ok else "pcc", pcc=pcc, device_ms=ms)
        est = f"  estimate={row['estimate']}" if row["estimate"] is not None else ""
        print(f"[{i}] {base[:70]}\n     T_total={row['T_total']}{est}  PCC={pcc:.6f} {'PASS' if ok else 'FAIL'}  "
              f"device={ms:.3f} ms  lower={row['lower_s']} s")
        rows.append(row)

    (ROOT / out / "results.json").write_text(json.dumps({"summary": summary, "candidates": rows}, indent=2))
    print(f"results: {out}/results.json")


if __name__ == "__main__":
    main()
