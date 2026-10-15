#!/usr/bin/env python
"""Search-space study for mamba_chunk_scan: compile with a tuning policy, lower
every candidate, then check and time each one on the board.

The same flow as benchmarks/run.py (whose helpers it uses) for the program
run.py cannot drive: mamba's inputs and golden reference live in
experiments/host/mamba_verify.py and its timing in mamba_bench.py, so each
candidate runs those two scripts in fresh processes. Writes
test/bench/mamba_chunk_scan/<tag>/results.json with run.py's row format.

    python -u benchmarks/mamba_space.py --tune random \\
        --tune-options "walks=1000000;len=8;seed=1;states=1000" --tag rw1000 --lower-jobs 16
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.programs import Program  # noqa: E402
from benchmarks.run import (KERNEL_STAGE_DIR, beam_trace, compile_program, lower_all,  # noqa: E402
                            p03_functions, parse_solver, parse_stage_times, pe_env, reset_board,
                            tune_estimates, tune_terms, write_results)

NAME = "mamba_chunk_scan"
CONFIG = "tmp_output/phaseG_mamba_cfg.json"
MIN_PCC = 0.99


def run_candidate(kdir: Path, ir: Path, iters: int, timeout_s: int, fidelity: str) -> dict:
    """PCC from mamba_verify.py, then device time from mamba_bench.py, each in
    its own process; a failure or hang resets the board."""
    for f in ("reader.cpp", "compute.cpp", "writer.cpp"):
        (KERNEL_STAGE_DIR / f).write_bytes((ROOT / kdir / f).read_bytes())
    env = pe_env(fidelity, True)
    common = ["--kernels", str(ROOT / kdir), "--ir", str(ROOT / ir)]
    try:
        v = subprocess.run([sys.executable, "-u", "experiments/host/mamba_verify.py", *common, "--causal", "element"],
                           capture_output=True, text=True, timeout=timeout_s, cwd=ROOT, env=env)
        m = re.search(r"^\s+PCC (\S+)", v.stdout, re.M)
        if not m:
            reset_board()
            return {"stage": "run", "error": "verify: " + (v.stderr or v.stdout)[-400:].strip()}
        pcc = float(m.group(1))
        b = subprocess.run([sys.executable, "-u", "experiments/host/mamba_bench.py", *common,
                            "--iters", str(iters)],
                           capture_output=True, text=True, timeout=timeout_s, cwd=ROOT, env=env)
        t = re.search(r"device\s+([\d.]+) ms\s+\(sd ([\d.]+)\)", b.stdout)
        if not t:
            reset_board()
            return {"stage": "run", "error": "bench: " + (b.stderr or b.stdout)[-400:].strip(), "pcc": pcc}
    except subprocess.TimeoutExpired:
        reset_board()
        return {"stage": "run", "error": f"hang: no result within {timeout_s} s; board reset"}
    ms, sd = float(t.group(1)), float(t.group(2))
    return {"pcc": pcc, "device_ms": ms, "sd_ms": sd, "timing": "trace"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="L1920_N32_H128_G4_D128_C192")
    ap.add_argument("--tune", required=True)
    ap.add_argument("--tune-options", default="")
    ap.add_argument("--per-order", type=int, default=1)
    ap.add_argument("--topk", type=int, default=100000)
    ap.add_argument("--njobs", type=int, default=16)
    ap.add_argument("--lower-jobs", type=int, default=16)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--fidelity", default="HiFi4")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reuse", action="store_true", help="keep an existing compile and lowered kernels")
    ap.add_argument("--no-device", action="store_true", help="compile and lower only")
    a = ap.parse_args()

    prog = Program(f"kernels/{NAME}.py", CONFIG, reference=None, shape=a.shape)
    out = Path("test/bench") / NAME / a.tag
    kroot = Path("tmp_output/bench") / NAME / a.tag
    t0 = time.perf_counter()
    if a.reuse and (ROOT / out / "IRs" / "p03_bufferized.mlir").exists():
        log = (ROOT / out / "compile.log").read_text()
    else:
        out, log = compile_program(NAME, a.tag, a.tune, a.tune_options, a.topk, a.njobs, False, prog, a.per_order)
    summary = {"compile_wall_s": round(time.perf_counter() - t0, 1), "stage_ms": parse_stage_times(log),
               "shape": a.shape, "tune": a.tune, "tune_options": a.tune_options}
    ir = out / "IRs" / "p03_bufferized.mlir"
    if not (ROOT / ir).exists():
        print("compile FAILED:\n" + log[-1500:])
        return
    ranks, estimates, terms = parse_solver(out), tune_estimates(out), tune_terms(out)
    funcs, paths = p03_functions(out), beam_trace(out, log)
    print(f"compiled {len(funcs)} candidate(s) in {summary['compile_wall_s']:.0f} s")

    todo = [i for i in range(1, len(funcs) + 1)
            if not (a.reuse and (ROOT / kroot / f"f{i}" / "compute.cpp").exists())]
    t1 = time.perf_counter()
    if todo:
        lower_all(out, kroot, todo, a.lower_jobs)
    summary["lower_all_s"] = round(time.perf_counter() - t1, 1)
    print(f"lowered {len(todo)} candidate(s) in {summary['lower_all_s']:.0f} s")

    if a.no_device:
        write_results(out, summary, [])
        return
    rows: list[dict] = []
    reset_board()
    for i, func in enumerate(funcs, start=1):
        if i % 25 == 0:
            write_results(out, summary, rows)
        base = func.split("__is_double_buffer")[0]
        row = {"function": func, "T_total": ranks.get(base), "estimate": estimates.get(func),
               "terms": terms.get(func), "index": i}
        sig = re.search(r"__order_(.+?)__x", base)
        if sig and sig.group(1) in paths:
            row["path"] = paths[sig.group(1)]
        kdir = kroot / f"f{i}"
        row["kdir"] = str(kdir)
        if not (ROOT / kdir / "compute.cpp").exists():
            log_f = ROOT / kroot / "lower_logs" / f"f{i}.log"
            msg = log_f.read_text()[-500:] if log_f.exists() else "not lowered"
            row.update(stage="lower", error=msg.strip())
            print(f"[{i}] lower FAILED")
            rows.append(row)
            continue
        t2 = time.perf_counter()
        res = run_candidate(kdir, ir, a.iters, a.timeout, a.fidelity)
        row["run_s"] = round(time.perf_counter() - t2, 1)
        row.update(res)
        if "error" not in res:
            row["stage"] = "ok" if res["pcc"] >= MIN_PCC else "pcc"
            print(f"[{i}] T_total={row['T_total']} estimate={row['estimate']} PCC={res['pcc']:.5f} "
                  f"device={res['device_ms']:.4f} ms")
        else:
            print(f"[{i}] run FAILED: {res['error'][:120]}")
        rows.append(row)
    summary["total_s"] = round(time.perf_counter() - t0, 1)
    write_results(out, summary, rows)
    print(f"results: {out}/results.json")


if __name__ == "__main__":
    main()
