"""The mapping-program tuning pass round-trips p00 and applies schedules.

`loom-tune-mapping-program` rebuilds each function from its loop tree. With the
identity policy the result must be byte-identical to the input, including on a
function whose accumulation loop carries `iter_args`. A `fixed` schedule must
change the loop order and hoist the inner trip count; an illegal move must be
rejected with a diagnostic instead of emitting a wrong program.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
EWISE = FIXTURES / "tune_ewise3d_p00.mlir"
MATMUL = FIXTURES / "tune_matmul_p00.mlir"
SCALED = FIXTURES / "tune_scaled_ewise_p00.mlir"
LOOM_OPT = Path(
    os.environ.get(
        "LOOM_OPT", ROOT / "third_party/loom-dataflow/build/tool/loom-opt/loom-opt"
    )
)


def run(*args: str) -> subprocess.CompletedProcess:
    if not LOOM_OPT.exists():
        pytest.skip(f"loom-opt not built at {LOOM_OPT}; set LOOM_OPT to override")
    return subprocess.run(
        [str(LOOM_OPT), *args], capture_output=True, text=True, timeout=300, cwd=ROOT
    )


def tune(fixture: Path, *opts: str) -> subprocess.CompletedProcess:
    return run(f"--loom-tune-mapping-program={' '.join(opts)}", str(fixture))


@pytest.mark.parametrize("fixture", [EWISE, MATMUL], ids=["ewise3d", "matmul"])
def test_identity_round_trip_is_byte_identical(fixture: Path) -> None:
    plain = run(str(fixture))
    tuned = tune(fixture, "policy=identity")
    assert plain.returncode == 0, plain.stderr[-2000:]
    assert tuned.returncode == 0, tuned.stderr[-2000:]
    assert plain.stdout == tuned.stdout


def test_interchange_swaps_loops_and_hoists_bounds() -> None:
    result = tune(EWISE, "policy=fixed", "options=interchange(m,n)", "dump-tree=true")
    assert result.returncode == 0, result.stderr[-2000:]

    after = result.stderr.split("loop tree (after", 1)[1]
    loops = re.findall(r"L\d+ temporal (\w+)", after)
    assert loops == ["n", "m"], loops

    ir = result.stdout
    body = ir.split("affine.parallel", 1)[1]
    # Both trip counts are computed before the first scf.for.
    prelude = body.split("scf.for", 1)[0]
    assert prelude.count("arith.ceildivui") == 2, prelude
    # The outer loop now iterates over n (bound derived from 512 / tile_n).
    outer_bound = re.search(r"scf.for %\w+ = %c0 to (%\d+)", body).group(1)
    assert re.search(rf"{outer_bound} = arith.ceildivui %c512, ", ir), outer_bound
    # The linalg statement survives untouched.
    assert "linalg.generic" in ir and "arith.addf" in ir


def test_enumerate_emits_one_function_per_loop_order() -> None:
    result = tune(EWISE, "policy=enumerate")
    assert result.returncode == 0, result.stderr[-2000:]
    names = sorted(re.findall(r"func\.func @(\S+?)\(", result.stdout))
    assert names == ["_ewise3d__order_b_m_n", "_ewise3d__order_b_n_m"], names


def test_greedy_moves_the_shared_operand_loop_innermost() -> None:
    # s[m, n] is invariant in b: with b innermost its tile is reused across
    # the batch, so the traffic estimate prefers order m, n, b.
    result = tune(SCALED, "policy=greedy", "dump-tree=true")
    assert result.returncode == 0, result.stderr[-2000:]
    after = result.stderr.split("loop tree (after", 1)[1]
    assert re.findall(r"L\d+ temporal (\w+)", after) == ["n", "b"]
    before = re.search(r"before, estimate ([0-9.e+]+)", result.stderr).group(1)
    chosen = re.search(r"after, m_n_b, estimate ([0-9.e+]+)", result.stderr).group(1)
    assert float(chosen) < float(before)


def test_hoist_moves_the_invariant_load_out_of_the_inner_loop() -> None:
    result = tune(SCALED, "policy=greedy", "hoist=true")
    assert result.returncode == 0, result.stderr[-2000:]
    ir = result.stdout
    # s (%arg1) is loaded between the n loop and the b loop; x (%arg0), which
    # depends on b, stays inside.
    outer, inner = [m.start() for m in re.finditer(r"scf\.for ", ir)][:2]
    s_load = ir.index("memref.subview %arg1")
    x_load = ir.index("memref.subview %arg0")
    assert outer < s_load < inner < x_load


def test_hoist_leaves_reduction_loads_alone() -> None:
    plain = run(str(MATMUL))
    hoisted = tune(MATMUL, "hoist=true")
    assert hoisted.returncode == 0, hoisted.stderr[-2000:]
    assert plain.stdout == hoisted.stdout


def test_random_keeps_the_start_when_nothing_is_legal() -> None:
    result = tune(MATMUL, "policy=random", "options=seed=3;walks=4;len=2")
    assert result.returncode == 0, result.stderr[-2000:]
    assert re.findall(r"func\.func @(\S+?)\(", result.stdout) == ["_matmul"]


def test_illegal_interchange_is_rejected() -> None:
    result = tune(MATMUL, "policy=fixed", "options=interchange(L0,k)")
    assert result.returncode != 0
    assert "illegal move interchange(m,n,k): both loops must be scf.for" in result.stderr


def test_unknown_policy_is_rejected() -> None:
    result = tune(EWISE, "policy=nope")
    assert result.returncode != 0
    assert "unknown tuning policy 'nope'" in result.stderr
