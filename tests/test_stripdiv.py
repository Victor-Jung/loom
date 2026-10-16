"""stripdiv: strip-mine a loop at the constant its index is divided by.

`stripdiv(h,K)` turns loop h (static trip count T, K | T) into a loop over
the T/K groups (`hqK`) holding a loop over the K members (`hrK`), with
h = q*K + r: every `h / K` (arith division, `affine.apply` floordiv) becomes
q and every `h % K` becomes r, wherever it appears -- in the index of a
tensor access or in a statement (a scalar used by compute). The target may
also be a statement (`S<id>`) or an argument (`arg<N>`) whose indices use
the division.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from test_beam_strategy import HW12x10
from test_mapping_tune import FIXTURES, run, tune

MAMBA = FIXTURES / "tune_mamba_p00.mlir"
GROUPS = FIXTURES / "tune_group_divisions_p00.mlir"


def fixed(fixture: Path, schedule: str, *opts: str) -> subprocess.CompletedProcess:
    return tune(fixture, "policy=fixed", f"options={{{schedule}}}", "dump-tree", *opts)


def after(r: subprocess.CompletedProcess) -> tuple[str, float, dict[str, str]]:
    """Signature, estimate and reuse-matrix row (axis -> x/.) per operand."""
    assert r.returncode == 0, r.stderr[-2000:]
    sig, est = re.search(r"loop tree \(after, (\S+), estimate (\S+)\)", r.stderr).groups()
    tail = r.stderr.split("loop tree (after", 1)[1]
    header = re.search(r"^operand\s+(.*)$", tail, re.M).group(1).split()
    axes = [h.rstrip("*") for h in header if not h.startswith(("dram@", "core@"))]
    rows = {}
    for m in re.finditer(r"^(arg\d+)\s+(.*)$", tail, re.M):
        marks = m.group(2).split()[: len(axes)]
        rows.setdefault(m.group(1), dict(zip(axes, marks)))
    return sig, float(est), rows


def stmt_id(fixture: Path, op: str, after_op: str | None = None) -> str:
    """Id of the first statement `op` (after the first `after_op`) in the tree."""
    r = tune(fixture, "policy=identity", "dump-tree")
    lines = r.stderr.split("loop tree (after", 1)[0].splitlines()
    start = 0
    if after_op:
        start = next(i for i, l in enumerate(lines) if l.strip().endswith(after_op))
    for line in lines[start:]:
        m = re.match(r"\s*(S\d+) (\S+)$", line)
        if m and m.group(2) == op:
            return m.group(1)
    raise AssertionError(f"no {op} statement")


def emitted(fixture: Path, schedule: str) -> str:
    r = run(f"--loom-tune-mapping-program=policy=fixed options={{{schedule}}}", str(fixture))
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_group_loads_do_not_depend_on_the_member_loop() -> None:
    # mamba: cb (arg0) and C (arg5) are indexed by h // 8, x (arg1) by h.
    sig, _, rows = after(fixed(MAMBA, "stripdiv(h,8)"))
    assert "_hq8_hr8_" in sig, sig
    assert rows["arg0"]["hq8"] == "x" and rows["arg0"]["hr8"] == ".", rows["arg0"]
    assert rows["arg5"]["hq8"] == "x" and rows["arg5"]["hr8"] == ".", rows["arg5"]
    assert rows["arg1"]["hq8"] == "x" and rows["arg1"]["hr8"] == "x", rows["arg1"]


def test_the_division_is_gone_and_the_symbols_are_declared() -> None:
    ir = emitted(MAMBA, "stripdiv(h,8)")
    assert ir.count("func.func") == 1, "scratch functions must be erased"
    assert "arith.divui" not in ir
    assert "loom.tile_hq8 = " in ir and "loom.tile_hr8 = " in ir
    assert "loom.sym @tile_hq8" in ir and "loom.sym @tile_hr8" in ir
    assert "stripdiv" not in ir


def test_tensor_index_and_statement_targets_name_the_same_move() -> None:
    sig = after(fixed(MAMBA, "stripdiv(h,8)"))[0]
    div = stmt_id(MAMBA, "arith.divui")
    view = stmt_id(MAMBA, "memref.subview", after_op="arith.divui")
    for target in (div, view, "arg5", "arg0", "arg5,8"):
        assert after(fixed(MAMBA, f"stripdiv({target})"))[0] == sig, target


def test_affine_floordiv_index_and_scalar_statements_fold() -> None:
    # w (arg1) is indexed by affine.apply (h floordiv 4); h / 4 and h % 4 are
    # also computed as statements whose values the compute body reads.
    sig, _, rows = after(fixed(GROUPS, "stripdiv(h,4)"))
    assert sig.endswith("hq4_hr4"), sig
    assert rows["arg1"]["hr4"] == "." and rows["arg0"]["hr4"] == "x", rows
    ir = emitted(GROUPS, "stripdiv(h,4)")
    assert "floordiv" not in ir and "arith.divui" not in ir and "arith.remui" not in ir
    # the casts read the group and member induction variables directly
    q, r = re.search(r"scf\.for (%\w+) = .*\n\s*scf\.for (%\w+) = ", ir).groups()
    assert f"arith.index_cast {q} " in ir and f"arith.index_cast {r} " in ir, ir


def test_statement_targets_on_scalar_uses() -> None:
    sig = after(fixed(GROUPS, "stripdiv(h,4)"))[0]
    for op in ("arith.divui", "arith.remui", "affine.apply", "linalg.generic"):
        assert after(fixed(GROUPS, f"stripdiv({stmt_id(GROUPS, op)})"))[0] == sig, op
    assert after(fixed(GROUPS, "stripdiv(arg1)"))[0] == sig


def test_hoisting_the_group_load_out_of_the_member_loop() -> None:
    _, plain, _ = after(fixed(GROUPS, "stripdiv(h,4)"))
    _, hoisted, _ = after(fixed(GROUPS, "stripdiv(h,4)", "hoist"))
    assert hoisted < plain, (hoisted, plain)


def test_refused_targets() -> None:
    r = fixed(GROUPS, "stripdiv(h,3)")
    assert r.returncode != 0 and "not divided by 3" in r.stderr
    r = fixed(GROUPS, "stripdiv(arg0)")
    assert r.returncode != 0 and "uses no division" in r.stderr
    r = fixed(GROUPS, "split(m,8);stripdiv(h,4)", f"hw-spec={HW12x10}")
    assert r.returncode != 0 and "has a split" in r.stderr


def test_split_of_the_member_loop_multicasts_the_group_load() -> None:
    # 4 cores on the members of one group: C and cb are the same on all four.
    _, _, rows = after(fixed(MAMBA, "stripdiv(h,8);split(hr8,4)", f"hw-spec={HW12x10}"))
    assert rows["arg5"]["hr84"] == "." and rows["arg0"]["hr84"] == ".", rows


def test_beam_offers_stripdiv_and_option_turns_it_off() -> None:
    def beam(extra: str) -> subprocess.CompletedProcess:
        return tune(MAMBA, "policy=beam",
                    f"options={{width=4;depth=2;keep=2;trace=1{extra}}}",
                    f"hw-spec={HW12x10}")
    on, off = beam(""), beam(";stripdiv=off")
    assert on.returncode == 0 and off.returncode == 0, on.stderr[-2000:]
    assert re.search(r"stripdiv \d+", on.stderr), on.stderr[-2000:]
    assert not re.search(r"stripdiv \d+", off.stderr)
