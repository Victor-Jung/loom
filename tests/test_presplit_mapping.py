"""The tuner splits spatial loops to core counts; the explorer only places them.

A pre-split mapping program has spatial loops with constant bounds (the
cores) tagged with `loom.block_syms`, explicit wave loops tagged temporal,
and the index recomposed as `axis + wave * cores`. The exploration assigns
each axis a physical dim and a level, keeps the tags, and never tiles or
creates wave loops itself. The `maxpar` policy produces such programs from
the frontend's p00 using the mesh of the hardware spec.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
HW12x10 = ROOT / "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x12y10.mlir"
HW1x1 = ROOT / "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x1y1.mlir"
LOOM_OPT = Path(os.environ.get(
    "LOOM_OPT", ROOT / "third_party/loom-dataflow/build/tool/loom-opt/loom-opt"))

loom_pipeline = pytest.importorskip("loom_pipeline")


def tune(fixture: Path, *opts: str) -> subprocess.CompletedProcess:
    if not LOOM_OPT.exists():
        pytest.skip(f"loom-opt not built at {LOOM_OPT}")
    return subprocess.run(
        [str(LOOM_OPT), f"--loom-tune-mapping-program={' '.join(opts)}", str(fixture)],
        capture_output=True, text=True, timeout=300, cwd=ROOT)


def test_explorer_places_presplit_axes_without_tiling() -> None:
    out, etg = loom_pipeline.run_exploration(
        input_mlir=(FIXTURES / "presplit_scaled_ewise_p00.mlir").read_text(),
        hw_spec_file=str(HW12x10), produce_etg=True)
    names = re.findall(r"func\.func @(\S+?)\(", out)
    # One axis of 8 cores: on x or on y, nothing else.
    assert sorted(n.split("__")[1] for n in names) == ["x1_y8", "x8_y1"], names
    assert out.count("affine.parallel (%arg3) = (0) to (8)") == 2
    assert "loom.physical_dim = @dim_x" in out and "loom.physical_dim = @dim_y" in out
    # The tuner's wave loop and recomposition survive untouched.
    assert out.count("loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>") == 2
    assert "affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>" in out
    assert "Divisible" in etg


def test_explorer_rejects_axes_that_do_not_fit() -> None:
    p00 = (FIXTURES / "presplit_scaled_ewise_p00.mlir").read_text()
    too_big = p00.replace("(0) to (8)", "(0) to (16)").replace("d1 * 8", "d1 * 16") \
                 .replace("%c8 = arith.constant 8", "%c8 = arith.constant 16")
    with pytest.raises(RuntimeError) as err:
        loom_pipeline.run_exploration(input_mlir=too_big, hw_spec_file=str(HW12x10),
                                      produce_etg=False)
    assert "fits the mesh" in str(err.value) or "no placement" in str(err.value), str(err.value)


def test_split_emits_a_constant_axis_and_a_wave_loop() -> None:
    # With a mesh every candidate is completed by maxpar, so the remaining
    # axes are split too: m over 8 and n over 4 next to b's 2 (2*4 on x, 8 on y).
    result = tune(FIXTURES / "tune_scaled_ewise_p00.mlir", "policy=fixed",
                  "options=split(b,2)", f"hw-spec={HW12x10}")
    assert result.returncode == 0, result.stderr[-2000:]
    out = result.stdout
    assert re.search(r"affine\.parallel \(%\w+, %\w+, %\w+\) = \(0, 0, 0\) to \(8, 2, 4\)", out), out
    assert "loom.block_syms = [@tile_m, @tile_b, @tile_n]" in out
    assert "affine.apply" in out
    assert "loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>" in out
    assert "loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>" in out


def test_split_needs_the_mesh_and_a_dividing_core_count() -> None:
    no_mesh = tune(FIXTURES / "tune_scaled_ewise_p00.mlir", "policy=fixed", "options=split(b,2)")
    assert no_mesh.returncode != 0 and "needs the mesh" in no_mesh.stderr
    bad = tune(FIXTURES / "tune_scaled_ewise_p00.mlir", "policy=fixed",
               "options=split(b,3)", f"hw-spec={HW12x10}")
    assert bad.returncode != 0 and "divide" in bad.stderr


@pytest.mark.parametrize("fixture, expected", [
    ("tune_scaled_ewise_p00.mlir", "b8Sm8"),      # 64 cores: b on one dim, m on the other
    ("tune_matmul_p00.mlir", "m8Sn8"),            # k carries the accumulator and stays
    ("etg_mamba_p00.mlir", "c10Sm6Sn2"),          # the whole 12x10 mesh
    ("tune_ewise3d_p00.mlir", "b2Sm8Sn4"),        # 2*4 = 8 on x, 8 on y
    ("bind_attention_fullrow_p00.mlir", "m2Sm8"),  # one loop of 16 tiles spanning both dims
])
def test_maxpar_fills_the_mesh(fixture: str, expected: str) -> None:
    result = tune(FIXTURES / fixture, "policy=maxpar", f"hw-spec={HW12x10}", "dump-tree")
    assert result.returncode == 0, result.stderr[-2000:]
    after = re.search(r"loop tree \(after, (\S+?),", result.stderr)
    assert after and after.group(1).startswith(expected), result.stderr[-1500:]
    assert "loom.block_syms = [" in result.stdout


def test_maxpar_on_one_core_makes_every_axis_constant() -> None:
    result = tune(FIXTURES / "tune_scaled_ewise_p00.mlir", "policy=maxpar", f"hw-spec={HW1x1}")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "affine.parallel (%arg3) = (0) to (1)" in result.stdout


def test_beam_with_the_mesh_returns_distinct_presplit_trees() -> None:
    result = tune(FIXTURES / "etg_mamba_p00.mlir", "policy=beam",
                  "options=width=4;depth=5;keep=3", f"hw-spec={HW12x10}", "dump-tree")
    assert result.returncode == 0, result.stderr[-2000:]
    names = re.findall(r"func\.func @(\S+?)\(", result.stdout)
    assert len(names) == len(set(names)) and names, names
    assert result.stdout.count("loom.block_syms = [") == len(names)
    # Trading cores for multicast: the head loop over 8 cores (cb and C are
    # invariant in h) on 80 cores is rated below maxpar's 120-core tree.
    estimates = [float(e) for e in re.findall(r"loop tree \(after, \S+ estimate ([0-9.e+]+)\)", result.stderr)]
    assert estimates and min(estimates) < 1.13e7, estimates
    assert any("h8" in n for n in names), names


def test_temporalize_after_split_keeps_the_recomposition_on_its_axis() -> None:
    # Split c (axis index 2 of the matmul-like (m, n, c) parallel), then
    # temporalize n: c moves to index 1 and its wave loop must still rebuild
    # the chunk index from the c axis, not from whatever now sits at index 2.
    result = tune(FIXTURES / "etg_mamba_p00.mlir", "policy=fixed",
                  "options=split(c,10);temporalize(n)", f"hw-spec={HW12x10}")
    assert result.returncode == 0, result.stderr[-2000:]
    out = result.stdout
    par = re.search(r"affine\.parallel \(([^)]*)\) = \([0, ]*\) to \(([^)]*)\) \{", out)
    assert par, out
    ivs = [v.strip() for v in par.group(1).split(",")]
    sizes = [int(v) for v in par.group(2).split(",")]
    c_iv = ivs[sizes.index(10)]
    alias = re.search(r"(#map\d*) = affine_map<\(d0, d1\) -> \(d0 \+ d1 \* 10\)>", out)
    assert alias, out
    assert re.search(rf"affine\.apply {re.escape(alias.group(1))}\({re.escape(c_iv)}, ", out), out
