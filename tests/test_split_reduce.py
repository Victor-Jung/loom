"""`splitreduce(L, S)`: a reduction loop split over S cores.

Core s of a new constant axis runs iterations [s*T/S, (s+1)*T/S) of the loop
(equal chunks: the solver keeps T divisible by S through the sequential
loop's divisibility); each carried value is gathered across the axis and
merged on core 0, where the statements after the loop run. Legal for loops
whose carried values have a known merge (sums here), from 0 with step 1,
followed only by statements, when every tile size can give equal chunks.
"""

from __future__ import annotations

import re

from test_mapping_tune import FIXTURES, MATMUL, tune

HW12x10 = "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x12y10.mlir"
SCALED = FIXTURES / "tune_scaled_ewise_p00.mlir"


def fixed(fixture, schedule: str):
    return tune(fixture, "policy=fixed", f"options={schedule}", f"hw-spec={HW12x10}", "dump-tree")


def test_split_reduction_emits_chunks_gather_merge_and_core0_epilogue() -> None:
    r = fixed(MATMUL, "splitreduce(k,4)")
    assert r.returncode == 0, r.stderr[-2000:]
    ir = r.stdout
    # 16384 / 4 = 4096: every core gets at least one k tile.
    assert "loom.sym @tile_k {upper_bound = 4096 : index}" in ir, ir[:1500]
    par = re.search(r"affine\.parallel \(([^)]*)\) = \(0, 0, 0\) to \(([^)]*)\)", ir)
    assert par and par.group(2).split(", ")[-1] == "4", par
    core = par.group(1).split(", ")[-1]
    # chunk loop from 0, original index core * chunk + j
    assert re.search(rf"arith\.muli {core}, %\w+", ir)
    gather = re.search(rf"loom\.gather %\w+, %\w+ across\({core} : index\).*memref<4x\?x\?xf16>", ir)
    assert gather, ir
    tail = ir[gather.end():]
    if_pos = tail.index("scf.if")
    assert tail.index("linalg.generic") > if_pos  # the merge runs on core 0
    assert tail.index("memref.copy") > if_pos     # so does the store


def test_split_reduction_legality() -> None:
    cases = [
        (MATMUL, "splitreduce(k,3)", "equal chunks"),        # 512 tiles of 32
        (SCALED, "splitreduce(b,2)", "carries no reduction"),    # temporal, no iter_args
        (MATMUL, "splitreduce(k,1)", "bad core count"),
        (MATMUL, "splitreduce(k,4);splitreduce(k,2)", "already split"),
    ]
    for fixture, schedule, reason in cases:
        r = fixed(fixture, schedule)
        assert r.returncode != 0 and reason in r.stderr, (schedule, r.stderr[-1200:])


def test_axes_after_a_temporalized_one_keep_the_reduction_axis() -> None:
    r = fixed(MATMUL, "splitreduce(k,4);split(m,8);temporalize(n)")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "loom.gather" in r.stdout


def test_beam_offers_split_reductions_and_reduce_off_removes_them() -> None:
    opts = "width=6;depth=2;keep=6;trace=1"
    on = tune(MATMUL, "policy=beam", f"options={opts}", f"hw-spec={HW12x10}")
    assert on.returncode == 0 and "splitreduce 1" not in on.stderr
    assert re.search(r"splitreduce \d+", on.stderr), on.stderr[-1500:]
    off = tune(MATMUL, "policy=beam", f"options={opts};reduce=off", f"hw-spec={HW12x10}")
    assert off.returncode == 0 and "splitreduce" not in off.stderr


GQA_DECODE = FIXTURES / "tune_gqa_decode_p00.mlir"


def test_online_softmax_reduction_merges_max_sum_and_output() -> None:
    # The KV loop carries (m, acc, l): m' = max(m, rowmax), acc' = acc *
    # exp(m - m') + P V, l' = l * exp(m - m') + rowsum. Split over 4 cores,
    # core 0 merges M = max_s m_s, w_s = exp(m_s - M), L = sum_s l_s w_s,
    # ACC = sum_s acc_s w_s.
    r = fixed(GQA_DECODE, "splitreduce(n,4)")
    assert r.returncode == 0, r.stderr[-2000:]
    ir = r.stdout
    assert len(re.findall(r"loom\.gather .* to memref<4x", ir)) == 3, ir
    merge = ir[ir.index("scf.if"):]
    for op in ("arith.maximumf", "math.exp", "loom.broadcast", "arith.mulf", "arith.addf"):
        assert op in merge, op


def test_a_carried_value_without_a_known_merge_is_refused() -> None:
    # A running product has no merge in the recogniser: refused, with a reason.
    r = fixed(FIXTURES / "tune_product_reduce_p00.mlir", "splitreduce(k,2)")
    assert r.returncode != 0 and "no known merge" in r.stderr, r.stderr[-1500:]
