"""The roofline estimate's terms: split-reduction merges on the merging core,
reads queued in one DRAM bank, and how the DRAM, compute and per-core bounds
combine (`norm=`)."""

from __future__ import annotations

import math
import re

from test_mapping_tune import FIXTURES, MATMUL, tune

HW12x10 = "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x12y10.mlir"
BATCH_SUM = FIXTURES / "tune_batch_sum_p00.mlir"  # x[32, 1024, 2048] summed over b


def terms(schedule: str, policy: str = "fixed", fixture=MATMUL) -> tuple[float, dict[str, float]]:
    r = tune(fixture, f"policy={policy}", f"options={{{schedule}}}", f"hw-spec={HW12x10}", "dump-tree")
    assert r.returncode == 0, (schedule, r.stderr[-1500:])
    est = float(re.search(r"loop tree \(after, \S+, estimate (\S+)\)", r.stderr).group(1))
    line = re.search(r"^cost terms: (.*)$", r.stderr, re.M).group(1)
    return est, {k: float(v) for k, v in (kv.split("=") for kv in line.split(";"))}


def test_split_reductions_charge_their_gather_and_merge() -> None:
    _, plain = terms("split(m,8)")
    _, four = terms("splitreduce(k,4)")
    _, eight = terms("splitreduce(k,8)")
    assert plain["merge_ns"] == 0
    assert 0 < four["merge_ns"] < eight["merge_ns"]


def test_narrow_blocks_queue_in_one_dram_bank() -> None:
    # 8 n-cores: 1x8-tile blocks over 8 banks; 24 n-cores: 8x1-tile blocks,
    # each tile of a block in the same bank (rows of 64 tiles).
    _, wide = terms("split(n,8);split(m,11)", fixture=BATCH_SUM)
    _, narrow = terms("split(n,8);split(n,3)", fixture=BATCH_SUM)
    assert wide["bank_excess"] == 0
    assert narrow["bank_excess"] > 0
    assert wide["dram_bytes"] == narrow["dram_bytes"]


def test_bounds_combine_by_their_norm() -> None:
    def bounds(t):
        return (t["dram_bytes"] / 421 + t["dram_blocks"] * 0.92,
                t["compute_ns"] + t["merge_ns"],
                t["core_bytes"] / 40.4 + t["blocks"] * 160 + t["mreceivers"] * 131)

    est, t = terms("splitreduce(k,4)")
    assert math.isclose(est, math.sqrt(sum(b * b for b in bounds(t))), rel_tol=1e-4)
    # Tuner keys go with a policy other than `fixed` (its options are moves).
    est, t = terms("norm=0", policy="maxpar")
    assert math.isclose(est, max(bounds(t)), rel_tol=1e-4)
    est, t = terms("dram-bw=100", policy="maxpar")
    d, k, c = bounds(t)
    d += t["dram_bytes"] / 100 - t["dram_bytes"] / 421
    assert math.isclose(est, math.sqrt(d * d + k * k + c * c), rel_tol=1e-4)
