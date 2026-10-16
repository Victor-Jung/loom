"""Unsplit and resplit: changing the cores of a loop that is already split.

`unsplit(X)` takes the constant axes of a split loop X off the parallel loop
and runs X sequentially over its whole range, where it is in the nest.
`resplit(X,Y,cores)` does that and splits Y (X with another core count,
another loop, or a reduction with `reduce`) in one move, `FxG` over both mesh
dims, so a search changes a split without passing through a tree with idle
cores. The beam offers both only with `resplit=1`.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from test_beam_strategy import HW12x10, beam, tried
from test_mapping_tune import FIXTURES, tune

GQA = FIXTURES / "tune_gqa_decode_p00.mlir"
ROTARY = FIXTURES / "tune_rotary_p00.mlir"


def fixed(fixture: Path, schedule: str) -> subprocess.CompletedProcess:
    return tune(fixture, "policy=fixed", f"options={schedule}", "dump-tree",
                f"hw-spec={HW12x10}")


def after(r: subprocess.CompletedProcess) -> tuple[str, float, list[str], list[str]]:
    """Signature, estimate, spatial axes and temporal loops of the result."""
    assert r.returncode == 0, r.stderr[-2000:]
    sig, est = re.search(r"loop tree \(after, (\S+), estimate (\S+)\)", r.stderr).groups()
    tree = r.stderr.split("loop tree (after", 1)[1]
    axes = re.search(r"L\d+ spatial  (\S+)", tree).group(1).split(",")
    loops = re.findall(r"L\d+ temporal (\w+)", tree)
    return sig, float(est), axes, loops


def test_resplit_reaches_the_tree_of_the_direct_split() -> None:
    sig, est, axes, _ = after(fixed(ROTARY, "split(t,12);split(h,10);resplit(h,h,5)"))
    assert "h5" in axes and "h10" not in axes, axes
    assert (sig, est) == after(fixed(ROTARY, "split(t,12);split(h,5)"))[:2]


def test_resplit_keeps_the_loop_order() -> None:
    _, _, _, loops = after(fixed(ROTARY, "split(t,12);split(h,10);interchange(t,h)"))
    assert loops == ["h", "t"], loops
    _, _, axes, resplit = after(fixed(
        ROTARY, "split(t,12);split(h,10);interchange(t,h);resplit(h,h,5)"))
    assert resplit == loops and "h5" in axes, (resplit, axes)


def test_resplit_gives_the_cores_to_another_loop() -> None:
    _, _, axes, _ = after(fixed(ROTARY, "split(t,12);split(h,10);resplit(h,t,10)"))
    assert sorted(axes) == ["t10", "t12"], axes


def test_resplit_over_both_mesh_dims() -> None:
    _, _, axes, _ = after(fixed(GQA, "split(b,8);split(b,8);resplit(b,b,6x10)"))
    assert {"b6", "b10"} <= set(axes) and "b8" not in axes, axes


def test_resplit_of_a_split_reduction() -> None:
    _, _, axes, _ = after(fixed(GQA, "splitreduce(n,4);resplit(n,n,2,reduce)"))
    assert "n2" in axes and "n4" not in axes, axes


def test_resplit_that_keeps_the_split_is_rejected() -> None:
    r = fixed(ROTARY, "split(t,12);split(h,10);resplit(h,h,10)")
    assert r.returncode != 0
    assert "the loop keeps the same split" in r.stderr


def test_unsplit_keeps_a_parallel_axis() -> None:
    r = fixed(ROTARY, "split(t,12);unsplit(t)")
    assert r.returncode != 0
    assert "the parallel loop would keep no axis" in r.stderr
    assert after(fixed(ROTARY, "split(t,12);split(h,10);unsplit(t)"))


def test_unsplit_needs_a_split_loop() -> None:
    r = fixed(ROTARY, "unsplit(h)")
    assert r.returncode != 0
    assert "the loop is not split over cores" in r.stderr


def test_beam_offers_resplits_only_with_the_option() -> None:
    # Wide enough that a kept level-1 tree splits two loops (unsplit
    # needs a parallel axis left).
    base = "width=16;depth=2;keep=4"
    off = beam(ROTARY, base).stderr
    on = beam(ROTARY, base + ";resplit=1").stderr
    assert tried(off, "resplit", 2) == 0 and tried(off, "unsplit", 2) == 0
    assert tried(on, "resplit", 2) > 0 and tried(on, "unsplit", 2) > 0
