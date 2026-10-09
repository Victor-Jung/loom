"""Search-strategy features of the beam.

- Wave classes: a split is offered only with the fewest cores that give its
  waves per core at the picked tile (more cores with the same waves idle).
- Auto-hoist: hoists are applied after every move instead of being searched.
- Diversity: at most `permap` trees per spatial map in the frontier and the
  result, and partial trees that complete to a tree seen before are dropped.
- Fitted cost: each candidate carries its cost terms (`loom.tune.terms`).
"""

from __future__ import annotations

import re

from test_mapping_tune import FIXTURES, tune

ROTARY = FIXTURES / "tune_rotary_p00.mlir"
MATMUL = FIXTURES / "tune_matmul_p00.mlir"
HW12x10 = "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x12y10.mlir"


def beam(fixture, options: str):
    r = tune(fixture, "policy=beam", f"options=trace=1;{options}", f"hw-spec={HW12x10}")
    assert r.returncode == 0, r.stderr[-2000:]
    return r


def tried(stderr: str, kind: str, level: int = 1) -> int:
    line = re.search(rf"beam: level {level} .*", stderr).group(0)
    m = re.search(rf"{kind} (\d+)", line)
    return int(m.group(1)) if m else 0


def returned(stderr: str) -> list[tuple[str, str]]:
    return re.findall(r"  #\d+ estimate .*?  (\S.*?)  -> (\S+)", stderr)


def test_wave_classes_drop_core_counts_that_only_idle() -> None:
    base = "width=4;depth=1;keep=4;cost=traffic"
    every = tried(beam(MATMUL, base + ";waves=off").stderr, "split")
    classes = tried(beam(MATMUL, base).stderr, "split")
    assert 0 < classes < every, (classes, every)


def test_auto_hoist_applies_hoists_instead_of_searching_them() -> None:
    on = beam(ROTARY, "width=4;depth=2;keep=3;cost=traffic").stderr
    assert tried(on, "hoist", 1) == 0 and "+hoist(" in on, on[-1500:]
    off = beam(ROTARY, "width=4;depth=2;keep=3;cost=traffic;autohoist=off").stderr
    assert tried(off, "hoist", 1) > 0 and "+hoist(" not in off


def test_permap_caps_trees_per_spatial_map() -> None:
    r = beam(MATMUL, "width=8;depth=3;keep=6;cost=traffic;permap=1").stderr
    maps = [sig.split("_")[0] for _, sig in returned(r)]
    assert len(maps) == len(set(maps)), maps


def test_candidates_carry_their_cost_terms() -> None:
    r = beam(MATMUL, "width=4;depth=2;keep=2")
    terms = re.findall(r'loom\.tune\.terms = "([^"]+)"', r.stdout)
    assert terms and all(set(dict(kv.split("=") for kv in t.split(";"))) ==
                         {"dram", "core", "blocks", "mblocks", "work", "cores"} for t in terms), terms
