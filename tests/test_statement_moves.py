"""Statement moves of the mapping-program tuner: `hoist` and `sink`.

`hoist(S, L)` moves statement S, with the index computations, views and loads
it needs from the bodies between it and loop L, right before L. It is legal
when every loop from S up to L is temporal, the moved statements are pure,
views, or loads of arguments nothing under L writes, none of them uses an
index or iteration value bound by L or a loop inside it, and none is a buffer
a statement left in the loop overwrites. `sink(S, L)` is the inverse: S and
the feeders only it uses move to the start of L's body when every use lies
inside it. The estimate charges a load for every temporal loop around it, so
hoisting an invariant load shows up as saved traffic, and loads above a loop
no longer stop that loop from being split over cores.
"""

from __future__ import annotations

import re

from test_mapping_tune import FIXTURES, tune

ROTARY = FIXTURES / "tune_rotary_p00.mlir"      # S12 x load, S14 cos, S16 sin, S17 rot, S26 store
MASK = FIXTURES / "tune_mask_softmax_p00.mlir"  # S20 mask load, S25 amax init fill
READ_WRITE = FIXTURES / "tune_read_write_p00.mlir"  # S4 loads b (written in the loop), S6 loads c
HW12x10 = "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x12y10.mlir"


def fixed(fixture, schedule: str, mesh: bool = True):
    opts = ["policy=fixed", f"options={schedule}", "dump-tree"]
    if mesh:
        opts.append(f"hw-spec={HW12x10}")
    return tune(fixture, *opts)


def after(result) -> tuple[str, float]:
    m = re.search(r"loop tree \(after, (\S+), estimate ([0-9.e+]+)\)", result.stderr)
    return m.group(1), float(m.group(2))


def test_hoisting_invariant_loads_out_of_the_head_loop_cuts_traffic() -> None:
    base = fixed(ROTARY, "split(h,4)")
    rot = fixed(ROTARY, "hoist(S17,h);split(h,4)")
    all3 = fixed(ROTARY, "hoist(S17,h);hoist(S14,h);hoist(S16,h);split(h,4)")
    for r in (base, rot, all3):
        assert r.returncode == 0, r.stderr[-2000:]
    (sig0, e0), (sig1, e1), (sig3, e3) = after(base), after(rot), after(all3)
    assert sig1 == sig0 + "_h17o3" and sig3 == sig0 + "_h14o3_h16o3_h17o3", (sig0, sig1, sig3)
    assert e3 < e1 < e0, (e0, e1, e3)
    # The rot load is emitted above the head loop, the x load inside it.
    ir = all3.stdout
    x_view = ir.index("memref.subview %arg0")
    head_loop = ir.rindex("scf.for", 0, x_view)  # the loop around the x load
    rot_load = ir.index("bufferization.to_tensor %arg3")
    assert rot_load < head_loop < x_view


def test_move_order_does_not_change_the_tree() -> None:
    a, b = fixed(ROTARY, "hoist(S17,h);split(h,4)"), fixed(ROTARY, "split(h,4);hoist(S17,h)")
    assert after(a) == after(b)


def test_sink_undoes_a_hoist() -> None:
    assert after(fixed(ROTARY, "hoist(S17,h);sink(S17,h);split(h,4)")) == after(fixed(ROTARY, "split(h,4)"))


def test_illegal_moves_are_refused_with_a_reason() -> None:
    cases = [
        (ROTARY, "hoist(S12,h)", "depends on an index or iteration value of h"),  # x[h, t]
        (ROTARY, "hoist(S26,h)", "memory effects: memref.copy"),                   # the store
        (MASK, "hoist(S25,h)", "overwrites inside h"),                             # reduction init
        (MASK, "sink(S12,h)", "a use of the statement lies outside h"),            # h's bound
        (READ_WRITE, "hoist(S4,L0)", "a loaded argument is written inside L0"),
    ]
    for fixture, schedule, reason in cases:
        r = fixed(fixture, schedule, mesh=fixture is not READ_WRITE)
        assert r.returncode != 0 and reason in r.stderr, (schedule, r.stderr[-1500:])
    ok = fixed(READ_WRITE, "hoist(S6,L0)", mesh=False)
    assert ok.returncode == 0 and after(ok)[0] == "L0_h6of", ok.stderr[-1500:]


def test_a_load_hoisted_above_a_loop_does_not_block_its_split() -> None:
    # The mask load sits in b's body once hoisted out of h; splitting h moves
    # its axis to the parallel loop, and every core simply runs the load.
    r = fixed(MASK, "hoist(S20,h);split(h,4)")
    assert r.returncode == 0, r.stderr[-2000:]
    assert after(r)[0].endswith("_h20o1"), after(r)


def test_beam_hoists_the_invariant_loads() -> None:
    r = tune(ROTARY, "policy=beam", "options=width=8;depth=5;keep=1", f"hw-spec={HW12x10}", "dump-tree")
    assert r.returncode == 0, r.stderr[-2000:]
    sig, _ = after(r)
    for stmt in ("14", "16", "17"):  # cos, sin, rot
        assert f"_h{stmt}o" in sig, sig
    # Without statement moves the same search keeps them in the head loop.
    off = tune(ROTARY, "policy=beam", "options=width=8;depth=5;keep=1;stmt=off",
               f"hw-spec={HW12x10}", "dump-tree")
    assert off.returncode == 0 and not re.search(r"_h\d+o", after(off)[0]), after(off)
