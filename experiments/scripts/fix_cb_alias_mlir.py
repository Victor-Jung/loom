#!/usr/bin/env python3
"""Apply the validated buffer-separation fix to a bufferized mamba p03 MLIR.

This is a *validation aid*, not a compiler fix. It reproduces by hand what the
compiler should do, so the two Blackhole blockers can be confirmed independently
of each other on hardware.

The defect it works around: a pass splits the fused elementwise

    cb * exp(dA_m - dA_k) * dt_k

into two binary ops and writes the intermediate into the same buffer that holds
the broadcast of dt_k. In memref terms that is a write-after-write hazard; in the
circular-buffer model it is also fatal, because each write is a
cb_reserve_back/cb_push_back pair and the buffer has only one cb_pop_front, so
the second cb_reserve_back can never be satisfied and the packer wedges.

Usage:  fix_cb_alias_mlir.py p03_bufferized.mlir p03_fixed.mlir
"""
import re
import sys


def main(src: str, dst: str) -> int:
    lines = open(src).read().split("\n")

    # Locate the broadcast whose destination is later overwritten, the two
    # generics that overwrite it, and the matmul that consumes the result.
    bcast = None
    for i, ln in enumerate(lines):
        m = re.search(r"%(\w+) = loom\.broadcast ins\(%\w+ : memref<1x\d+xf16>\) "
                      r"outs\(%(\w+) :", ln)
        if m and f"outs(%{m.group(2)} :" in "\n".join(lines[i + 1:i + 20]):
            bcast, victim = m.group(1), m.group(2)
            break
    if bcast is None:
        print("no clobbered broadcast found; IR may already be fixed", file=sys.stderr)
        return 1

    new, take = "%fix_alloc", "%fix"
    n = 0
    for i, ln in enumerate(lines):
        if "loom.broadcast" in ln:
            continue                      # the broadcast keeps the original buffer
        if f"outs(%{victim} :" in ln or f"ins(%{victim}, %{bcast} :" in ln \
                or f"ins(%{victim}, %" in ln and "loom.matmul" in ln:
            lines[i] = ln.replace(f"outs(%{victim} :", f"outs({take} :") \
                          .replace(f"ins(%{victim}, ", f"ins({take}, ")
            n += 1

    # Declare the new buffer next to the one it relieves, and release it.
    for i, ln in enumerate(lines):
        if re.search(rf"%{victim} = loom\.semaphore_take ", ln):
            ind = re.match(r"\s*", ln).group(0)
            ty = re.search(r"-> (memref<[^>]*>)", ln).group(1)
            lines.insert(i + 1, f"{ind}{new} = loom.alloc [32, 192] on @L1 : {ty}")
            lines.insert(i + 2, f"{ind}{take} = loom.semaphore_take {new} : {ty} -> {ty}")
            break
    for i, ln in enumerate(lines):
        if f"loom.semaphore_give %{victim} " in ln:
            ind = re.match(r"\s*", ln).group(0)
            ty = re.search(r": (memref<[^>]*>)", ln).group(1)
            lines.insert(i + 1, f"{ind}loom.semaphore_give {take} : {ty}")
            break

    open(dst, "w").write("\n".join(lines))
    print(f"separated %{victim} from the %{bcast} broadcast at {n} sites -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
