"""The binary-chain splitter must not reuse a destination that is still live.

`splitBinaryScalarChain` parks the chain's intermediate in the original
destination buffer. That is safe only when none of the second half's inputs
lives in that same buffer. When one does -- as in mamba_chunk_scan, where the
dt_k broadcast writes the very buffer the chain is destined for -- the first
half's write destroys a value the second half still reads.

The pass runs after bufferization, so nothing downstream catches it, and once
lowered the buffer also takes two cb_push_back against a single cb_pop_front:
the packer then blocks forever in a cb_reserve_back that can never be satisfied,
which on device is a silent hang that wedges the board.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "split_chain_aliased_dest.mlir"
TT_OPT = Path(
    os.environ.get(
        "LOOM_TT_OPT",
        ROOT / "third_party/loom-dataflow/build/tool/tt-opt/single_stage/tt-opt",
    )
)


@pytest.fixture(scope="module")
def split_output() -> str:
    if not TT_OPT.exists():
        pytest.skip(f"tt-opt not built at {TT_OPT}; set LOOM_TT_OPT to override")
    result = subprocess.run(
        [str(TT_OPT), f"--input={FIXTURE}"],
        capture_output=True, text=True, timeout=300, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def _generics(text: str) -> list[tuple[str, str]]:
    """(ins, outs) operand lists of each linalg.generic, in order."""
    return [
        (m.group(1), m.group(2))
        for m in re.finditer(r"linalg\.generic.*?ins\(([^)]*)\)\s*outs\(([^)]*)\)", text)
    ]


def test_chain_is_split_into_two_ops(split_output: str) -> None:
    assert len(_generics(split_output)) == 2


def test_intermediate_gets_its_own_buffer(split_output: str) -> None:
    """Two allocations: one still holding the broadcast, one for the chain."""
    assert split_output.count("loom.alloc") == 2


def test_broadcast_result_survives_to_its_consumer(split_output: str) -> None:
    """The second op must still read the broadcast, and neither op may write the
    buffer the broadcast produced into."""
    bcast = re.search(r"(%\w+) = loom\.broadcast ins\([^)]*\) outs\((%\w+)", split_output)
    assert bcast, split_output
    bcast_result, bcast_dest = bcast.group(1), bcast.group(2)

    generics = _generics(split_output)
    assert bcast_result in generics[1][0], "second op no longer reads the broadcast"
    for ins, outs in generics:
        dest = outs.split(":")[0].strip()
        assert dest != bcast_dest, f"chain still writes the broadcast's buffer {dest}"
