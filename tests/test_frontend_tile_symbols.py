"""Distinct tile loops that share a name get distinct symbols.

`softmax_twopass` tiles `n` three times. The frontend used to emit
`%tile_n = loom.sym @tile_n` once per loop and the module did not parse;
the symbols are now `tile_n`, `tile_n_1`, `tile_n_2`, each with its own
module attribute, and the mapping program parses.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

helion = pytest.importorskip("helion")
loom_pipeline = pytest.importorskip("loom_pipeline")


def test_repeated_tile_loops_get_unique_symbols() -> None:
    from kernels.softmax_twopass import SoftmaxTwoPass

    p00 = SoftmaxTwoPass.generate_mlir()
    syms = re.findall(r'"loom\.sym"\(\) \{[^}]*symbol_ref = @(\w+)', p00)
    assert len(syms) == len(set(syms)), syms
    assert {"tile_n", "tile_n_1", "tile_n_2"} <= set(syms), syms
    for name in ("tile_n", "tile_n_1", "tile_n_2"):
        assert f"loom.{name} = " in p00
    # The program parses: the identity tuner round-trips it.
    assert "func.func" in loom_pipeline.run_mapping_tune(p00, "identity")
