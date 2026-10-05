"""A program that needs more than 32 L1 buffers is rejected at memory binding.

Each `loom.alloc` becomes one circular buffer and a Tensix core has 32 of
them. The check runs once per function in step 1, before the spatial
enumeration multiplies the function into variants and long before lowering
would discover it. A sum of N input tiles needs N input buffers plus the
output: 4 inputs bind, 34 do not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HW_SPEC = ROOT / "third_party/loom-mlar/tests/2d_mesh/2d_mesh_torus_x1y1.mlir"

loom_pipeline = pytest.importorskip("loom_pipeline")


def sum_of_inputs_p00(n: int, m: int = 256, cols: int = 512) -> str:
    """Stage-00 program out[m, n] = sum_i x_i[m, n] with one tile per input."""
    args = ", ".join(f"%x{i}: memref<{m}x{cols}xf16>" for i in range(n))
    maps = ", ".join(["affine_map<(d0, d1) -> (d0, d1)>"] * (n + 1))
    loads = "\n".join(
        f"        %sv{i} = memref.subview %x{i}[%om, %on] [%tm, %tn] [1, 1] : memref<{m}x{cols}xf16> to memref<?x?xf16, strided<[{cols}, 1], offset: ?>>\n"
        f"        %t{i} = bufferization.to_tensor %sv{i} restrict : memref<?x?xf16, strided<[{cols}, 1], offset: ?>> to tensor<?x?xf16>"
        for i in range(n)
    )
    ins = ", ".join(f"%t{i}" for i in range(n))
    in_types = ", ".join(["tensor<?x?xf16>"] * n)
    block_args = ", ".join(f"%in{i}: f16" for i in range(n))
    adds = "\n".join(
        f"          %s{i} = arith.addf %s{i-1}, %in{i} : f16" for i in range(1, n)
    )
    return f"""
module attributes {{loom.tile_m = {{asure_divisible = false, is_reduction = false, upper_bound = {m} : index}}, loom.tile_n = {{asure_divisible = false, is_reduction = false, upper_bound = {cols} : index}}}} {{
  func.func @sum_inputs({args}, %out_arg: memref<{m}x{cols}xf16>) {{
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cm = arith.constant {m} : index
    %cn = arith.constant {cols} : index
    %tm = "loom.sym"() {{asure_divisible = false, is_reduction = false, symbol_ref = @tile_m, upper_bound = {m} : index}} : () -> index
    %tn = "loom.sym"() {{asure_divisible = false, is_reduction = false, symbol_ref = @tile_n, upper_bound = {cols} : index}} : () -> index
    %trips_m = arith.ceildivui %cm, %tm : index
    affine.parallel (%im) = (0) to (symbol(%trips_m)) {{
      %trips_n = arith.ceildivui %cn, %tn : index
      scf.for %in_ = %c0 to %trips_n step %c1 {{
        %om = arith.muli %im, %tm : index
        %on = arith.muli %in_, %tn : index
{loads}
        %empty = tensor.empty(%tm, %tn) : tensor<?x?xf16>
        %r = linalg.generic {{indexing_maps = [{maps}], iterator_types = ["parallel", "parallel"]}} ins({ins} : {in_types}) outs(%empty : tensor<?x?xf16>) {{
        ^bb0({block_args}, %o: f16):
          %s0 = arith.addf %in0, %in0 : f16
{adds}
          linalg.yield %s{n-1} : f16
        }} -> tensor<?x?xf16>
        %svo = memref.subview %out_arg[%om, %on] [%tm, %tn] [1, 1] : memref<{m}x{cols}xf16> to memref<?x?xf16, strided<[{cols}, 1], offset: ?>>
        %buf = bufferization.to_buffer %r : tensor<?x?xf16> to memref<?x?xf16, strided<[{cols}, 1], offset: ?>>
        memref.copy %buf, %svo : memref<?x?xf16, strided<[{cols}, 1], offset: ?>> to memref<?x?xf16, strided<[{cols}, 1], offset: ?>>
      }}
    }}
    return
  }}
}}
"""


def explore(p00: str) -> str:
    if not HW_SPEC.exists():
        pytest.skip(f"hardware spec missing: {HW_SPEC}")
    out, _ = loom_pipeline.run_exploration(
        input_mlir=p00, hw_spec_file=str(HW_SPEC), produce_etg=False
    )
    return out


def test_four_inputs_bind() -> None:
    out = explore(sum_of_inputs_p00(4))
    assert out.count("loom.alloc") >= 5


def test_thirty_four_inputs_are_rejected_at_memory_binding() -> None:
    with pytest.raises(RuntimeError) as err:
        explore(sum_of_inputs_p00(34))
    assert "circular buffers" in str(err.value), str(err.value)
