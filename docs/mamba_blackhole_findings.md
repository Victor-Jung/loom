# Why mamba_chunk_scan hangs on Blackhole

Three defects, established on a p150a (13x10 grid, tt-metal `ad07818`). The
first two were blockers and are now **fixed in the compiler**; the third was a
coverage gap and is **fixed in the kernel**. All three are present in the repository's own
solved configuration
(`test/mamba/solved/L1920_N32_H128_G4_D128_C192`), so none of them depend on
local edits to `kernels/mamba_chunk_scan.py`.

Flash attention is the useful control throughout: it runs correctly on the same
board and exhibits none of the three.

## 1. NOC DRAM-read congruence (Blackhole-only)

The watcher reports, on the first core to run:

```
ncrisc tried to unicast read 32 bytes to local L1[0x019040]
from DRAM[addr=0x00084060]  (invalid address alignment in NOC transaction)
```

`sanitize_noc.h` requires the L1 and DRAM addresses to agree *modulo* the
alignment, not to be absolutely aligned:

```c
if ((worker_addr & alignment_mask) != (noc_addr & alignment_mask))
```

`NOC_DRAM_READ_ALIGNMENT_BYTES` is **64 on Blackhole and 32 on Wormhole**. Here
`0x19040 % 64 == 0` but `0x84060 % 64 == 32`; under Wormhole's mask of 31 both
are congruent, which is why this never surfaced there.

The trigger is the tilization of the degenerate-dim arguments
(`memref<...x1x192xf16>`, `memref<...x192x1xf16>`): 16-element (32-byte) chunks
whose DRAM offsets are 32 apart are written into L1 face slots 512 apart, so the
two phases cannot both match -- at least one read of every pair is illegal, on
every core.

These are emitted as raw `noc_async_read`, not `noc_async_read_tile`. Flash
attention contains **zero** such reads (tile reads are 2048 bytes and therefore
always congruent); the paper's mamba config contains 14, and the small config 26.

**Fixed** in `third_party/loom2ttkernel/split_kernel.py`: raw sub-tile reads are
routed through a `loom_read_congruent` shim that reads the aligned 64-byte
superset into the destination slot (itself 64B-aligned and at least 512B wide)
and shifts the wanted bytes down in place. Reads that are already congruent go
straight through, and kernels that read only whole tiles get no helper at all.
Covered by `tests/test_dram_read_congruence.py`.

## 2. One L1 buffer assigned to two simultaneously-live tensors

`p01_explored.mlir` contains a single fused elementwise op:

```
subf(%in_0, %in_1); exp; mulf(%in, exp); mulf(that, %in_2)   // cb * exp(dA_m - dA_k) * dt_k
```

A later pass splits it into two binary ops and gives the intermediate the buffer
that already holds the broadcast of `dt_k`, producing in `p03`:

```mlir
%43 = loom.alloc [32, 192] on @L1
%58 = loom.broadcast ins(%52) outs(%44)        // W1: dt_k
linalg.generic ins(%42, %60, %59) outs(%44)    // W2: overwrites it
linalg.generic ins(%44, %58)      outs(%44)    // reads the destroyed value
```

`%58` and `%44` are the same buffer, so this has two consequences:

* **Numerically** the `dt_k` operand is destroyed before use -- the generated code
  becomes `mul_tiles(internal4, internal4, ...)`, squaring one operand.
* **On device** each write is a `cb_reserve_back`/`cb_push_back` pair and the
  buffer has a single `cb_pop_front`, so the second `cb_reserve_back` can never
  be satisfied. The packer (TRISC2) wedges there, unpack and math run ahead until
  DEST fills, and all cores stall identically at
  `CWFW, W, UPAW, MWDD, K`.

Confirmed in the paper's own config: `cb_id_internal4` has `reserve=1 push=1`
plus one `loom_unary_bcast_block` writing into it (which reserves and pushes
internally) = two pushes, against `pop=1`.

**Fixed** in `third_party/loom-dataflow/lib/passes/tt-opt/src/split_binary_scalar_chain_pass.cpp`:
`splitBinaryScalarChain` now detects when one of the second half's inputs shares
storage with the destination and gives the intermediate its own `loom.alloc`,
redirecting the chain's downstream consumers to it. Detection follows only
destination-passing inits and views, so two values compare equal exactly when
they share storage -- the pre-existing `traceToRootAllocOp` walks *all* operands
and would report whichever alloc it reached first.

`loom.broadcast`'s bufferization model is correct (`result == init`,
Equivalent/definite) and was never at fault; the pass runs after bufferization
and introduced the aliasing directly.

With the fix the kernel completes instead of hanging, and the slice it computes
matches the reference at **PCC 0.986** (vs `causal=element`). Covered by
`tests/test_split_chain_alias.py`.

## 3. Batch and head dimensions were never swept

The program wrote 12 of 768 output tiles -- 98.4% of the output was zero. This
was *not* a host defect: the generated host issues a single `ttnn.generic_op`
exactly as every other Loom kernel does, and flash attention covers its entire
output that way (PCC 0.994 on positions sampled at random across a
`[128, 15360, 512]` output). Coverage comes from the core grid plus per-core
loops, and mamba had none:

| | flash attention | mamba (before) |
|---|---|---|
| grid | `scf.parallel (10,12)` = 120 cores | `scf.parallel (6,2)` = 12 cores |
| per-core loops | `for 0..13`, `for 0..40`, `for 0..480` | none |
| writer | 4 loops around the write | 0 loops, 1 write |

For every tiled dimension the frontend emits a trip count of
`ceil(extent / tile_X)`, a per-iteration offset of `arg * tile_X`, and a slice.
For `m` and `n` the slice extent *is* `tile_X`, so striding by `tile_X` covers
the dimension. For batch and head the extent is hardcoded to **1**, because the
kernel indexes them with `.begin`, a scalar:

```python
dA_cumsum_m[tile_b.begin, tile_h.begin, tile_c.begin, tile_m, :]
```

Those loops therefore stride by `tile_X` while consuming one element, and are
only correct when `tile_X == 1`. Nothing enforced that: the frontend advertised
`@tile_b upper_bound = 2` and `@tile_h upper_bound = 32` -- the full extents --
and the solver picked the maximum, because fewer iterations model as cheaper. It
reported "Optimal T_total: 181" for a program doing 1/64 of the work.

`tile_c` is the control: it is indexed by `.begin` identically and has an
extent-1 slice identically, but its `hl.tile` call passed `block_size=1`, so its
symbol was pinned, its loop ran `ceil(2/1) = 2` times and it covered both chunks
correctly.

**Fixed** in `kernels/mamba_chunk_scan.py` by pinning the two free tiles:

```python
for tile_b in hl.tile(batch, block_size=1):
    for tile_h in hl.tile(nheads, block_size=1):
```

The frontend then reports `upper_bound = 1` for both, the solver's only choice
is 1, and `p03` gains `scf.for 0..2` (batch) and `scf.for 0..32` (heads) inside
the 6x2 grid: 12 x 2 x 32 = 768 tiles, the whole output. Covered by
`tests/test_mamba_tile_domains.py`.

Note the solver's cost for the correct program is 9,168 units against 181 for
the broken one -- the old "optimum" was cheap precisely because it computed
almost nothing.

This is a kernel-level fix for a trap the frontend still allows. The general
hardening is to derive a tile symbol's domain from how its induction variable is
used: a tile consumed only via `.begin` can only ever have block size 1, and
advertising the full extent hands the solver a domain in which almost every
point silently computes a fraction of the answer. Any kernel using `.begin` on
an unpinned `hl.tile` has the same trap. That generalization is not done.

## Result

From the fixed kernel, through the full pipeline, on a p150a:

```
got: mean -0.00010  std 0.53111  absmax 5.84375  zeros 0.0%
ref: mean -0.00025  std 0.53328  absmax 5.87889  zeros 0.0%
per-(b,h) PCC: min 0.815 median 0.987 max 0.999; 60/64 rows >0.9
PCC 0.995254   max|err|/max|ref| 0.0666   PASS
```

The four rows below 0.9 are heads 9 and 22 in both batches. Those are the two
smallest `|D[h]|` of 32 (0.097 and 0.135); since `xD = x * D[h]`, their outputs
are the smallest in magnitude and carry the largest relative bf16 error. It is a
property of the test data, not of the generated code.

## Running it

Both fixes are in the compiler, so the normal flow is enough. The IRs under
`test/` are generated artifacts (gitignored); regenerate `p03` from the cached
`p01_explored.mlir` plus the solved block sizes in `constraints/solver.log`, then
lower and run:

```sh
SPLIT_KERNEL_OUTPUT_DIR=$PWD/tmp_output/k_mamba \
    ./third_party/loom2ttkernel/lower.sh test/mamba/small/IRs/p03_bufferized.mlir 1
cp tmp_output/k_mamba/{reader,compute,writer}.cpp \
   $TT_METAL_HOME/tt_metal/programming_examples/mlir_matmul_simple/kernels/
python experiments/host/mamba_verify.py --kernels tmp_output/k_mamba \
    --ir test/mamba/small/IRs/p03_bufferized.mlir --causal element
```

`experiments/scripts/fix_cb_alias_mlir.py` and `apply_noc_shim.py` remain as the
standalone reproductions used to isolate each defect before either was fixed;
they are not needed by the normal flow.

Note that `loom-dataflow` is installed into `.venv` as a scikit-build-core
editable package that does **not** rebuild on import, so a change to a pass needs
an explicit reinstall before the Python pipeline picks it up:

```sh
SKBUILD_BUILD_DIR=$PWD/third_party/loom-dataflow/build-py310 \
uv pip install --no-deps --no-build-isolation --force-reinstall \
  --config-settings=cmake.define.ADLDialect_DIR=$PWD/third_party/adl-dialect/build/install/lib/cmake/ADLDialect \
  --config-settings=cmake.define.MLIR_DIR=/opt/ttmlir-toolchain/lib/cmake/mlir \
  --config-settings=cmake.define.LLVM_DIR=/opt/ttmlir-toolchain/lib/cmake/llvm \
  -e third_party/loom-dataflow
```

Run under `TT_METAL_WATCHER=1` to turn a NOC violation into a named error instead
of a silent hang, and `TT_METAL_DPRINT_CORES=0,0 TT_METAL_DPRINT_RISCVS=TR0,TR1,TR2`
to see which of unpack/math/pack is stuck. A wedged board needs `tt-smi -r`.
