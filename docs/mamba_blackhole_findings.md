# Why mamba_chunk_scan hangs on Blackhole

Three defects, established on a p150a (13x10 grid, tt-metal `ad07818`). The first
two are blockers; the third is a coverage gap. All three are present in the
repository's own solved configuration
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

*Fix belongs in* the TTKernel lowering that emits raw sub-tile DRAM reads.
`experiments/scripts/apply_noc_shim.py` is a working reference: it reads the
64B-aligned superset into the (>=512B, 64B-aligned) destination slot and shifts
the wanted bytes down in place.

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

Separating the buffers (`experiments/scripts/fix_cb_alias_mlir.py`) removes the
deadlock: the kernel completes, and the slice it computes matches the reference
at **PCC 0.986** (slope 0.973, vs `causal=element`).

*Fix belongs in* the pass that splits the fused elementwise op; the intermediate
needs its own allocation. `loom.broadcast`'s bufferization model is correct
(`result == init`, Equivalent/definite) and is not at fault.

## 3. Batch and head dimensions are never swept

The program writes 12 of 768 output tiles (measured: 98.4% of the output is
zero). This is *not* a host defect -- the generated host issues a single
`ttnn.generic_op`, exactly as every other Loom kernel does, and flash attention
covers its entire output that way (PCC 0.994 on positions sampled at random
across a `[128, 15360, 512]` output).

Coverage normally comes from the core grid plus per-core loops:

| | flash attention | mamba (small) |
|---|---|---|
| grid | `scf.parallel (10,12)` = 120 cores | `scf.parallel (6,2)` = 12 cores |
| per-core loops | `for 0..13`, `for 0..40`, `for 0..480` | none |
| writer | 4 loops around the write | 0 loops, 1 write |

The Helion source does have `for tile_b in hl.tile(batch)` and
`for tile_h in hl.tile(nheads)`, and the solved config sets `tile_b=2`,
`tile_h=32` -- one tile spanning the *entire* batch and head extent. Those loops
therefore become single-trip and fold away, but the body is emitted for a single
element rather than for the whole tile. The generated reader contains no batch or
head stride arithmetic at all (only `6144`, the chunk stride *within* one (b,h)
slice), so no choice of base address would let a relaunch reach a different head:
making the host launch repeatedly could not fix this as the code stands.

The paper's solved config has the same `tile_b=2`/`tile_h=32` collapse and a
similarly shallow nest (grid 6x10 plus one `scf.for 0..2`). Its coverage has not
been measured on device.

## Reproducing

```sh
# 1. separate the aliased buffers in the bufferized IR
experiments/scripts/fix_cb_alias_mlir.py \
    test/mamba/small/IRs/p03_bufferized.mlir test/mamba/small/IRs/p03_fixed.mlir

# 2. lower it
SPLIT_KERNEL_OUTPUT_DIR=$PWD/tmp_output/k_mamba_fixed \
    ./third_party/loom2ttkernel/lower.sh test/mamba/small/IRs/p03_fixed.mlir 1

# 3. make the sub-tile DRAM reads congruent on Blackhole
experiments/scripts/apply_noc_shim.py tmp_output/k_mamba_fixed/reader.cpp

# 4. stage and run
cp tmp_output/k_mamba_fixed/{reader,compute,writer}.cpp \
   $TT_METAL_HOME/tt_metal/programming_examples/mlir_matmul_simple/kernels/
python experiments/host/mamba_verify.py --kernels tmp_output/k_mamba_fixed \
    --ir test/mamba/small/IRs/p03_fixed.mlir --causal element
```

Run under `TT_METAL_WATCHER=1` to turn a NOC violation into a named error instead
of a silent hang, and `TT_METAL_DPRINT_CORES=0,0 TT_METAL_DPRINT_RISCVS=TR0,TR1,TR2`
to see which of unpack/math/pack is stuck. A wedged board needs `tt-smi -r`.
