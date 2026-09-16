#!/usr/bin/env bash
# Phase A (container): lower every successfully-compiled config to its own kernels dir.
cd /workspace/loom
: > lower_results.txt
for ir in $(find test -name p03_bufferized.mlir | sort); do
  # test/<kernel>/blackhole/<setting>/IRs/p03_bufferized.mlir
  name=$(echo "$ir" | sed 's|^test/||; s|/IRs/p03_bufferized.mlir$||; s|/|_|g')
  outdir="tmp_output/k_${name}"
  mkdir -p "$outdir"
  if SPLIT_KERNEL_OUTPUT_DIR="/workspace/loom/$outdir" \
     timeout 600 ./third_party/loom2ttkernel/lower.sh "$ir" 1 > "/workspace/loom/$outdir/lower.log" 2>&1; then
    echo "PASS $name $ir" >> lower_results.txt
  else
    echo "FAIL $name :: $(grep -iE 'error' /workspace/loom/$outdir/lower.log | head -1 | cut -c1-130)" >> lower_results.txt
  fi
done
echo "=== lowering done ===" >> lower_results.txt
