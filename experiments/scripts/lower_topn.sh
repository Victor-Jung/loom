#!/usr/bin/env bash
# Lower ALL func groups from one IR: run the MLIR chain once, then split per index.
set -u
cd /workspace/loom
IR=test/topn/base/M2048_N2048_K16384/IRs/p03_bufferized.mlir
N=$(grep -cE '^\s*func\.func @' "$IR")
echo "funcs=$N"
# one full pass (index 1) to produce tmp_output/tmp_mlir_files/kernel.cpp
SPLIT_KERNEL_OUTPUT_DIR=/workspace/loom/tmp_output/topn/f1 \
  ./third_party/loom2ttkernel/lower.sh "$IR" 1 >/tmp/l1.log 2>&1 || { echo "BASE LOWER FAILED"; tail -3 /tmp/l1.log; exit 1; }
CPP=tmp_output/tmp_mlir_files/kernel.cpp
echo "kernel.cpp groups=$(grep -cE '^// .*__(host_cpp|compute|reader|writer)' "$CPP" 2>/dev/null || echo '?')"
ok=0
for i in $(seq 1 "$N"); do
  out=tmp_output/topn/f$i
  mkdir -p "$out"
  if python3 third_party/loom2ttkernel/split_kernel.py "$CPP" --output-dir "$out" --func-index "$i" >/dev/null 2>&1 \
     && [ -f "$out/host_ttnn.py" ]; then
    ok=$((ok+1))
  else
    echo "SPLITFAIL $i" >> /workspace/loom/topn_lower.txt
  fi
done
echo "lowered_ok=$ok / $N"
