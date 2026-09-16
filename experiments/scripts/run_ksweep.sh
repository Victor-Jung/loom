#!/usr/bin/env bash
# K sweep: 3 frontend variants x K in {256,2048,16384}, M=N=2048, solver top-1.
cd /workspace/loom
: > ksweep_compile.txt
for v in base nm splitk; do
  case $v in base) script=kernels/matmul.py;; *) script=kernels/matmul_$v.py;; esac
  for s in M2048_N2048_K256 M2048_N2048_K2048 M2048_N2048_K16384; do
    out="test/ksweep/$v/$s"
    python3 -c "
import json;c=json.load(open('kernels/config_files/matmul/blackhole/$s.json'))
c.pop('assigned_block_size',None);c['output_path']='$out';json.dump(c,open('/tmp/c.json','w'))"
    t0=$SECONDS
    if timeout 2400 env -u VIRTUAL_ENV uv run python "$script" "-$s" --config /tmp/c.json \
        --njobs 16 --debug --topk-candidates 1 --topk-block-size 1 > "/tmp/${v}_${s}.log" 2>&1; then
      ct=$((SECONDS-t0))
      ir="$out/IRs/p03_bufferized.mlir"
      nf=$(grep -cE '^\s*func\.func @' "$ir")
      cand=$(grep -cE '^\s*func\.func @' "$out/IRs/p01_explored.mlir" 2>/dev/null || echo 0)
      bs=$(grep -oE 'tile_k[0-9]+__tile_m[0-9]+__tile_n[0-9]+' "$ir" | head -1)
      if SPLIT_KERNEL_OUTPUT_DIR="/workspace/loom/tmp_output/ks_${v}_${s}" \
           timeout 900 ./third_party/loom2ttkernel/lower.sh "$ir" 1 > /tmp/low.log 2>&1; then
        echo "OK $v $s compile=${ct}s funcs=$nf cands=$cand $bs" >> ksweep_compile.txt
      else
        echo "LOWERFAIL $v $s compile=${ct}s :: $(grep -i error /tmp/low.log|head -1|cut -c1-80)" >> ksweep_compile.txt
      fi
    else
      echo "COMPILEFAIL $v $s $((SECONDS-t0))s :: $(grep -iE 'error|ValueError' /tmp/${v}_${s}.log|head -1|cut -c1-80)" >> ksweep_compile.txt
    fi
  done
done
echo "=== ksweep compile done ===" >> ksweep_compile.txt
