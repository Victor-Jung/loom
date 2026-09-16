#!/usr/bin/env bash
# 3 frontend variants x 14 GEMM shapes, solver ENABLED (best-vs-best).
cd /workspace/loom
: > variant_results.txt
for v in base nm splitk; do
  case $v in base) script=kernels/matmul.py;; *) script=kernels/matmul_$v.py;; esac
  for cfg in $(find kernels/config_files/matmul/blackhole -name '*.json' | sort); do
    s=$(basename "${cfg%.json}")
    out="test/variants/$v/$s"
    python3 -c "
import json;c=json.load(open('$cfg'));c.pop('assigned_block_size',None)
c['output_path']='$out';json.dump(c,open('/tmp/c.json','w'))"
    t0=$SECONDS
    if timeout 2400 env -u VIRTUAL_ENV uv run python "$script" "-$s" --config /tmp/c.json \
         --njobs 16 --debug > "/tmp/${v}_${s}.log" 2>&1; then
      cands=$(grep -cE '^\s*func\.func @' "$out/IRs/p01_explored.mlir" 2>/dev/null || echo 0)
      echo "PASS $v $s $((SECONDS-t0))s cands=$cands" >> variant_results.txt
    else
      echo "FAIL $v $s $((SECONDS-t0))s :: $(grep -iE 'error|ValueError' /tmp/${v}_${s}.log | head -1 | cut -c1-90)" >> variant_results.txt
    fi
  done
done
echo "=== matrix done ===" >> variant_results.txt
