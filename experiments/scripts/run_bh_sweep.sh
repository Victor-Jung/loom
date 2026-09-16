#!/usr/bin/env bash
# Compile-run every blackhole config through the Loom pipeline (stages 0-4).
cd /workspace/loom
pass=0; fail=0
: > /workspace/loom/sweep_results.txt
for kernel in matmul flash_attention mqa_decode mamba_chunk_scan; do
  for cfg in $(find kernels/config_files/$kernel/blackhole -name '*.json' 2>/dev/null | sort); do
    setting=$(basename "${cfg%.json}")
    hw=$(python3 -c "import json;print(json.load(open('$cfg')).get('hw_spec','?'))")
    if [ ! -f "$hw" ]; then
      echo "SKIP_NOMESH $kernel/$setting" >> sweep_results.txt; continue
    fi
    if timeout 900 env -u VIRTUAL_ENV uv run python "kernels/$kernel.py" "-$setting" \
         --config "$cfg" --njobs 8 --debug > "/tmp/${kernel}_${setting}.log" 2>&1; then
      echo "PASS $kernel/$setting" >> sweep_results.txt; pass=$((pass+1))
    else
      echo "FAIL $kernel/$setting :: $(grep -iE 'error|Error|assert' /tmp/${kernel}_${setting}.log | head -1 | cut -c1-140)" >> sweep_results.txt
      fail=$((fail+1))
    fi
  done
done
echo "=== compile sweep: $pass pass, $fail fail ===" >> sweep_results.txt
