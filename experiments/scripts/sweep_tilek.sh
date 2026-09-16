#!/usr/bin/env bash
# Direct block-size sweep, bypassing the cost model via assigned_block_size.
set -u
cd /workspace/loom
KEY='_matmul__x12_y1y10__d0i1_d1i1_d2i0__f01__dim_x_level0_bc12_dim_y_level1_bc10_n'
: > tilek_compile.txt
for tk in 128 256 512 1024 2048; do
  for tm in 224; do for tn in 224; do
    tag="k${tk}_m${tm}_n${tn}"
    out="test/tilek/$tag"
    python3 -c "
import json
c=json.load(open('kernels/config_files/matmul/blackhole/M2048_N2048_K16384.json'))
c['assigned_block_size']={'$KEY':{'tile_m':$tm,'tile_n':$tn,'tile_k':$tk}}
c['output_path']='$out'
json.dump(c,open('/tmp/tk.json','w'))"
    if timeout 900 env -u VIRTUAL_ENV uv run python kernels/matmul.py -M2048_N2048_K16384 \
         --config /tmp/tk.json --njobs 16 --debug >/tmp/tk.log 2>&1; then
      ir="$out/IRs/p03_bufferized.mlir"
      if SPLIT_KERNEL_OUTPUT_DIR="/workspace/loom/tmp_output/tk_$tag" \
           timeout 900 ./third_party/loom2ttkernel/lower.sh "$ir" 1 >/dev/null 2>&1; then
        echo "OK $tag" >> tilek_compile.txt
      else echo "LOWERFAIL $tag" >> tilek_compile.txt; fi
    else
      echo "COMPILEFAIL $tag :: $(grep -iE 'error|ValueError' /tmp/tk.log|head -1|cut -c1-70)" >> tilek_compile.txt
    fi
  done; done
done
cat tilek_compile.txt
