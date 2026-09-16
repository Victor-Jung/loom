#!/usr/bin/env bash
set -u
cd /workspace/loom
KEY='_matmul__x12_y1y10__d0i1_d1i1_d2i0__f01__dim_x_level0_bc12_dim_y_level1_bc10_n'
: > mn_compile.txt
for tm in 112 176 224 288 352; do for tn in 112 176 224 288 352; do
  tag="k256_m${tm}_n${tn}"; out="test/mn/$tag"
  python3 -c "
import json
c=json.load(open('kernels/config_files/matmul/blackhole/M2048_N2048_K16384.json'))
c['assigned_block_size']={'$KEY':{'tile_m':$tm,'tile_n':$tn,'tile_k':256}}
c['output_path']='$out'; json.dump(c,open('/tmp/mn.json','w'))"
  if timeout 600 env -u VIRTUAL_ENV uv run python kernels/matmul.py -M2048_N2048_K16384 \
       --config /tmp/mn.json --njobs 16 --debug >/tmp/mn.log 2>&1 \
     && SPLIT_KERNEL_OUTPUT_DIR="/workspace/loom/tmp_output/mn_$tag" \
        timeout 600 ./third_party/loom2ttkernel/lower.sh "$out/IRs/p03_bufferized.mlir" 1 >/dev/null 2>&1; then
    echo "OK $tag" >> mn_compile.txt
  else
    echo "FAIL $tag" >> mn_compile.txt
  fi
done; done
grep -c '^OK' mn_compile.txt
