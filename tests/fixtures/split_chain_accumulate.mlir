// An online-softmax accumulator update: acc = acc * alpha + pv, written back
// into acc. The chain's first input is the destination's own storage, so the
// split must keep the intermediate in place (same SSA value) and allocate
// nothing: the backend lowers the broadcast multiply only in place.
module {
  func.func @accumulate(%alpha_src: memref<1x32x1xf16>, %pv_src: memref<1x32x512xf16>) {
    %acc_buf = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
    %acc = loom.semaphore_take %acc_buf : memref<1x32x512xf16> -> memref<1x32x512xf16>
    %alpha_buf = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
    %alpha = loom.semaphore_take %alpha_buf : memref<1x32x1xf16> -> memref<1x32x1xf16>
    %pv_buf = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
    %pv = loom.semaphore_take %pv_buf : memref<1x32x512xf16> -> memref<1x32x512xf16>
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%acc, %alpha, %pv : memref<1x32x512xf16>, memref<1x32x1xf16>, memref<1x32x512xf16>)
      outs(%acc : memref<1x32x512xf16>) {
    ^bb0(%a: f16, %al: f16, %p: f16, %out: f16):
      %0 = arith.mulf %a, %al : f16
      %1 = arith.addf %0, %p : f16
      linalg.yield %1 : f16
    }
    loom.semaphore_give %pv : memref<1x32x512xf16>
    loom.semaphore_give %alpha : memref<1x32x1xf16>
    loom.semaphore_give %acc : memref<1x32x512xf16>
    return
  }
}
