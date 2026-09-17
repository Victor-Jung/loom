// A fused binary chain whose destination buffer also holds one of its inputs:
// %bc broadcasts into %sem, and the generic both reads %bc and writes %sem.
// Splitting this chain must not park the intermediate in %sem, or the first
// half destroys %bc before the second half reads it.
module {
  func.func @aliased_chain(%src: memref<1x192xf16>, %a: memref<32x192xf16>,
                           %m: memref<32x192xf16>, %k: memref<32x192xf16>) {
    %buf = loom.alloc [32, 192] on @L1 : memref<32x192xf16>
    %sem = loom.semaphore_take %buf : memref<32x192xf16> -> memref<32x192xf16>
    %bc = loom.broadcast ins(%src : memref<1x192xf16>) outs(%sem : memref<32x192xf16>) dim(0) -> memref<32x192xf16>
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%a, %m, %k, %bc : memref<32x192xf16>, memref<32x192xf16>, memref<32x192xf16>, memref<32x192xf16>)
      outs(%sem : memref<32x192xf16>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
      %0 = arith.subf %in_0, %in_1 : f16
      %1 = math.exp %0 : f16
      %2 = arith.mulf %in, %1 : f16
      %3 = arith.mulf %2, %in_2 : f16
      linalg.yield %3 : f16
    }
    loom.semaphore_give %sem : memref<32x192xf16>
    return
  }
}
