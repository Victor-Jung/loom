#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @_prod(%x: memref<256x4096xf16>, %out: memref<256x128xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c4096 = arith.constant 4096 : index
    %one = arith.constant 1.000000e+00 : f16
    %tm = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %tk = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_k, upper_bound = 128 : index} : () -> index
    %nm = arith.ceildivui %c256, %tm : index
    affine.parallel (%i) = (0) to (symbol(%nm)) {
      %e = tensor.empty(%tm) : tensor<?x128xf16>
      %init = linalg.fill ins(%one : f16) outs(%e : tensor<?x128xf16>) -> tensor<?x128xf16>
      %nk = arith.ceildivui %c4096, %tk : index
      %r = scf.for %k = %c0 to %nk step %c1 iter_args(%acc = %init) -> (tensor<?x128xf16>) {
        %row = arith.muli %i, %tm : index
        %col = arith.muli %k, %tk : index
        %v = memref.subview %x[%row, %col] [%tm, 128] [1, 1] : memref<256x4096xf16> to memref<?x128xf16, strided<[4096, 1], offset: ?>>
        %t = bufferization.to_tensor %v restrict : memref<?x128xf16, strided<[4096, 1], offset: ?>> to tensor<?x128xf16>
        %p = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%acc, %t : tensor<?x128xf16>, tensor<?x128xf16>) outs(%e : tensor<?x128xf16>) {
        ^bb0(%a: f16, %b: f16, %o: f16):
          %m = arith.mulf %a, %b : f16
          linalg.yield %m : f16
        } -> tensor<?x128xf16>
        scf.yield %p : tensor<?x128xf16>
      }
      %row2 = arith.muli %i, %tm : index
      %ov = memref.subview %out[%row2, 0] [%tm, 128] [1, 1] : memref<256x128xf16> to memref<?x128xf16, strided<[128, 1], offset: ?>>
      %rb = bufferization.to_buffer %r : tensor<?x128xf16> to memref<?x128xf16, strided<[128, 1], offset: ?>>
      memref.copy %rb, %ov : memref<?x128xf16, strided<[128, 1], offset: ?>> to memref<?x128xf16, strided<[128, 1], offset: ?>>
    }
    return
  }
}
