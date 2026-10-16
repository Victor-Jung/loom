#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {loom.tile_b = {asure_divisible = false, is_reduction = false, upper_bound = 1 : index}, loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 1024 : index}, loom.tile_n = {asure_divisible = false, is_reduction = false, upper_bound = 2048 : index}} {
  func.func @_batch_sum(%x_arg: memref<32x1024x2048xf16>, %out__arg: memref<1024x2048xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    %c32 = arith.constant 32 : index
    %0 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_m, upper_bound = 1024 : index} : () -> index
    %1 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_n, upper_bound = 2048 : index} : () -> index
    %2 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_b, upper_bound = 1 : index} : () -> index
    %3 = arith.ceildivui %c1024, %0 : index
    affine.parallel (%arg2) = (0) to (symbol(%3)) {
      %4 = arith.ceildivui %c2048, %1 : index
      scf.for %arg3 = %c0 to %4 step %c1 {
        %5 = tensor.empty(%0, %1) : tensor<?x?xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %7 = arith.ceildivui %c32, %2 : index
        %8 = scf.for %arg4 = %c0 to %7 step %c1 iter_args(%arg5 = %6) -> (tensor<?x?xf16>) {
          %12 = arith.muli %arg4, %2 : index
          %13 = arith.muli %arg2, %0 : index
          %14 = arith.muli %arg3, %1 : index
          %subview_0 = memref.subview %x_arg[%12, %13, %14] [1, %0, %1] [1, 1, 1] : memref<32x1024x2048xf16> to memref<?x?xf16, strided<[2048, 1], offset: ?>>
          %15 = bufferization.to_tensor %subview_0 restrict : memref<?x?xf16, strided<[2048, 1], offset: ?>> to tensor<?x?xf16>
          %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg5, %15 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_1: f16, %out: f16):
            %17 = arith.addf %in, %in_1 : f16
            linalg.yield %17 : f16
          } -> tensor<?x?xf16>
          scf.yield %16 : tensor<?x?xf16>
        }
        %9 = arith.muli %arg2, %0 : index
        %10 = arith.muli %arg3, %1 : index
        %subview = memref.subview %out__arg[%9, %10] [%0, %1] [1, 1] : memref<1024x2048xf16> to memref<?x?xf16, strided<[2048, 1], offset: ?>>
        %11 = bufferization.to_buffer %8 : tensor<?x?xf16> to memref<?x?xf16, strided<[2048, 1], offset: ?>>
        memref.copy %11, %subview : memref<?x?xf16, strided<[2048, 1], offset: ?>> to memref<?x?xf16, strided<[2048, 1], offset: ?>>
      }
    }
    return
  }
}

