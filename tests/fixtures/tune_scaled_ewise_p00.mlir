#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {loom.tile_b = {asure_divisible = false, is_reduction = false, upper_bound = 1 : index}, loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {asure_divisible = false, is_reduction = false, upper_bound = 256 : index}} {
  func.func @_scaled_ewise(%x_arg: memref<8x256x256xf16>, %s_arg: memref<256x256xf16>, %out__arg: memref<8x256x256xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %0 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_b, upper_bound = 1 : index} : () -> index
    %2 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_n, upper_bound = 256 : index} : () -> index
    %3 = arith.ceildivui %c256, %0 : index
    affine.parallel (%arg3) = (0) to (symbol(%3)) {
      %4 = arith.ceildivui %c8, %1 : index
      scf.for %arg4 = %c0 to %4 step %c1 {
        %5 = arith.ceildivui %c256, %2 : index
        scf.for %arg5 = %c0 to %5 step %c1 {
          %6 = arith.muli %arg4, %1 : index
          %7 = arith.muli %arg3, %0 : index
          %8 = arith.muli %arg5, %2 : index
          %subview = memref.subview %x_arg[%6, %7, %8] [1, %0, %2] [1, 1, 1] : memref<8x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
          %9 = bufferization.to_tensor %subview restrict : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
          %subview_0 = memref.subview %s_arg[%7, %8] [%0, %2] [1, 1] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
          %10 = bufferization.to_tensor %subview_0 restrict : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
          %11 = tensor.empty(%0, %2) : tensor<?x?xf16>
          %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %10 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_2: f16, %out: f16):
            %14 = arith.mulf %in, %in_2 : f16
            linalg.yield %14 : f16
          } -> tensor<?x?xf16>
          %subview_1 = memref.subview %out__arg[%6, %7, %8] [1, %0, %2] [1, 1, 1] : memref<8x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
          %13 = bufferization.to_buffer %12 : tensor<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
          memref.copy %13, %subview_1 : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        }
      }
    }
    return
  }
}

