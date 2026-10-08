#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {loom.tile_h = {asure_divisible = false, is_reduction = false, upper_bound = 1 : index}, loom.tile_t = {asure_divisible = false, is_reduction = false, upper_bound = 4096 : index}} {
  func.func @_rotary(%x_arg: memref<32x4096x128xf16>, %cos_arg: memref<4096x128xf16>, %sin_arg: memref<4096x128xf16>, %rot_arg: memref<128x128xf16>, %out__arg: memref<32x4096x128xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    %c32 = arith.constant 32 : index
    %0 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_t, upper_bound = 4096 : index} : () -> index
    %1 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_h, upper_bound = 1 : index} : () -> index
    %2 = arith.ceildivui %c4096, %0 : index
    affine.parallel (%arg5) = (0) to (symbol(%2)) {
      %3 = arith.ceildivui %c32, %1 : index
      scf.for %arg6 = %c0 to %3 step %c1 {
        %4 = arith.muli %arg6, %1 : index
        %5 = arith.muli %arg5, %0 : index
        %subview = memref.subview %x_arg[%4, %5, 0] [1, %0, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x128xf16, strided<[128, 1], offset: ?>>
        %6 = bufferization.to_tensor %subview restrict : memref<?x128xf16, strided<[128, 1], offset: ?>> to tensor<?x128xf16>
        %subview_0 = memref.subview %cos_arg[%5, 0] [%0, 128] [1, 1] : memref<4096x128xf16> to memref<?x128xf16, strided<[128, 1], offset: ?>>
        %7 = bufferization.to_tensor %subview_0 restrict : memref<?x128xf16, strided<[128, 1], offset: ?>> to tensor<?x128xf16>
        %subview_1 = memref.subview %sin_arg[%5, 0] [%0, 128] [1, 1] : memref<4096x128xf16> to memref<?x128xf16, strided<[128, 1], offset: ?>>
        %8 = bufferization.to_tensor %subview_1 restrict : memref<?x128xf16, strided<[128, 1], offset: ?>> to tensor<?x128xf16>
        %9 = bufferization.to_tensor %rot_arg restrict : memref<128x128xf16> to tensor<128x128xf16>
        %10 = tensor.empty(%0) : tensor<?x128xf16>
        %11 = linalg.fill ins(%cst : f16) outs(%10 : tensor<?x128xf16>) -> tensor<?x128xf16>
        %12 = linalg.matmul ins(%6, %9 : tensor<?x128xf16>, tensor<128x128xf16>) outs(%11 : tensor<?x128xf16>) -> tensor<?x128xf16>
        %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %7 : tensor<?x128xf16>, tensor<?x128xf16>) outs(%10 : tensor<?x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %17 = arith.mulf %in, %in_3 : f16
          linalg.yield %17 : f16
        } -> tensor<?x128xf16>
        %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%12, %8 : tensor<?x128xf16>, tensor<?x128xf16>) outs(%10 : tensor<?x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %17 = arith.mulf %in, %in_3 : f16
          linalg.yield %17 : f16
        } -> tensor<?x128xf16>
        %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%13, %14 : tensor<?x128xf16>, tensor<?x128xf16>) outs(%10 : tensor<?x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %17 = arith.addf %in, %in_3 : f16
          linalg.yield %17 : f16
        } -> tensor<?x128xf16>
        %subview_2 = memref.subview %out__arg[%4, %5, 0] [1, %0, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x128xf16, strided<[128, 1], offset: ?>>
        %16 = bufferization.to_buffer %15 : tensor<?x128xf16> to memref<?x128xf16, strided<[128, 1], offset: ?>>
        memref.copy %16, %subview_2 : memref<?x128xf16, strided<[128, 1], offset: ?>> to memref<?x128xf16, strided<[128, 1], offset: ?>>
      }
    }
    return
  }
}

