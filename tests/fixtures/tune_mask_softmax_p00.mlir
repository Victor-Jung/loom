#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module attributes {loom.tile_b = {asure_divisible = false, is_reduction = false, upper_bound = 1 : index}, loom.tile_h = {asure_divisible = false, is_reduction = false, upper_bound = 1 : index}, loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 1024 : index}} {
  func.func @_mask_softmax(%x_arg: memref<2x32x1024x1024xf16>, %mask_arg: memref<1024x1024xf16>, %out__arg: memref<2x32x1024x1024xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 0xFC00 : f16
    %cst_1 = arith.constant 3.125000e-02 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %0 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_m, upper_bound = 1024 : index} : () -> index
    %1 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_b, upper_bound = 1 : index} : () -> index
    %2 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_h, upper_bound = 1 : index} : () -> index
    %3 = arith.ceildivui %c1024, %0 : index
    affine.parallel (%arg3) = (0) to (symbol(%3)) {
      %4 = arith.ceildivui %c2, %1 : index
      scf.for %arg4 = %c0 to %4 step %c1 {
        %5 = arith.ceildivui %c32, %2 : index
        scf.for %arg5 = %c0 to %5 step %c1 {
          %6 = arith.muli %arg4, %1 : index
          %7 = arith.muli %arg5, %2 : index
          %8 = arith.muli %arg3, %0 : index
          %subview = memref.subview %x_arg[%6, %7, %8, 0] [1, 1, %0, 1024] [1, 1, 1, 1] : memref<2x32x1024x1024xf16> to memref<?x1024xf16, strided<[1024, 1], offset: ?>>
          %9 = bufferization.to_tensor %subview restrict : memref<?x1024xf16, strided<[1024, 1], offset: ?>> to tensor<?x1024xf16>
          %subview_2 = memref.subview %mask_arg[%8, 0] [%0, 1024] [1, 1] : memref<1024x1024xf16> to memref<?x1024xf16, strided<[1024, 1], offset: ?>>
          %10 = bufferization.to_tensor %subview_2 restrict : memref<?x1024xf16, strided<[1024, 1], offset: ?>> to tensor<?x1024xf16>
          %11 = tensor.empty(%0) : tensor<?x1024xf16>
          %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<?x1024xf16>) outs(%11 : tensor<?x1024xf16>) {
          ^bb0(%in: f16, %out: f16):
            %25 = arith.mulf %in, %cst_1 : f16
            linalg.yield %25 : f16
          } -> tensor<?x1024xf16>
          %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%12, %10 : tensor<?x1024xf16>, tensor<?x1024xf16>) outs(%11 : tensor<?x1024xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %25 = arith.addf %in, %in_4 : f16
            linalg.yield %25 : f16
          } -> tensor<?x1024xf16>
          %14 = tensor.empty(%0) : tensor<?x1xf16>
          %15 = linalg.fill ins(%cst_0 : f16) outs(%14 : tensor<?x1xf16>) -> tensor<?x1xf16>
          %16 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%13 : tensor<?x1024xf16>) outs(%15 : tensor<?x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %25 = arith.maximumf %in, %out : f16
            linalg.yield %25 : f16
          } -> tensor<?x1xf16>
          %17 = "loom.broadcast"(%16, %11) {dim = 1 : i64} : (tensor<?x1xf16>, tensor<?x1024xf16>) -> tensor<?x1024xf16>
          %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%13, %17 : tensor<?x1024xf16>, tensor<?x1024xf16>) outs(%11 : tensor<?x1024xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %25 = arith.subf %in, %in_4 : f16
            linalg.yield %25 : f16
          } -> tensor<?x1024xf16>
          %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%18 : tensor<?x1024xf16>) outs(%11 : tensor<?x1024xf16>) {
          ^bb0(%in: f16, %out: f16):
            %25 = math.exp %in : f16
            linalg.yield %25 : f16
          } -> tensor<?x1024xf16>
          %20 = linalg.fill ins(%cst : f16) outs(%14 : tensor<?x1xf16>) -> tensor<?x1xf16>
          %21 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%19 : tensor<?x1024xf16>) outs(%20 : tensor<?x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %25 = arith.addf %in, %out : f16
            linalg.yield %25 : f16
          } -> tensor<?x1xf16>
          %22 = "loom.broadcast"(%21, %11) {dim = 1 : i64} : (tensor<?x1xf16>, tensor<?x1024xf16>) -> tensor<?x1024xf16>
          %23 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%19, %22 : tensor<?x1024xf16>, tensor<?x1024xf16>) outs(%11 : tensor<?x1024xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %25 = arith.divf %in, %in_4 : f16
            linalg.yield %25 : f16
          } -> tensor<?x1024xf16>
          %subview_3 = memref.subview %out__arg[%6, %7, %8, 0] [1, 1, %0, 1024] [1, 1, 1, 1] : memref<2x32x1024x1024xf16> to memref<?x1024xf16, strided<[1024, 1], offset: ?>>
          %24 = bufferization.to_buffer %23 : tensor<?x1024xf16> to memref<?x1024xf16, strided<[1024, 1], offset: ?>>
          memref.copy %24, %subview_3 : memref<?x1024xf16, strided<[1024, 1], offset: ?>> to memref<?x1024xf16, strided<[1024, 1], offset: ?>>
        }
      }
    }
    return
  }
}

