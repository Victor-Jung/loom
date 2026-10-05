#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module attributes {loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 512 : index}} {
  func.func @_attention_fullrow(%q_arg: memref<512x64xf16>, %k_arg: memref<256x64xf16>, %v_arg: memref<256x64xf16>, %out__arg: memref<512x64xf16>) {
    %cst = arith.constant 0xFC00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c512 = arith.constant 512 : index
    %0 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_m, upper_bound = 512 : index} : () -> index
    %1 = arith.ceildivui %c512, %0 : index
    affine.parallel (%arg4) = (0) to (symbol(%1)) {
      %2 = arith.muli %arg4, %0 : index
      %subview = memref.subview %q_arg[%2, 0] [%0, 64] [1, 1] : memref<512x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      %3 = bufferization.to_tensor %subview restrict : memref<?x64xf16, strided<[64, 1], offset: ?>> to tensor<?x64xf16>
      %4 = bufferization.to_tensor %k_arg restrict : memref<256x64xf16> to tensor<256x64xf16>
      %5 = tensor.empty() : tensor<64x256xf16>
      %transposed = linalg.transpose ins(%4 : tensor<256x64xf16>) outs(%5 : tensor<64x256xf16>) permutation = [1, 0] 
      %6 = tensor.empty(%0) : tensor<?x256xf16>
      %7 = linalg.fill ins(%cst_0 : f16) outs(%6 : tensor<?x256xf16>) -> tensor<?x256xf16>
      %8 = linalg.matmul ins(%3, %transposed : tensor<?x64xf16>, tensor<64x256xf16>) outs(%7 : tensor<?x256xf16>) -> tensor<?x256xf16>
      %9 = tensor.empty(%0) : tensor<?x1xf16>
      %10 = linalg.fill ins(%cst : f16) outs(%9 : tensor<?x1xf16>) -> tensor<?x1xf16>
      %11 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%8 : tensor<?x256xf16>) outs(%10 : tensor<?x1xf16>) {
      ^bb0(%in: f16, %out: f16):
        %24 = arith.maximumf %in, %out : f16
        linalg.yield %24 : f16
      } -> tensor<?x1xf16>
      %12 = "loom.broadcast"(%11, %6) {dim = 1 : i64} : (tensor<?x1xf16>, tensor<?x256xf16>) -> tensor<?x256xf16>
      %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %12 : tensor<?x256xf16>, tensor<?x256xf16>) outs(%6 : tensor<?x256xf16>) {
      ^bb0(%in: f16, %in_2: f16, %out: f16):
        %24 = arith.subf %in, %in_2 : f16
        linalg.yield %24 : f16
      } -> tensor<?x256xf16>
      %14 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<?x256xf16>) outs(%6 : tensor<?x256xf16>) {
      ^bb0(%in: f16, %out: f16):
        %24 = math.exp %in : f16
        linalg.yield %24 : f16
      } -> tensor<?x256xf16>
      %15 = linalg.fill ins(%cst_0 : f16) outs(%9 : tensor<?x1xf16>) -> tensor<?x1xf16>
      %16 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%14 : tensor<?x256xf16>) outs(%15 : tensor<?x1xf16>) {
      ^bb0(%in: f16, %out: f16):
        %24 = arith.addf %in, %out : f16
        linalg.yield %24 : f16
      } -> tensor<?x1xf16>
      %17 = "loom.broadcast"(%16, %6) {dim = 1 : i64} : (tensor<?x1xf16>, tensor<?x256xf16>) -> tensor<?x256xf16>
      %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %17 : tensor<?x256xf16>, tensor<?x256xf16>) outs(%6 : tensor<?x256xf16>) {
      ^bb0(%in: f16, %in_2: f16, %out: f16):
        %24 = arith.divf %in, %in_2 : f16
        linalg.yield %24 : f16
      } -> tensor<?x256xf16>
      %19 = bufferization.to_tensor %v_arg restrict : memref<256x64xf16> to tensor<256x64xf16>
      %20 = tensor.empty(%0) : tensor<?x64xf16>
      %21 = linalg.fill ins(%cst_0 : f16) outs(%20 : tensor<?x64xf16>) -> tensor<?x64xf16>
      %22 = linalg.matmul ins(%18, %19 : tensor<?x256xf16>, tensor<256x64xf16>) outs(%21 : tensor<?x64xf16>) -> tensor<?x64xf16>
      %subview_1 = memref.subview %out__arg[%2, 0] [%0, 64] [1, 1] : memref<512x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      %23 = bufferization.to_buffer %22 : tensor<?x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      memref.copy %23, %subview_1 : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1], offset: ?>>
    }
    return
  }
}

