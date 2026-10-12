#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module attributes {loom.tile_b = {asure_divisible = false, is_reduction = false, upper_bound = 1 : index}, loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 32 : index}, loom.tile_n = {asure_divisible = false, is_reduction = false, upper_bound = 4096 : index}} {
  func.func @_gqa_decode(%k_view_arg: memref<64x128x4096xf16>, %v_view_arg: memref<64x4096x128xf16>, %q_view_arg: memref<64x32x128xf16>, %out__arg: memref<64x32x128xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %cst_1 = arith.constant 0xFC00 : f16
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 8.837890e-02 : f16
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %0 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_b, upper_bound = 1 : index} : () -> index
    %1 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_m, upper_bound = 32 : index} : () -> index
    %2 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_n, upper_bound = 4096 : index} : () -> index
    %3 = arith.ceildivui %c64, %0 : index
    %4 = arith.ceildivui %c32, %1 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0, %1) : tensor<?x?x1xf16>
      %6 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %7 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %8 = arith.muli %arg4, %0 : index
      %9 = arith.muli %arg5, %1 : index
      %subview = memref.subview %q_view_arg[%8, %9, 0] [%0, %1, 128] [1, 1, 1] : memref<64x32x128xf16> to memref<?x?x128xf16, strided<[4096, 128, 1], offset: ?>>
      %10 = bufferization.to_tensor %subview restrict : memref<?x?x128xf16, strided<[4096, 128, 1], offset: ?>> to tensor<?x?x128xf16>
      %11 = tensor.empty(%0, %1) : tensor<?x?x128xf16>
      %12 = linalg.fill ins(%cst : f16) outs(%11 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %13 = arith.ceildivui %c4096, %2 : index
      %14:3 = scf.for %arg6 = %c0 to %13 step %c1 iter_args(%arg7 = %6, %arg8 = %12, %arg9 = %7) -> (tensor<?x?x1xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) {
        %18 = arith.muli %arg6, %2 : index
        %subview_4 = memref.subview %k_view_arg[%8, 0, %18] [%0, 128, %2] [1, 1, 1] : memref<64x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
        %19 = bufferization.to_tensor %subview_4 restrict : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to tensor<?x128x?xf16, {local_mem_kind = 1 : i64}>
        %20 = arith.index_cast %0 : index to i64
        %21 = arith.cmpi eq, %20, %20 : i64
        cf.assert %21, "mismatching contracting dimension"
        %22 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf16>
        %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %24 = linalg.batch_matmul ins(%10, %19 : tensor<?x?x128xf16>, tensor<?x128x?xf16, {local_mem_kind = 1 : i64}>) outs(%23 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %25 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24 : tensor<?x?x?xf16>) outs(%22 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %43 = arith.mulf %in, %cst_2 : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x?xf16>
        %26 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%25 : tensor<?x?x?xf16>) outs(%6 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %43 = arith.maximumf %in, %out : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x1xf16>
        %27 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %26 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %43 = arith.cmpf ogt, %in, %in_6 : f16
          %44 = arith.select %43, %in, %in_6 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x1xf16>
        %28 = "loom.broadcast"(%27, %22) {dim = 2 : i64} : (tensor<?x?x1xf16>, tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %29 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %28 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%22 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %43 = arith.subf %in, %in_6 : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x?xf16>
        %30 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29 : tensor<?x?x?xf16>) outs(%22 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %43 = math.exp %in : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x?xf16>
        %31 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %27 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %43 = arith.subf %in, %in_6 : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x1xf16>
        %32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31 : tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %43 = math.exp %in : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x1xf16>
        %subview_5 = memref.subview %v_view_arg[%8, %18, 0] [%0, %2, 128] [1, 1, 1] : memref<64x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        %33 = bufferization.to_tensor %subview_5 restrict : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        %34 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %32 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%11 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %43 = arith.mulf %in, %in_6 : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x128xf16>
        cf.assert %21, "mismatching contracting dimension"
        %35 = arith.index_cast %2 : index to i64
        %36 = arith.cmpi eq, %35, %35 : i64
        cf.assert %36, "mismatching contracting dimension"
        %37 = linalg.batch_matmul ins(%30, %33 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%12 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %38 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %37 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%11 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %43 = arith.addf %in, %in_6 : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x128xf16>
        %39 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
        %40 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%30 : tensor<?x?x?xf16>) outs(%39 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %43 = arith.addf %in, %out : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x1xf16>
        %41 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %32 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %43 = arith.mulf %in, %in_6 : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x1xf16>
        %42 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %40 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %43 = arith.addf %in, %in_6 : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x1xf16>
        scf.yield %27, %38, %42 : tensor<?x?x1xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>
      }
      %15 = "loom.broadcast"(%14#2, %11) {dim = 2 : i64} : (tensor<?x?x1xf16>, tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14#1, %15 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%11 : tensor<?x?x128xf16>) {
      ^bb0(%in: f16, %in_4: f16, %out: f16):
        %18 = arith.divf %in, %in_4 : f16
        linalg.yield %18 : f16
      } -> tensor<?x?x128xf16>
      %subview_3 = memref.subview %out__arg[%8, %9, 0] [%0, %1, 128] [1, 1, 1] : memref<64x32x128xf16> to memref<?x?x128xf16, strided<[4096, 128, 1], offset: ?>>
      %17 = bufferization.to_buffer %16 : tensor<?x?x128xf16> to memref<?x?x128xf16, strided<[4096, 128, 1], offset: ?>>
      memref.copy %17, %subview_3 : memref<?x?x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x?x128xf16, strided<[4096, 128, 1], offset: ?>>
    }
    return
  }
}

