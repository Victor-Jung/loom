#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {loom.tile_b = {asure_divisible = false, is_reduction = false, upper_bound = 1 : index}, loom.tile_c = {asure_divisible = false, is_reduction = false, upper_bound = 1 : index}, loom.tile_h = {asure_divisible = false, is_reduction = false, upper_bound = 1 : index}, loom.tile_k = {asure_divisible = false, is_reduction = false, upper_bound = 192 : index}, loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 192 : index}, loom.tile_n = {asure_divisible = false, is_reduction = false, upper_bound = 128 : index}} {
  func.func @_mamba_chunk_scan(%cb_arg: memref<2x10x4x192x192xf16>, %x_arg: memref<2x32x1920x128xf16>, %dt_k_arg: memref<2x32x10x1x192xf16>, %dA_cumsum_m_arg: memref<2x32x10x192x1xf16>, %dA_cumsum_k_arg: memref<2x32x10x1x192xf16>, %C_arg: memref<2x4x1920x128xf16>, %D_arg: memref<2x32x192x128xf16>, %prev_states_T_arg: memref<2x10x32x128x128xf16>, %out__arg: memref<2x32x1920x128xf16>) {
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c192 = arith.constant 192 : index
    %0 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_m, upper_bound = 192 : index} : () -> index
    %1 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_n, upper_bound = 128 : index} : () -> index
    %2 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_k, upper_bound = 192 : index} : () -> index
    %3 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_c, upper_bound = 1 : index} : () -> index
    %4 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_b, upper_bound = 1 : index} : () -> index
    %5 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_h, upper_bound = 1 : index} : () -> index
    %6 = arith.ceildivui %c192, %0 : index
    %7 = arith.ceildivui %c128, %1 : index
    %8 = arith.ceildivui %c10, %3 : index
    affine.parallel (%arg9, %arg10, %arg11) = (0, 0, 0) to (symbol(%6), symbol(%7), symbol(%8)) {
      %9 = arith.ceildivui %c2, %4 : index
      scf.for %arg12 = %c0 to %9 step %c1 {
        %10 = arith.ceildivui %c32, %5 : index
        scf.for %arg13 = %c0 to %10 step %c1 {
          %11 = tensor.empty(%0, %1) : tensor<?x?xf16>
          %12 = linalg.fill ins(%cst : f16) outs(%11 : tensor<?x?xf16>) -> tensor<?x?xf16>
          %13 = arith.muli %arg12, %4 : index
          %14 = arith.muli %arg13, %5 : index
          %15 = arith.muli %arg11, %3 : index
          %16 = arith.muli %arg9, %0 : index
          %subview = memref.subview %dA_cumsum_m_arg[%13, %14, %15, %16, 0] [1, 1, 1, %0, 1] [1, 1, 1, 1, 1] : memref<2x32x10x192x1xf16> to memref<?x1xf16, strided<[1, 1], offset: ?>>
          %17 = bufferization.to_tensor %subview restrict : memref<?x1xf16, strided<[1, 1], offset: ?>> to tensor<?x1xf16>
          %18 = "loom.broadcast"(%17, %11) {dim = 1 : i64} : (tensor<?x1xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
          %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%18 : tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %35 = math.exp %in : f16
            linalg.yield %35 : f16
          } -> tensor<?x?xf16>
          %20 = arith.divui %14, %c8 : index
          %21 = arith.muli %15, %c192 : index
          %22 = arith.addi %16, %21 : index
          %subview_0 = memref.subview %C_arg[%13, %20, %22, 0] [1, 1, %0, 128] [1, 1, 1, 1] : memref<2x4x1920x128xf16> to memref<?x128xf16, strided<[128, 1], offset: ?>>
          %23 = bufferization.to_tensor %subview_0 restrict : memref<?x128xf16, strided<[128, 1], offset: ?>> to tensor<?x128xf16>
          %24 = arith.muli %arg10, %1 : index
          %subview_1 = memref.subview %prev_states_T_arg[%13, %15, %14, 0, %24] [1, 1, 1, 128, %1] [1, 1, 1, 1, 1] : memref<2x10x32x128x128xf16> to memref<128x?xf16, strided<[128, 1], offset: ?>>
          %25 = bufferization.to_tensor %subview_1 restrict : memref<128x?xf16, strided<[128, 1], offset: ?>> to tensor<128x?xf16>
          %26 = linalg.matmul ins(%23, %25 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%12 : tensor<?x?xf16>) -> tensor<?x?xf16>
          %27 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%26, %19 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %35 = arith.mulf %in, %in_5 : f16
            linalg.yield %35 : f16
          } -> tensor<?x?xf16>
          %subview_2 = memref.subview %D_arg[%13, %14, %16, %24] [1, 1, %0, %1] [1, 1, 1, 1] : memref<2x32x192x128xf16> to memref<?x?xf16, strided<[128, 1], offset: ?>>
          %28 = bufferization.to_tensor %subview_2 restrict : memref<?x?xf16, strided<[128, 1], offset: ?>> to tensor<?x?xf16>
          %subview_3 = memref.subview %x_arg[%13, %14, %22, %24] [1, 1, %0, %1] [1, 1, 1, 1] : memref<2x32x1920x128xf16> to memref<?x?xf16, strided<[128, 1], offset: ?>>
          %29 = bufferization.to_tensor %subview_3 restrict : memref<?x?xf16, strided<[128, 1], offset: ?>> to tensor<?x?xf16>
          %30 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%29, %28 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %35 = arith.mulf %in, %in_5 : f16
            linalg.yield %35 : f16
          } -> tensor<?x?xf16>
          %31 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%27, %30 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %35 = arith.addf %in, %in_5 : f16
            linalg.yield %35 : f16
          } -> tensor<?x?xf16>
          %32 = arith.ceildivui %c192, %2 : index
          %33 = scf.for %arg14 = %c0 to %32 step %c1 iter_args(%arg15 = %31) -> (tensor<?x?xf16>) {
            %35 = arith.muli %arg14, %2 : index
            %subview_5 = memref.subview %cb_arg[%13, %15, %20, %16, %35] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<2x10x4x192x192xf16> to memref<?x?xf16, strided<[192, 1], offset: ?>>
            %36 = bufferization.to_tensor %subview_5 restrict : memref<?x?xf16, strided<[192, 1], offset: ?>> to tensor<?x?xf16>
            %subview_6 = memref.subview %dA_cumsum_k_arg[%13, %14, %15, 0, %35] [1, 1, 1, 1, %2] [1, 1, 1, 1, 1] : memref<2x32x10x1x192xf16> to memref<1x?xf16, strided<[192, 1], offset: ?>>
            %37 = bufferization.to_tensor %subview_6 restrict : memref<1x?xf16, strided<[192, 1], offset: ?>> to tensor<1x?xf16>
            %38 = tensor.empty(%0, %2) : tensor<?x?xf16>
            %39 = "loom.broadcast"(%17, %38) {dim = 1 : i64} : (tensor<?x1xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
            %40 = "loom.broadcast"(%37, %38) {dim = 0 : i64} : (tensor<1x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
            %41 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%39, %40 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%38 : tensor<?x?xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %51 = arith.subf %in, %in_9 : f16
              linalg.yield %51 : f16
            } -> tensor<?x?xf16>
            %42 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%41 : tensor<?x?xf16>) outs(%38 : tensor<?x?xf16>) {
            ^bb0(%in: f16, %out: f16):
              %51 = math.exp %in : f16
              linalg.yield %51 : f16
            } -> tensor<?x?xf16>
            %43 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%36, %42 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%38 : tensor<?x?xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %51 = arith.mulf %in, %in_9 : f16
              linalg.yield %51 : f16
            } -> tensor<?x?xf16>
            %subview_7 = memref.subview %dt_k_arg[%13, %14, %15, 0, %35] [1, 1, 1, 1, %2] [1, 1, 1, 1, 1] : memref<2x32x10x1x192xf16> to memref<1x?xf16, strided<[192, 1], offset: ?>>
            %44 = bufferization.to_tensor %subview_7 restrict : memref<1x?xf16, strided<[192, 1], offset: ?>> to tensor<1x?xf16>
            %45 = "loom.broadcast"(%44, %38) {dim = 0 : i64} : (tensor<1x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
            %46 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%43, %45 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%38 : tensor<?x?xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %51 = arith.mulf %in, %in_9 : f16
              linalg.yield %51 : f16
            } -> tensor<?x?xf16>
            %47 = arith.addi %35, %21 : index
            %subview_8 = memref.subview %x_arg[%13, %14, %47, %24] [1, 1, %2, %1] [1, 1, 1, 1] : memref<2x32x1920x128xf16> to memref<?x?xf16, strided<[128, 1], offset: ?>>
            %48 = bufferization.to_tensor %subview_8 restrict : memref<?x?xf16, strided<[128, 1], offset: ?>> to tensor<?x?xf16>
            %49 = linalg.matmul ins(%46, %48 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%12 : tensor<?x?xf16>) -> tensor<?x?xf16>
            %50 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg15, %49 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %51 = arith.addf %in, %in_9 : f16
              linalg.yield %51 : f16
            } -> tensor<?x?xf16>
            scf.yield %50 : tensor<?x?xf16>
          }
          %subview_4 = memref.subview %out__arg[%13, %14, %22, %24] [1, 1, %0, %1] [1, 1, 1, 1] : memref<2x32x1920x128xf16> to memref<?x?xf16, strided<[128, 1], offset: ?>>
          %34 = bufferization.to_buffer %33 : tensor<?x?xf16> to memref<?x?xf16, strided<[128, 1], offset: ?>>
          memref.copy %34, %subview_4 : memref<?x?xf16, strided<[128, 1], offset: ?>> to memref<?x?xf16, strided<[128, 1], offset: ?>>
        }
      }
    }
    return
  }
}

