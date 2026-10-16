#map = affine_map<(d0, d1) -> (d0, d1)>
#grp = affine_map<(d0) -> (d0 floordiv 4)>
module attributes {loom.tile_h = {asure_divisible = false, is_reduction = false, upper_bound = 1 : index}, loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 256 : index}} {
  func.func @_grouped(%x_arg: memref<16x256x512xf16>, %w_arg: memref<4x256x512xf16>, %out__arg: memref<16x256x512xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %0 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_h, upper_bound = 1 : index} : () -> index
    %1 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %2 = arith.ceildivui %c256, %1 : index
    affine.parallel (%arg3) = (0) to (symbol(%2)) {
      %3 = arith.ceildivui %c16, %0 : index
      scf.for %arg4 = %c0 to %3 step %c1 {
        %4 = arith.muli %arg4, %0 : index
        %5 = arith.muli %arg3, %1 : index
        %subview = memref.subview %x_arg[%4, %5, 0] [1, %1, 512] [1, 1, 1] : memref<16x256x512xf16> to memref<?x512xf16, strided<[512, 1], offset: ?>>
        %6 = bufferization.to_tensor %subview restrict : memref<?x512xf16, strided<[512, 1], offset: ?>> to tensor<?x512xf16>
        %7 = affine.apply #grp(%4)
        %subview_0 = memref.subview %w_arg[%7, %5, 0] [1, %1, 512] [1, 1, 1] : memref<4x256x512xf16> to memref<?x512xf16, strided<[512, 1], offset: ?>>
        %8 = bufferization.to_tensor %subview_0 restrict : memref<?x512xf16, strided<[512, 1], offset: ?>> to tensor<?x512xf16>
        %9 = arith.divui %4, %c4 : index
        %10 = arith.remui %4, %c4 : index
        %11 = arith.index_cast %9 : index to i32
        %12 = arith.sitofp %11 : i32 to f16
        %13 = arith.index_cast %10 : index to i32
        %14 = arith.sitofp %13 : i32 to f16
        %15 = tensor.empty(%1) : tensor<?x512xf16>
        %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %8 : tensor<?x512xf16>, tensor<?x512xf16>) outs(%15 : tensor<?x512xf16>) {
        ^bb0(%in: f16, %in_1: f16, %out: f16):
          %20 = arith.addf %in, %in_1 : f16
          %21 = arith.mulf %20, %12 : f16
          %22 = arith.addf %21, %14 : f16
          linalg.yield %22 : f16
        } -> tensor<?x512xf16>
        %subview_2 = memref.subview %out__arg[%4, %5, 0] [1, %1, 512] [1, 1, 1] : memref<16x256x512xf16> to memref<?x512xf16, strided<[512, 1], offset: ?>>
        %17 = bufferization.to_buffer %16 : tensor<?x512xf16> to memref<?x512xf16, strided<[512, 1], offset: ?>>
        memref.copy %17, %subview_2 : memref<?x512xf16, strided<[512, 1], offset: ?>> to memref<?x512xf16, strided<[512, 1], offset: ?>>
      }
    }
    return
  }
}
