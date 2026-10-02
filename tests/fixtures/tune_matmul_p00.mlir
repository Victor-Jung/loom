module attributes {loom.tile_k = {asure_divisible = false, is_reduction = false, upper_bound = 16384 : index}, loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {asure_divisible = false, is_reduction = false, upper_bound = 2048 : index}} {
  func.func @_matmul(%x_arg: memref<2048x16384xf16>, %y_arg: memref<16384x2048xf16>, %out__arg: memref<2048x2048xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c2048 = arith.constant 2048 : index
    %c16384 = arith.constant 16384 : index
    %0 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_m, upper_bound = 2048 : index} : () -> index
    %1 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_n, upper_bound = 2048 : index} : () -> index
    %2 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_k, upper_bound = 16384 : index} : () -> index
    %3 = arith.ceildivui %c2048, %0 : index
    %4 = arith.ceildivui %c2048, %1 : index
    affine.parallel (%arg3, %arg4) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %7 = arith.ceildivui %c16384, %2 : index
      %8 = scf.for %arg5 = %c0 to %7 step %c1 iter_args(%arg6 = %6) -> (tensor<?x?xf16>) {
        %12 = arith.muli %arg3, %0 : index
        %13 = arith.muli %arg5, %2 : index
        %subview_0 = memref.subview %x_arg[%12, %13] [%0, %2] [1, 1] : memref<2048x16384xf16> to memref<?x?xf16, strided<[16384, 1], offset: ?>>
        %14 = bufferization.to_tensor %subview_0 : memref<?x?xf16, strided<[16384, 1], offset: ?>> to tensor<?x?xf16>
        %15 = arith.muli %arg4, %1 : index
        %subview_1 = memref.subview %y_arg[%13, %15] [%2, %1] [1, 1] : memref<16384x2048xf16> to memref<?x?xf16, strided<[2048, 1], offset: ?>>
        %16 = bufferization.to_tensor %subview_1 : memref<?x?xf16, strided<[2048, 1], offset: ?>> to tensor<?x?xf16>
        %17 = linalg.matmul ins(%14, %16 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg6 : tensor<?x?xf16>) -> tensor<?x?xf16>
        scf.yield %17 : tensor<?x?xf16>
      }
      %9 = arith.muli %arg3, %0 : index
      %10 = arith.muli %arg4, %1 : index
      %subview = memref.subview %out__arg[%9, %10] [%0, %1] [1, 1] : memref<2048x2048xf16> to memref<?x?xf16, strided<[2048, 1], offset: ?>>
      %11 = bufferization.to_buffer %8 : tensor<?x?xf16> to memref<?x?xf16, strided<[2048, 1], offset: ?>>
      memref.copy %11, %subview : memref<?x?xf16, strided<[2048, 1], offset: ?>> to memref<?x?xf16, strided<[2048, 1], offset: ?>>
    }
    return
  }
}

