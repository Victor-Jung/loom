module attributes {loom.tile_k = {asure_divisible = false, is_reduction = false, upper_bound = 64 : index}, loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {asure_divisible = false, is_reduction = false, upper_bound = 256 : index}} {
  func.func @_chain_fused(%x_arg: memref<256x64xf16>, %y_arg: memref<64x256xf16>, %z_arg: memref<256x64xf16>, %out__arg: memref<256x64xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %0 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_n, upper_bound = 256 : index} : () -> index
    %2 = "loom.sym"() {asure_divisible = false, is_reduction = false, symbol_ref = @tile_k, upper_bound = 64 : index} : () -> index
    %3 = arith.ceildivui %c256, %0 : index
    affine.parallel (%arg4) = (0) to (symbol(%3)) {
      %4 = tensor.empty(%0) : tensor<?x64xf16>
      %5 = linalg.fill ins(%cst : f16) outs(%4 : tensor<?x64xf16>) -> tensor<?x64xf16>
      %6 = arith.ceildivui %c256, %1 : index
      %7 = scf.for %arg5 = %c0 to %6 step %c1 iter_args(%arg6 = %5) -> (tensor<?x64xf16>) {
        %10 = tensor.empty(%0, %1) : tensor<?x?xf16>
        %11 = linalg.fill ins(%cst : f16) outs(%10 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %12 = arith.ceildivui %c64, %2 : index
        %13 = scf.for %arg7 = %c0 to %12 step %c1 iter_args(%arg8 = %11) -> (tensor<?x?xf16>) {
          %17 = arith.muli %arg4, %0 : index
          %18 = arith.muli %arg7, %2 : index
          %subview_1 = memref.subview %x_arg[%17, %18] [%0, %2] [1, 1] : memref<256x64xf16> to memref<?x?xf16, strided<[64, 1], offset: ?>>
          %19 = bufferization.to_tensor %subview_1 restrict : memref<?x?xf16, strided<[64, 1], offset: ?>> to tensor<?x?xf16>
          %20 = arith.muli %arg5, %1 : index
          %subview_2 = memref.subview %y_arg[%18, %20] [%2, %1] [1, 1] : memref<64x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
          %21 = bufferization.to_tensor %subview_2 restrict : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
          %22 = linalg.matmul ins(%19, %21 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
          scf.yield %22 : tensor<?x?xf16>
        }
        %14 = arith.muli %arg5, %1 : index
        %subview_0 = memref.subview %z_arg[%14, 0] [%1, 64] [1, 1] : memref<256x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
        %15 = bufferization.to_tensor %subview_0 restrict : memref<?x64xf16, strided<[64, 1], offset: ?>> to tensor<?x64xf16>
        %16 = linalg.matmul ins(%13, %15 : tensor<?x?xf16>, tensor<?x64xf16>) outs(%arg6 : tensor<?x64xf16>) -> tensor<?x64xf16>
        scf.yield %16 : tensor<?x64xf16>
      }
      %8 = arith.muli %arg4, %0 : index
      %subview = memref.subview %out__arg[%8, 0] [%0, 64] [1, 1] : memref<256x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      %9 = bufferization.to_buffer %7 : tensor<?x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      memref.copy %9, %subview : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1], offset: ?>>
    }
    return
  }
}

