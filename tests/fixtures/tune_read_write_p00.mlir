module {
  func.func @_read_write(%a: memref<4x64x64xf16>, %b: memref<64x64xf16>, %c: memref<64x64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %bv = memref.subview %b[0, 0] [64, 64] [1, 1] : memref<64x64xf16> to memref<64x64xf16, strided<[64, 1]>>
      %bt = bufferization.to_tensor %bv restrict : memref<64x64xf16, strided<[64, 1]>> to tensor<64x64xf16>
      %cv = memref.subview %c[0, 0] [64, 64] [1, 1] : memref<64x64xf16> to memref<64x64xf16, strided<[64, 1]>>
      %ct = bufferization.to_tensor %cv restrict : memref<64x64xf16, strided<[64, 1]>> to tensor<64x64xf16>
      %av = memref.subview %a[%i, 0, 0] [1, 64, 64] [1, 1, 1] : memref<4x64x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
      %bb = bufferization.to_buffer %ct : tensor<64x64xf16> to memref<64x64xf16, strided<[64, 1]>>
      memref.copy %bb, %av : memref<64x64xf16, strided<[64, 1]>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
      %bw = bufferization.to_buffer %bt : tensor<64x64xf16> to memref<64x64xf16, strided<[64, 1]>>
      memref.copy %bw, %bv : memref<64x64xf16, strided<[64, 1]>> to memref<64x64xf16, strided<[64, 1]>>
    }
    return
  }
}
