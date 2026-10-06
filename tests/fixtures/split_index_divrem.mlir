// Cyclic split index h = axis + wave * 8 with axis in [0, 8): h / 8 is the
// wave and h % 8 the axis. Dividing by 4 is not foldable (axis / 4 remains).
module {
  func.func @f(%out: memref<32xindex>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %one = loom.sym @tile_h {upper_bound = 1 : index} : index
    affine.parallel (%axis) = (0) to (8) {
      scf.for %wave = %c0 to %c4 step %c1 {
        %h0 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%axis, %wave)
        %h = arith.muli %h0, %one : index
        %g = arith.divui %h, %c8 : index
        %r = arith.remui %h, %c8 : index
        %q4 = arith.divui %h, %c4 : index
        memref.store %g, %out[%h] : memref<32xindex>
        memref.store %r, %out[%g] : memref<32xindex>
        memref.store %q4, %out[%r] : memref<32xindex>
      }
    }
    return
  }
}
