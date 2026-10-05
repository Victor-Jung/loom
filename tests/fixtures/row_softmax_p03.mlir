module attributes {loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 512 : index}} {
  %0 = adl.memory.bank "mem_DRAM_bank", {bsize = 8192 : i64, nblk = 196608 : i64}
  %1 = adl.spatial_dim "dim_dram_channel", 8
  %2 = adl.memory.array "mem_DRAM", [%1] of %0
  %3 = adl.memory.bank "mem_bank", {bsize = 16 : i64, nblk = 5464 : i64}
  %4 = adl.spatial_dim "dim_nbank", 16
  %5 = adl.memory.array "mem_L1", [%4] of %3
  %6 = adl.resource.exclusive "res_matrix_lane"
  %7 = adl.resource.exclusive "res_vector_lane"
  %8 = adl.processor.compute @proc_matrix_lane, from %5 to %5, with [%6]
  %9 = adl.processor.compute @proc_vector_lane, from %5 to %5, with [%7]
  %10 = adl.arch.compose "arch_mesh", arch[%8, %9], mem[%5]
  %11 = adl.spatial_dim "dim_x", 1
  %12 = adl.spatial_dim "dim_y", 1
  %13 = adl.memory.array "mem_array_L1", [%11, %12] of %5
  %14 = adl.arch.scale "arch_mesh", [%11, %12] of %10, mem_region %13
  %15 = adl.resource.exclusive "res_noc0"
  %16 = adl.resource.exclusive "res_noc1"
  %17 = adl.processor.dmover @proc_dram_l1_noc0, from %2 to %13, with [%15]
  %18 = adl.processor.dmover @proc_l1_l1_noc0, from %13 to %13, with [%15]
  %19 = adl.processor.dmover @proc_l1_dram_noc1, from %13 to %2, with [%16]
  %20 = adl.arch.compose "arch_system", arch[%14, %17, %18, %19], mem[%2]
  module attributes {loom.pass_name = "Materialize", loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 512 : index}} {
    func.func @_row_softmax__x1_y1__d0i0_d1i0__f0__n_n__is_double_buffer1__tile_m64(%arg0: memref<512x256xf16>, %arg1: memref<512x256xf16>) {
      %c16384 = arith.constant 16384 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0xFC00 : f16
      %c8 = arith.constant 8 : index
      scf.parallel (%arg2) = (%c0) to (%c1) step (%c1) {
        scf.for %arg3 = %c0 to %c8 step %c1 {
          %21 = loom.alloc [64, 256] on @L1 : memref<64x256xf16>
          %22 = loom.semaphore_take %21 : memref<64x256xf16> -> memref<64x256xf16>
          %23 = arith.muli %arg3, %c16384 : index
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 256], strides: [256, 1] : memref<512x256xf16> to memref<64x256xf16, strided<[256, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg2, %c0], LR : [%arg2, %c0]) : memref<64x256xf16, strided<[256, 1], offset: ?>> to memref<64x256xf16>
          %24 = loom.alloc [64, 1] on @L1 : memref<64x1xf16>
          %25 = loom.semaphore_take %24 : memref<64x1xf16> -> memref<64x1xf16>
          %26 = loom.semaphore_take %24 : memref<64x1xf16> -> memref<64x1xf16>
          linalg.fill ins(%cst : f16) outs(%26 : memref<64x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>], iterator_types = ["parallel", "reduction"]} ins(%22 : memref<64x256xf16>) outs(%26 : memref<64x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %35 = arith.maximumf %in, %out : f16
            linalg.yield %35 : f16
          }
          %27 = loom.alloc [64, 256] on @L1 : memref<64x256xf16>
          %28 = loom.semaphore_take %27 : memref<64x256xf16> -> memref<64x256xf16>
          %29 = loom.broadcast ins(%26 : memref<64x1xf16>) outs(%28 : memref<64x256xf16>) dim(1) -> memref<64x256xf16, strided<[?, ?], offset: ?>>
          loom.semaphore_give %26 : memref<64x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %29 : memref<64x256xf16>, memref<64x256xf16, strided<[?, ?], offset: ?>>) outs(%28 : memref<64x256xf16>) {
          ^bb0(%in: f16, %in_1: f16, %out: f16):
            %35 = arith.subf %in, %in_1 : f16
            %36 = math.exp %35 : f16
            linalg.yield %36 : f16
          }
          loom.semaphore_give %22 : memref<64x256xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>], iterator_types = ["parallel", "reduction"]} ins(%28 : memref<64x256xf16>) outs(%25 : memref<64x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %35 = arith.addf %in, %out : f16
            linalg.yield %35 : f16
          }
          %30 = loom.alloc [64, 256] on @L1 : memref<64x256xf16>
          %31 = loom.semaphore_take %30 : memref<64x256xf16> -> memref<64x256xf16>
          %32 = loom.broadcast ins(%25 : memref<64x1xf16>) outs(%31 : memref<64x256xf16>) dim(1) -> memref<64x256xf16, strided<[?, ?], offset: ?>>
          loom.semaphore_give %25 : memref<64x1xf16>
          %33 = loom.alloc [64, 256] on @L1 : memref<64x256xf16>
          %34 = loom.semaphore_take %33 : memref<64x256xf16> -> memref<64x256xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%28, %32 : memref<64x256xf16>, memref<64x256xf16, strided<[?, ?], offset: ?>>) outs(%34 : memref<64x256xf16>) {
          ^bb0(%in: f16, %in_1: f16, %out: f16):
            %35 = arith.divf %in, %in_1 : f16
            linalg.yield %35 : f16
          }
          loom.semaphore_give %31 : memref<64x256xf16>
          loom.semaphore_give %28 : memref<64x256xf16>
          %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%23], sizes: [64, 256], strides: [256, 1] : memref<512x256xf16> to memref<64x256xf16, strided<[256, 1], offset: ?>>
          loom.copy %34, %reinterpret_cast_0 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg2, %c0], LR : [%arg2, %c0]) : memref<64x256xf16> to memref<64x256xf16, strided<[256, 1], offset: ?>>
          loom.semaphore_give %34 : memref<64x256xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m], loom.iter_types = [#loom.iter_type<spatial>], loom.logical_levels = [0], loom.physical_dims = [@dim_x]}
      return
    }
  }
}
