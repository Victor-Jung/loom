module attributes {loom.tile_n = {asure_divisible = false, is_reduction = false, upper_bound = 8192 : index}} {
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
  %11 = adl.spatial_dim "dim_x", 12
  %12 = adl.spatial_dim "dim_y", 10
  %13 = adl.memory.array "mem_array_L1", [%11, %12] of %5
  %14 = adl.arch.scale "arch_mesh", [%11, %12] of %10, mem_region %13
  %15 = adl.resource.exclusive "res_noc0"
  %16 = adl.resource.exclusive "res_noc1"
  %17 = adl.processor.dmover @proc_dram_l1_noc0, from %2 to %13, with [%15]
  %18 = adl.processor.dmover @proc_l1_l1_noc0, from %13 to %13, with [%15]
  %19 = adl.processor.dmover @proc_l1_dram_noc1, from %13 to %2, with [%16]
  %20 = adl.arch.compose "arch_system", arch[%14, %17, %18, %19], mem[%2]
  module attributes {loom.pass_name = "Materialize", loom.tile_n = {asure_divisible = false, is_reduction = false, upper_bound = 8192 : index}} {
    func.func @_col_softmax__order_n11Sn8_n__x11_y8__d0i0_d1i1__f01__n_n__is_double_buffer1__tile_n32(%arg0: memref<1024x8192xf16>, %arg1: memref<1024x8192xf16>) attributes {loom.tune.estimate = 3538 : i64, loom.tune.order = "n11Sn8_n", loom.tune.terms = "dram=1.73015e+07;core=196608;blocks=6;mblocks=0;work=491904;cores=88"} {
      %c88 = arith.constant 88 : index
      %c3 = arith.constant 3 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0xFC00 : f16
      %c32 = arith.constant 32 : index
      %c11 = arith.constant 11 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c11, %c8) step (%c1, %c1) {
        scf.for %arg4 = %c0 to %c3 step %c1 {
          %21 = arith.muli %arg3, %c11 overflow<nsw> : index
          %22 = arith.addi %arg2, %21 : index
          %23 = arith.muli %arg4, %c88 overflow<nsw> : index
          %24 = arith.addi %22, %23 : index
          %25 = arith.cmpi ult, %24, %c256 : index
          scf.if %25 {
            %26 = arith.muli %24, %c32 : index
            %27 = loom.alloc [1024, 32] on @L1 : memref<1024x32xf16>
            %28 = loom.semaphore_take %27 : memref<1024x32xf16> -> memref<1024x32xf16>
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1024, 32], strides: [8192, 1] : memref<1024x8192xf16> to memref<1024x32xf16, strided<[8192, 1], offset: ?>>
            loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg2, %arg3], LR : [%arg2, %arg3]) : memref<1024x32xf16, strided<[8192, 1], offset: ?>> to memref<1024x32xf16>
            %29 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
            %30 = loom.semaphore_take %29 : memref<1x32xf16> -> memref<1x32xf16>
            %31 = loom.semaphore_take %29 : memref<1x32xf16> -> memref<1x32xf16>
            linalg.fill ins(%cst : f16) outs(%31 : memref<1x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>], iterator_types = ["reduction", "parallel"]} ins(%28 : memref<1024x32xf16>) outs(%31 : memref<1x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %40 = arith.maximumf %in, %out : f16
              linalg.yield %40 : f16
            }
            %32 = loom.alloc [1024, 32] on @L1 : memref<1024x32xf16>
            %33 = loom.semaphore_take %32 : memref<1024x32xf16> -> memref<1024x32xf16>
            %34 = loom.broadcast ins(%31 : memref<1x32xf16>) outs(%33 : memref<1024x32xf16>) dim(0) -> memref<1024x32xf16, strided<[?, ?], offset: ?>>
            loom.semaphore_give %31 : memref<1x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%28, %34 : memref<1024x32xf16>, memref<1024x32xf16, strided<[?, ?], offset: ?>>) outs(%33 : memref<1024x32xf16>) {
            ^bb0(%in: f16, %in_1: f16, %out: f16):
              %40 = arith.subf %in, %in_1 : f16
              %41 = math.exp %40 : f16
              linalg.yield %41 : f16
            }
            loom.semaphore_give %28 : memref<1024x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d1)>], iterator_types = ["reduction", "parallel"]} ins(%33 : memref<1024x32xf16>) outs(%30 : memref<1x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %40 = arith.addf %in, %out : f16
              linalg.yield %40 : f16
            }
            %35 = loom.alloc [1024, 32] on @L1 : memref<1024x32xf16>
            %36 = loom.semaphore_take %35 : memref<1024x32xf16> -> memref<1024x32xf16>
            %37 = loom.broadcast ins(%30 : memref<1x32xf16>) outs(%36 : memref<1024x32xf16>) dim(0) -> memref<1024x32xf16, strided<[?, ?], offset: ?>>
            loom.semaphore_give %30 : memref<1x32xf16>
            %38 = loom.alloc [1024, 32] on @L1 : memref<1024x32xf16>
            %39 = loom.semaphore_take %38 : memref<1024x32xf16> -> memref<1024x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<1024x32xf16>, memref<1024x32xf16, strided<[?, ?], offset: ?>>) outs(%39 : memref<1024x32xf16>) {
            ^bb0(%in: f16, %in_1: f16, %out: f16):
              %40 = arith.divf %in, %in_1 : f16
              linalg.yield %40 : f16
            }
            loom.semaphore_give %36 : memref<1024x32xf16>
            loom.semaphore_give %33 : memref<1024x32xf16>
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [1024, 32], strides: [8192, 1] : memref<1024x8192xf16> to memref<1024x32xf16, strided<[8192, 1], offset: ?>>
            loom.copy %39, %reinterpret_cast_0 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg2, %arg3], LR : [%arg2, %arg3]) : memref<1024x32xf16> to memref<1024x32xf16, strided<[8192, 1], offset: ?>>
            loom.semaphore_give %39 : memref<1024x32xf16>
          }
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>, loom.partial_wave = "guard"}
        scf.reduce 
      } {loom.block_syms = [@tile_n, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
}
