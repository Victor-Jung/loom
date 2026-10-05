module attributes {loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {asure_divisible = false, is_reduction = false, upper_bound = 512 : index}, loom.tile_n_1 = {asure_divisible = false, is_reduction = false, upper_bound = 512 : index}, loom.tile_n_2 = {asure_divisible = false, is_reduction = false, upper_bound = 512 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {asure_divisible = false, is_reduction = false, upper_bound = 512 : index}, loom.tile_n_1 = {asure_divisible = false, is_reduction = false, upper_bound = 512 : index}, loom.tile_n_2 = {asure_divisible = false, is_reduction = false, upper_bound = 512 : index}} {
    func.func @_softmax_twopass__x1_y1__d0i0_d1i0__f0__n_n_n_n__is_double_buffer1__tile_m64__tile_n128__tile_n_132__tile_n_2128(%arg0: memref<256x512xf16>, %arg1: memref<256x512xf16>) {
      %c32768 = arith.constant 32768 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg2) = (%c0) to (%c1) step (%c1) {
        scf.for %arg3 = %c0 to %c4 step %c1 {
          %21 = loom.alloc [64, 1] on @L1 : memref<64x1xf16>
          %22 = loom.semaphore_take %21 : memref<64x1xf16> -> memref<64x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%22 : memref<64x1xf16>)
          %23 = loom.alloc [64, 1] on @L1 : memref<64x1xf16>
          %24 = loom.semaphore_take %23 : memref<64x1xf16> -> memref<64x1xf16>
          %25 = loom.semaphore_take %23 : memref<64x1xf16> -> memref<64x1xf16>
          %26 = loom.alloc [64, 128] on @L1 : memref<64x128xf16>
          %27 = loom.semaphore_take %26 : memref<64x128xf16> -> memref<64x128xf16>
          scf.for %arg4 = %c0 to %c4 step %c1 {
            %32 = arith.muli %arg4, %c128 : index
            %33 = arith.muli %arg3, %c32768 : index
            %34 = arith.addi %33, %32 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [64, 128], strides: [512, 1] : memref<256x512xf16> to memref<64x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg2, %c0], LR : [%arg2, %c0]) : memref<64x128xf16, strided<[512, 1], offset: ?>> to memref<64x128xf16>
            linalg.fill ins(%cst_0 : f16) outs(%24 : memref<64x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>], iterator_types = ["parallel", "reduction"]} ins(%27 : memref<64x128xf16>) outs(%24 : memref<64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %35 = arith.maximumf %in, %out : f16
              linalg.yield %35 : f16
            }
            loom.semaphore_give %27 : memref<64x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %24 : memref<64x1xf16>, memref<64x1xf16>) outs(%22 : memref<64x1xf16>) {
            ^bb0(%in: f16, %in_1: f16, %out: f16):
              %35 = arith.cmpf ogt, %in, %in_1 : f16
              %36 = arith.select %35, %in, %in_1 : f16
              linalg.yield %36 : f16
            }
            loom.semaphore_give %24 : memref<64x1xf16>
          }
          linalg.fill ins(%cst : f16) outs(%25 : memref<64x1xf16>)
          %28 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %29 = loom.semaphore_take %28 : memref<64x32xf16> -> memref<64x32xf16>
          %30 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %31 = loom.semaphore_take %30 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg4 = %c0 to %c16 step %c1 {
            %32 = arith.muli %arg4, %c32 : index
            %33 = arith.muli %arg3, %c32768 : index
            %34 = arith.addi %33, %32 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [64, 32], strides: [512, 1] : memref<256x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg2, %c0], LR : [%arg2, %c0]) : memref<64x32xf16, strided<[512, 1], offset: ?>> to memref<64x32xf16>
            %35 = loom.broadcast ins(%22 : memref<64x1xf16>) outs(%31 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%29, %35 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%31 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_1: f16, %out: f16):
              %36 = arith.subf %in, %in_1 : f16
              %37 = math.exp %36 : f16
              linalg.yield %37 : f16
            }
            loom.semaphore_give %29 : memref<64x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>], iterator_types = ["parallel", "reduction"]} ins(%31 : memref<64x32xf16>) outs(%25 : memref<64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %36 = arith.addf %in, %out : f16
              linalg.yield %36 : f16
            }
            loom.semaphore_give %31 : memref<64x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          scf.for %arg4 = %c0 to %c4 step %c1 {
            %32 = arith.muli %arg4, %c128 : index
            %33 = loom.alloc [64, 128] on @L1 : memref<64x128xf16>
            %34 = loom.semaphore_take %33 : memref<64x128xf16> -> memref<64x128xf16>
            %35 = arith.muli %arg3, %c32768 : index
            %36 = arith.addi %35, %32 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%36], sizes: [64, 128], strides: [512, 1] : memref<256x512xf16> to memref<64x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg2, %c0], LR : [%arg2, %c0]) : memref<64x128xf16, strided<[512, 1], offset: ?>> to memref<64x128xf16>
            %37 = loom.alloc [64, 128] on @L1 : memref<64x128xf16>
            %38 = loom.semaphore_take %37 : memref<64x128xf16> -> memref<64x128xf16>
            %39 = loom.broadcast ins(%25 : memref<64x1xf16>) outs(%38 : memref<64x128xf16>) dim(1) -> memref<64x128xf16, strided<[?, ?], offset: ?>>
            %40 = loom.alloc [64, 128] on @L1 : memref<64x128xf16>
            %41 = loom.semaphore_take %40 : memref<64x128xf16> -> memref<64x128xf16>
            %42 = loom.broadcast ins(%22 : memref<64x1xf16>) outs(%41 : memref<64x128xf16>) dim(1) -> memref<64x128xf16, strided<[?, ?], offset: ?>>
            %43 = loom.alloc [64, 128] on @L1 : memref<64x128xf16>
            %44 = loom.semaphore_take %43 : memref<64x128xf16> -> memref<64x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%34, %42, %39 : memref<64x128xf16>, memref<64x128xf16, strided<[?, ?], offset: ?>>, memref<64x128xf16, strided<[?, ?], offset: ?>>) outs(%44 : memref<64x128xf16>) {
            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
              %45 = arith.subf %in, %in_2 : f16
              %46 = math.exp %45 : f16
              %47 = arith.divf %46, %in_3 : f16
              linalg.yield %47 : f16
            }
            loom.semaphore_give %41 : memref<64x128xf16>
            loom.semaphore_give %38 : memref<64x128xf16>
            loom.semaphore_give %34 : memref<64x128xf16>
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%36], sizes: [64, 128], strides: [512, 1] : memref<256x512xf16> to memref<64x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %44, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg2, %c0], LR : [%arg2, %c0]) : memref<64x128xf16> to memref<64x128xf16, strided<[512, 1], offset: ?>>
            loom.semaphore_give %44 : memref<64x128xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %22 : memref<64x1xf16>
          loom.semaphore_give %25 : memref<64x1xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m], loom.iter_types = [#loom.iter_type<spatial>], loom.logical_levels = [0], loom.physical_dims = [@dim_x]}
      return
    }
  }
}
