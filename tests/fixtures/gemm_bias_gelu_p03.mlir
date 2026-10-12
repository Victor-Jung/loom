module attributes {loom.tile_k = {asure_divisible = false, is_reduction = false, upper_bound = 1024 : index}, loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {asure_divisible = false, is_reduction = false, upper_bound = 4096 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {asure_divisible = false, is_reduction = false, upper_bound = 1024 : index}, loom.tile_m = {asure_divisible = false, is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {asure_divisible = false, is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_gemm_bias_gelu__order_m6Sn2Sn8_m_n_k__x6x2_y8__d0i0_d1i2_d2i1__f012__dim_y_level0_bc8_dim_x_level0_bc6_dim_x_level0_bc6_n__is_double_buffer1__tile_k512__tile_m256__tile_n128(%arg0: memref<4096x1024xf16>, %arg1: memref<1024x4096xf16>, %arg2: memref<1x4096xf16>, %arg3: memref<4096x4096xf16>) attributes {loom.tune.estimate = 31864 : i64, loom.tune.order = "m6Sn2Sn8_m_n_k", loom.tune.terms = "dram=3.61882e+07;core=1.77024e+06;blocks=18;mblocks=15;work=1.3824e+06;cores=96"} {
      %c4096 = arith.constant 4096 : index
      %c1024 = arith.constant 1024 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c3968 = arith.constant 3968 : index
      %c3840 = arith.constant 3840 : index
      %c5 = arith.constant 5 : index
      %c7 = arith.constant 7 : index
      %c16 = arith.constant 16 : index
      %c6 = arith.constant 6 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 1.000000e+00 : f16
      %cst_0 = arith.constant -1.702150e+00 : f16
      %cst_1 = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c512 = arith.constant 512 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c6, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c3 step %c1 {
          %21 = arith.muli %arg7, %c6 overflow<nsw> : index
          %22 = arith.addi %arg4, %21 : index
          %23 = arith.cmpi ult, %22, %c16 : index
          scf.for %arg8 = %c0 to %c2 step %c1 {
            %24 = arith.muli %arg5, %c8 overflow<nsw> : index
            %25 = arith.addi %arg6, %24 : index
            %26 = arith.muli %arg8, %c16 overflow<nsw> : index
            %27 = arith.addi %25, %26 : index
            %28 = loom.alloc [256, 128] on @L1 : memref<256x128xf16>
            %29 = loom.semaphore_take %28 : memref<256x128xf16> -> memref<256x128xf16>
            linalg.fill ins(%cst_1 : f16) outs(%29 : memref<256x128xf16>)
            %30 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
            %31 = loom.semaphore_take %30 : memref<256x512xf16> -> memref<256x512xf16>
            %32 = loom.alloc [512, 128] on @L1 : memref<512x128xf16>
            %33 = loom.semaphore_take %32 : memref<512x128xf16> -> memref<512x128xf16>
            scf.for %arg9 = %c0 to %c2 step %c1 {
              %52 = arith.muli %22, %c256 : index
              %53 = arith.muli %arg9, %c512 : index
              %54 = arith.cmpi ugt, %52, %c3840 : index
              %55 = arith.select %54, %c3840, %52 : index
              %56 = arith.cmpi ugt, %53, %c512 : index
              %57 = arith.select %56, %c512, %53 : index
              %58 = arith.muli %55, %c1024 overflow<nsw> : index
              %59 = arith.addi %58, %57 : index
              %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%59], sizes: [256, 512], strides: [1024, 1] : memref<4096x1024xf16> to memref<256x512xf16, strided<[1024, 1], offset: ?>>
              %60 = arith.muli %arg5, %c6 : index
              %61 = arith.addi %arg4, %60 : index
              loom.copy %reinterpret_cast_3, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%61, %c0], LR : [%61, %c7]) : memref<256x512xf16, strided<[1024, 1], offset: ?>> to memref<256x512xf16>
              %62 = arith.muli %27, %c128 : index
              %63 = arith.cmpi ugt, %62, %c3968 : index
              %64 = arith.select %63, %c3968, %62 : index
              %65 = arith.muli %57, %c4096 overflow<nsw> : index
              %66 = arith.addi %65, %64 : index
              %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%66], sizes: [512, 128], strides: [4096, 1] : memref<1024x4096xf16> to memref<512x128xf16, strided<[4096, 1], offset: ?>>
              %67 = arith.addi %60, %c5 : index
              loom.copy %reinterpret_cast_4, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [6, 1] region : (UL : [%60, %arg6], LR : [%67, %arg6]) : memref<512x128xf16, strided<[4096, 1], offset: ?>> to memref<512x128xf16>
              linalg.matmul ins(%31, %33 : memref<256x512xf16>, memref<512x128xf16>) outs(%29 : memref<256x128xf16>)
              loom.semaphore_give %33 : memref<512x128xf16>
              loom.semaphore_give %31 : memref<256x512xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            %34 = arith.muli %27, %c128 : index
            %35 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %36 = loom.semaphore_take %35 : memref<1x128xf16> -> memref<1x128xf16>
            %37 = arith.cmpi ugt, %34, %c3968 : index
            %38 = arith.select %37, %c3968, %34 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%38], sizes: [1, 128], strides: [4096, 1] : memref<1x4096xf16> to memref<1x128xf16, strided<[4096, 1], offset: ?>>
            %39 = arith.muli %arg5, %c6 : index
            %40 = arith.addi %39, %c5 : index
            loom.copy %reinterpret_cast, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [6, 1] region : (UL : [%39, %arg6], LR : [%40, %arg6]) : memref<1x128xf16, strided<[4096, 1], offset: ?>> to memref<1x128xf16>
            %41 = loom.alloc [256, 128] on @L1 : memref<256x128xf16>
            %42 = loom.semaphore_take %41 : memref<256x128xf16> -> memref<256x128xf16>
            %43 = loom.broadcast ins(%36 : memref<1x128xf16>) outs(%42 : memref<256x128xf16>) dim(0) -> memref<256x128xf16, strided<[?, ?], offset: ?>>
            loom.semaphore_give %36 : memref<1x128xf16>
            %44 = loom.alloc [256, 128] on @L1 : memref<256x128xf16>
            %45 = loom.semaphore_take %44 : memref<256x128xf16> -> memref<256x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%29, %43 : memref<256x128xf16>, memref<256x128xf16, strided<[?, ?], offset: ?>>) outs(%45 : memref<256x128xf16>) {
            ^bb0(%in: f16, %in_3: f16, %out: f16):
              %52 = arith.addf %in, %in_3 : f16
              %53 = arith.mulf %52, %cst_0 : f16
              %54 = math.exp %53 : f16
              %55 = arith.addf %54, %cst : f16
              %56 = arith.divf %52, %55 : f16
              linalg.yield %56 : f16
            }
            loom.semaphore_give %42 : memref<256x128xf16>
            loom.semaphore_give %29 : memref<256x128xf16>
            %46 = arith.muli %22, %c256 : index
            %47 = arith.cmpi ugt, %46, %c3840 : index
            %48 = arith.select %47, %c3840, %46 : index
            %49 = arith.muli %48, %c4096 overflow<nsw> : index
            %50 = arith.addi %49, %38 : index
            %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%50], sizes: [256, 128], strides: [4096, 1] : memref<4096x4096xf16> to memref<256x128xf16, strided<[4096, 1], offset: ?>>
            %51 = arith.addi %arg4, %39 : index
            loom.copy %45, %reinterpret_cast_2 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%51, %arg6], LR : [%51, %arg6]) if %23 : memref<256x128xf16> to memref<256x128xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %45 : memref<256x128xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>, loom.partial_wave = "shadow"}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_n, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 1, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
}
