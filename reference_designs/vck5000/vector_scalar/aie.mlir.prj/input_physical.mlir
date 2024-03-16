module {
  aie.device(xcvc1902) {
    memref.global "public" @out_cons : memref<16xi32>
    memref.global "public" @out : memref<16xi32>
    memref.global "public" @in_cons : memref<16xi32>
    memref.global "public" @in : memref<16xi32>
    %tile_6_0 = aie.tile(6, 0)
    %switchbox_6_0 = aie.switchbox(%tile_6_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<North : 0, South : 2>
    }
    %tile_6_2 = aie.tile(6, 2)
    %switchbox_6_2 = aie.switchbox(%tile_6_2) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    %out_buff_0 = aie.buffer(%tile_6_2) {address = 1024 : i32, sym_name = "out_buff_0"} : memref<16xi32> 
    %out_buff_1 = aie.buffer(%tile_6_2) {address = 1088 : i32, sym_name = "out_buff_1"} : memref<16xi32> 
    %out_lock_0 = aie.lock(%tile_6_2, 2) {init = 0 : i32, sym_name = "out_lock_0"}
    %out_lock_1 = aie.lock(%tile_6_2, 3) {init = 0 : i32, sym_name = "out_lock_1"}
    %in_cons_buff_0 = aie.buffer(%tile_6_2) {address = 1152 : i32, sym_name = "in_cons_buff_0"} : memref<16xi32> 
    %in_cons_buff_1 = aie.buffer(%tile_6_2) {address = 1216 : i32, sym_name = "in_cons_buff_1"} : memref<16xi32> 
    %in_cons_lock_0 = aie.lock(%tile_6_2, 0) {init = 0 : i32, sym_name = "in_cons_lock_0"}
    %in_cons_lock_1 = aie.lock(%tile_6_2, 1) {init = 0 : i32, sym_name = "in_cons_lock_1"}
    %shim_mux_6_0 = aie.shim_mux(%tile_6_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %tile_6_1 = aie.tile(6, 1)
    %switchbox_6_1 = aie.switchbox(%tile_6_1) {
      aie.connect<South : 0, North : 0>
      aie.connect<North : 0, South : 0>
    }
    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb11
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb12
    ^bb2:  // pred: ^bb1
      %c0_0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb10
      %3 = arith.cmpi slt, %2, %c4 : index
      cf.cond_br %3, ^bb4, ^bb11
    ^bb4:  // pred: ^bb3
      aie.use_lock(%in_cons_lock_0, Acquire, 1)
      aie.use_lock(%out_lock_0, Acquire, 0)
      %c0_2 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1_3 = arith.constant 1 : index
      cf.br ^bb5(%c0_2 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %6 = memref.load %in_cons_buff_0[%4] : memref<16xi32>
      %c3_i32 = arith.constant 3 : i32
      %7 = arith.muli %6, %c3_i32 : i32
      memref.store %7, %out_buff_0[%4] : memref<16xi32>
      %8 = arith.addi %4, %c1_3 : index
      cf.br ^bb5(%8 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%in_cons_lock_0, Release, 0)
      aie.use_lock(%out_lock_0, Release, 1)
      aie.use_lock(%in_cons_lock_1, Acquire, 1)
      aie.use_lock(%out_lock_1, Acquire, 0)
      %c0_4 = arith.constant 0 : index
      %c16_5 = arith.constant 16 : index
      %c1_6 = arith.constant 1 : index
      cf.br ^bb8(%c0_4 : index)
    ^bb8(%9: index):  // 2 preds: ^bb7, ^bb9
      %10 = arith.cmpi slt, %9, %c16_5 : index
      cf.cond_br %10, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      %11 = memref.load %in_cons_buff_1[%9] : memref<16xi32>
      %c3_i32_7 = arith.constant 3 : i32
      %12 = arith.muli %11, %c3_i32_7 : i32
      memref.store %12, %out_buff_1[%9] : memref<16xi32>
      %13 = arith.addi %9, %c1_6 : index
      cf.br ^bb8(%13 : index)
    ^bb10:  // pred: ^bb8
      aie.use_lock(%in_cons_lock_1, Release, 0)
      aie.use_lock(%out_lock_1, Release, 1)
      %14 = arith.addi %2, %c2 : index
      cf.br ^bb3(%14 : index)
    ^bb11:  // pred: ^bb3
      %15 = arith.addi %0, %c1 : index
      cf.br ^bb1(%15 : index)
    ^bb12:  // pred: ^bb1
      aie.end
    }
    aie.shim_dma_allocation @in(MM2S, 0, 6)
    func.func @sequence(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
      aiex.ipu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0]) {id = 0 : i64, metadata = @out} : memref<64xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0]) {id = 1 : i64, metadata = @in} : memref<64xi32>
      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
    aie.shim_dma_allocation @out(S2MM, 0, 6)
    %mem_6_2 = aie.mem(%tile_6_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in_cons_lock_0, Acquire, 0)
      aie.dma_bd(%in_cons_buff_0 : memref<16xi32>, 0, 16) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in_cons_lock_1, Acquire, 0)
      aie.dma_bd(%in_cons_buff_1 : memref<16xi32>, 0, 16) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in_cons_lock_1, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out_lock_0, Acquire, 1)
      aie.dma_bd(%out_buff_0 : memref<16xi32>, 0, 16) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out_lock_0, Release, 0)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out_lock_1, Acquire, 1)
      aie.dma_bd(%out_buff_1 : memref<16xi32>, 0, 16) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out_lock_1, Release, 0)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.wire(%shim_mux_6_0 : North, %switchbox_6_0 : South)
    aie.wire(%tile_6_0 : DMA, %shim_mux_6_0 : DMA)
    aie.wire(%tile_6_1 : Core, %switchbox_6_1 : Core)
    aie.wire(%tile_6_1 : DMA, %switchbox_6_1 : DMA)
    aie.wire(%switchbox_6_0 : North, %switchbox_6_1 : South)
    aie.wire(%tile_6_2 : Core, %switchbox_6_2 : Core)
    aie.wire(%tile_6_2 : DMA, %switchbox_6_2 : DMA)
    aie.wire(%switchbox_6_1 : North, %switchbox_6_2 : South)
  }
}

