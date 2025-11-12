module {
  aie.device(npu2_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %out2_buff_0 = aie.buffer(%tile_0_4) {sym_name = "out2_buff_0"} : memref<8xi32> 
    %out2_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 1 : i32, sym_name = "out2_prod_lock_0"}
    %out2_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "out2_cons_lock_0"}
    %out1_buff_0 = aie.buffer(%tile_0_3) {sym_name = "out1_buff_0"} : memref<8xi32> 
    %out1_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 1 : i32, sym_name = "out1_prod_lock_0"}
    %out1_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "out1_cons_lock_0"}
    %out0_buff_0 = aie.buffer(%tile_0_2) {sym_name = "out0_buff_0"} : memref<8xi32> 
    %out0_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "out0_prod_lock_0"}
    %out0_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "out0_cons_lock_0"}
    %out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "out_cons_prod_lock_0"}
    %out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "out_cons_cons_lock_0"}
    %out_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "out_buff_0"} : memref<24xi32> 
    %out_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 1 : i32, sym_name = "out_prod_lock_0"}
    %out_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "out_cons_lock_0"}
    %out_prod_lock_1 = aie.lock(%mem_tile_0_1, 2) {init = 1 : i32, sym_name = "out_prod_lock_1"}
    %out_cons_lock_1 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "out_cons_lock_1"}
    %out_prod_lock_2 = aie.lock(%mem_tile_0_1, 4) {init = 1 : i32, sym_name = "out_prod_lock_2"}
    %out_cons_lock_2 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "out_cons_lock_2"}
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_0_1, DMA : 2)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        aie.use_lock(%out0_prod_lock_0, AcquireGreaterEqual, 1)
        %c0_0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c8 step %c1_1 {
          %c1_i32 = arith.constant 1 : i32
          memref.store %c1_i32, %out0_buff_0[%arg1] : memref<8xi32>
        }
        aie.use_lock(%out0_cons_lock_0, Release, 1)
      }
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        aie.use_lock(%out1_prod_lock_0, AcquireGreaterEqual, 1)
        %c0_0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c8 step %c1_1 {
          %c2_i32 = arith.constant 2 : i32
          memref.store %c2_i32, %out1_buff_0[%arg1] : memref<8xi32>
        }
        aie.use_lock(%out1_cons_lock_0, Release, 1)
      }
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        aie.use_lock(%out2_prod_lock_0, AcquireGreaterEqual, 1)
        %c0_0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c8 step %c1_1 {
          %c3_i32 = arith.constant 3 : i32
          memref.store %c3_i32, %out2_buff_0[%arg1] : memref<8xi32>
        }
        aie.use_lock(%out2_cons_lock_0, Release, 1)
      }
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<48xi32>, %arg1: memref<48xi32>, %arg2: memref<48xi32>) {
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 48][0, 0, 0, 1]) {id = 0 : i64, metadata = @out_shim_alloc} : memref<48xi32>
      aiex.npu.dma_wait {symbol = @out_shim_alloc}
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb6)
    // ^bb1:  // 2 preds: ^bb0, ^bb3
    //   aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, 1)
    //   aie.dma_bd(%out_buff_0 : memref<24xi32>, 0, 8)
    //   aie.use_lock(%out_prod_lock_0, Release, 1)
    //   aie.next_bd ^bb2
    // ^bb2:  // pred: ^bb1
    //   aie.use_lock(%out_cons_lock_1, AcquireGreaterEqual, 1)
    //   aie.dma_bd(%out_buff_0 : memref<24xi32>, 8, 8)
    //   aie.use_lock(%out_prod_lock_1, Release, 1)
    //   aie.next_bd ^bb3
    // ^bb3:  // pred: ^bb2
    //   aie.use_lock(%out_cons_lock_2, AcquireGreaterEqual, 1)
    //   aie.dma_bd(%out_buff_0 : memref<24xi32>, 16, 8)
    //   aie.use_lock(%out_prod_lock_2, Release, 1)
    //   aie.next_bd ^bb1

    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%out_cons_lock_2, AcquireGreaterEqual, 1)
      aie.use_lock(%out_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<24xi32>, 0, 24, [<size = 8, stride = 1>, <size = 3, stride = 8>])
      aie.use_lock(%out_prod_lock_0, Release, 1)
      aie.use_lock(%out_prod_lock_2, Release, 1)
      aie.use_lock(%out_prod_lock_1, Release, 1)
      aie.next_bd ^bb1

    // ^bb1:  // 2 preds: ^bb0, ^bb3
    //   aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, 1)
    //   aie.dma_bd(%out_buff_0 : memref<24xi32>, 0, 0, [<size = 8, stride = 1>, <size = 3, stride = 8>])
    //   aie.next_bd ^bb2
    // ^bb2:  // pred: ^bb1
    //   aie.use_lock(%out_cons_lock_1, AcquireGreaterEqual, 1)
    //   aie.dma_bd(%out_buff_0 : memref<24xi32>, 0, 0, [<size = 8, stride = 1>, <size = 3, stride = 8>])
    //   aie.next_bd ^bb3
    // ^bb3:  // pred: ^bb2
    //   aie.use_lock(%out_cons_lock_2, AcquireGreaterEqual, 1)
    //   aie.dma_bd(%out_buff_0 : memref<24xi32>, 0, 24, [<size = 8, stride = 1>, <size = 3, stride = 8>])
    //   aie.use_lock(%out_prod_lock_0, Release, 1)
    //   aie.next_bd ^bb4
    // ^bb4:  // pred: ^bb3
    //   aie.dma_bd(%out_buff_0 : memref<24xi32>, 0, 0, [<size = 8, stride = 1>, <size = 3, stride = 8>])
    //   aie.use_lock(%out_prod_lock_1, Release, 1)
    //   aie.next_bd ^bb5
    // ^bb5:  // pred: ^bb4
    //   aie.dma_bd(%out_buff_0 : memref<24xi32>, 0, 0, [<size = 8, stride = 1>, <size = 3, stride = 8>])
    //   aie.use_lock(%out_prod_lock_2, Release, 1)
    //   aie.next_bd ^bb1

    ^bb6:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%out_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<24xi32>, 0, 8)
      aie.use_lock(%out_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %2 = aie.dma_start(S2MM, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%out_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<24xi32>, 8, 8)
      aie.use_lock(%out_cons_lock_1, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %3 = aie.dma_start(S2MM, 2, ^bb11, ^bb12)
    ^bb11:  // 2 preds: ^bb10, ^bb11
      aie.use_lock(%out_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<24xi32>, 16, 8)
      aie.use_lock(%out_cons_lock_2, Release, 1)
      aie.next_bd ^bb11
    ^bb12:  // pred: ^bb10
      aie.end
    }
    aie.shim_dma_allocation @out_shim_alloc(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%out0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out0_buff_0 : memref<8xi32>, 0, 8)
      aie.use_lock(%out0_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%out1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_buff_0 : memref<8xi32>, 0, 8)
      aie.use_lock(%out1_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%out2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_buff_0 : memref<8xi32>, 0, 8)
      aie.use_lock(%out2_prod_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
  }
}
