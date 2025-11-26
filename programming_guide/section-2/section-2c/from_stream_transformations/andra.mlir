module {
  aie.device(npu2_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %out2_buff_0 = aie.buffer(%tile_0_4) {sym_name = "out2_buff_0"} : memref<8xi32> 
    %out2_prod_lock_0 = aie.lock(%tile_0_4, 2) {init = 1 : i32, sym_name = "out2_prod_lock_0"}
    %out2_cons_lock_0 = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "out2_cons_lock_0"}
    %out1_buff_0 = aie.buffer(%tile_0_3) {sym_name = "out1_buff_0"} : memref<8xi32> 
    %out1_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 1 : i32, sym_name = "out1_prod_lock_0"}
    %out1_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "out1_cons_lock_0"}
    %out0_buff_0 = aie.buffer(%tile_0_2) {sym_name = "out0_buff_0"} : memref<8xi32> 
    %out0_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "out0_prod_lock_0"}
    %out0_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out0_cons_lock_0"}
    %out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "out_cons_prod_lock_0"}
    %out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "out_cons_cons_lock_0"}
    %out_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "out_buff_0"} : memref<24xi32> 
    %out_prod_lock_0 = aie.lock(%mem_tile_0_1, 6) {init = 1 : i32, sym_name = "out_prod_lock_0"}
    %out_cons_lock_0 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "out_cons_lock_0"}
    %out_prod_lock_1 = aie.lock(%mem_tile_0_1, 8) {init = 1 : i32, sym_name = "out_prod_lock_1"}
    %out_cons_lock_1 = aie.lock(%mem_tile_0_1, 9) {init = 0 : i32, sym_name = "out_cons_lock_1"}
    %out_prod_lock_2 = aie.lock(%mem_tile_0_1, 10) {init = 1 : i32, sym_name = "out_prod_lock_2"}
    %out_cons_lock_2 = aie.lock(%mem_tile_0_1, 11) {init = 0 : i32, sym_name = "out_cons_lock_2"}
    %in2_cons_buff_0 = aie.buffer(%tile_0_4) {sym_name = "in2_cons_buff_0"} : memref<8xi32> 
    %in2_cons_prod_lock_0 = aie.lock(%tile_0_4, 0) {init = 1 : i32, sym_name = "in2_cons_prod_lock_0"}
    %in2_cons_cons_lock_0 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "in2_cons_cons_lock_0"}
    %in1_cons_buff_0 = aie.buffer(%tile_0_3) {sym_name = "in1_cons_buff_0"} : memref<8xi32> 
    %in1_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 1 : i32, sym_name = "in1_cons_prod_lock_0"}
    %in1_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "in1_cons_cons_lock_0"}
    %in0_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "in0_cons_buff_0"} : memref<8xi32> 
    %in0_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "in0_cons_prod_lock_0"}
    %in0_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in0_cons_cons_lock_0"}
    %in_cons_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "in_cons_buff_0"} : memref<3x8xi32> 
    %in_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 1 : i32, sym_name = "in_cons_prod_lock_0"}
    %in_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "in_cons_cons_lock_0"}
    %in_cons_prod_lock_1 = aie.lock(%mem_tile_0_1, 2) {init = 1 : i32, sym_name = "in_cons_prod_lock_1"}
    %in_cons_cons_lock_1 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "in_cons_cons_lock_1"}
    %in_cons_prod_lock_2 = aie.lock(%mem_tile_0_1, 4) {init = 1 : i32, sym_name = "in_cons_prod_lock_2"}
    %in_cons_cons_lock_2 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "in_cons_cons_lock_2"}
    %in_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "in_prod_lock_0"}
    %in_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "in_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 3, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_0_1, DMA : 3)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          aie.use_lock(%in0_cons_cons_lock_0, AcquireGreaterEqual, 1)
          aie.use_lock(%out0_prod_lock_0, AcquireGreaterEqual, 1)
          %c0_2 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c8 step %c1_3 {
            %0 = memref.load %in0_cons_buff_0[%arg2] : memref<8xi32>
            memref.store %0, %out0_buff_0[%arg2] : memref<8xi32>
          }
          aie.use_lock(%in0_cons_prod_lock_0, Release, 1)
          aie.use_lock(%out0_cons_lock_0, Release, 1)
        }
      }
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          aie.use_lock(%in1_cons_cons_lock_0, AcquireGreaterEqual, 1)
          aie.use_lock(%out1_prod_lock_0, AcquireGreaterEqual, 1)
          %c0_2 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c8 step %c1_3 {
            %0 = memref.load %in1_cons_buff_0[%arg2] : memref<8xi32>
            memref.store %0, %out1_buff_0[%arg2] : memref<8xi32>
          }
          aie.use_lock(%in1_cons_prod_lock_0, Release, 1)
          aie.use_lock(%out1_cons_lock_0, Release, 1)
        }
      }
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c2 step %c1_1 {
          aie.use_lock(%in2_cons_cons_lock_0, AcquireGreaterEqual, 1)
          aie.use_lock(%out2_prod_lock_0, AcquireGreaterEqual, 1)
          %c0_2 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c8 step %c1_3 {
            %0 = memref.load %in2_cons_buff_0[%arg2] : memref<8xi32>
            memref.store %0, %out2_buff_0[%arg2] : memref<8xi32>
          }
          aie.use_lock(%in2_cons_prod_lock_0, Release, 1)
          aie.use_lock(%out2_cons_lock_0, Release, 1)
        }
      }
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<48xi32>, %arg1: memref<48xi32>, %arg2: memref<48xi32>) {
      %0 = aiex.dma_configure_task_for @in_shim_alloc {
        aie.dma_bd(%arg0 : memref<48xi32>, 0, 48, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out_shim_alloc {
        aie.dma_bd(%arg2 : memref<48xi32>, 0, 48, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 48, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
    aie.shim_dma_allocation @in_shim_alloc(MM2S, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      aie.use_lock(%in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_0 : memref<3x8xi32>, 0, 24, [<size = 3, stride = 1>, <size = 8, stride = 3>])
      aie.use_lock(%in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in_cons_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_0 : memref<3x8xi32>, 0, 0, [<size = 3, stride = 1>, <size = 8, stride = 3>])
      aie.use_lock(%in_cons_cons_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%in_cons_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_0 : memref<3x8xi32>, 0, 0, [<size = 3, stride = 1>, <size = 8, stride = 3>])
      aie.use_lock(%in_cons_cons_lock_2, Release, 1)
      aie.next_bd ^bb1
    ^bb4:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_0 : memref<3x8xi32>, 0, 8)
      aie.use_lock(%in_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%in_cons_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_0 : memref<3x8xi32>, 8, 8)
      aie.use_lock(%in_cons_prod_lock_1, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 2, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb9
      aie.use_lock(%in_cons_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_0 : memref<3x8xi32>, 16, 8)
      aie.use_lock(%in_cons_prod_lock_2, Release, 1)
      aie.next_bd ^bb9
    ^bb10:  // pred: ^bb8
      %4 = aie.dma_start(MM2S, 3, ^bb11, ^bb14)
    ^bb11:  // 2 preds: ^bb10, ^bb13
      aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<24xi32>, 0, 8)
      aie.use_lock(%out_prod_lock_0, Release, 1)
      aie.next_bd ^bb12
    ^bb12:  // pred: ^bb11
      aie.use_lock(%out_cons_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<24xi32>, 8, 8)
      aie.use_lock(%out_prod_lock_1, Release, 1)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      aie.use_lock(%out_cons_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<24xi32>, 16, 8)
      aie.use_lock(%out_prod_lock_2, Release, 1)
      aie.next_bd ^bb11
    ^bb14:  // pred: ^bb10
      %5 = aie.dma_start(S2MM, 1, ^bb15, ^bb16)
    ^bb15:  // 2 preds: ^bb14, ^bb15
      aie.use_lock(%out_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<24xi32>, 0, 8)
      aie.use_lock(%out_cons_lock_0, Release, 1)
      aie.next_bd ^bb15
    ^bb16:  // pred: ^bb14
      %6 = aie.dma_start(S2MM, 2, ^bb17, ^bb18)
    ^bb17:  // 2 preds: ^bb16, ^bb17
      aie.use_lock(%out_prod_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<24xi32>, 8, 8)
      aie.use_lock(%out_cons_lock_1, Release, 1)
      aie.next_bd ^bb17
    ^bb18:  // pred: ^bb16
      %7 = aie.dma_start(S2MM, 3, ^bb19, ^bb20)
    ^bb19:  // 2 preds: ^bb18, ^bb19
      aie.use_lock(%out_prod_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<24xi32>, 16, 8)
      aie.use_lock(%out_cons_lock_2, Release, 1)
      aie.next_bd ^bb19
    ^bb20:  // pred: ^bb18
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%in0_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in0_cons_buff_0 : memref<8xi32>, 0, 8)
      aie.use_lock(%in0_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%out0_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out0_buff_0 : memref<8xi32>, 0, 8)
      aie.use_lock(%out0_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%in1_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_cons_buff_0 : memref<8xi32>, 0, 8)
      aie.use_lock(%in1_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%out1_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_buff_0 : memref<8xi32>, 0, 8)
      aie.use_lock(%out1_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%in2_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_cons_buff_0 : memref<8xi32>, 0, 8)
      aie.use_lock(%in2_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%out2_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_buff_0 : memref<8xi32>, 0, 8)
      aie.use_lock(%out2_prod_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    aie.shim_dma_allocation @out_shim_alloc(S2MM, 0, 0)
  }
}

