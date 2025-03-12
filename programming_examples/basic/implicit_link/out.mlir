module {
  aie.device(npu1_1col) {
    memref.global "public" @out_cons : memref<1024xi32>
    memref.global "public" @out : memref<1024xi32>
    memref.global "public" @out2_cons : memref<512xi32>
    memref.global "public" @out2 : memref<512xi32>
    memref.global "public" @out1_cons : memref<512xi32>
    memref.global "public" @out1 : memref<512xi32>
    memref.global "public" @in_from_memTile_1 : memref<1024xi32>
    memref.global "public" @in_from_memTile_0 : memref<1024xi32>
    memref.global "public" @in_to_memTile : memref<1024xi32>
    memref.global "public" @in_0_cons : memref<1024xi32>
    memref.global "public" @in_1_cons : memref<1024xi32>
    memref.global "public" @in : memref<1024xi32>
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %out_cons_prod_lock = aie.lock(%shim_noc_tile_0_0, 4) {init = 1 : i32, sym_name = "out_cons_prod_lock"}
    %out_cons_cons_lock = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "out_cons_cons_lock"}
    %out_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "out_buff_0"} : memref<1024xi32> 
    %out_buff_1 = aie.buffer(%mem_tile_0_1) {sym_name = "out_buff_1"} : memref<1024xi32> 
    %out_prod_lock = aie.lock(%mem_tile_0_1, 0) {init = 4 : i32, sym_name = "out_prod_lock"}
    %out_cons_lock = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "out_cons_lock"}
    %out2_buff_0 = aie.buffer(%tile_0_3) {sym_name = "out2_buff_0"} : memref<512xi32> 
    %out2_buff_1 = aie.buffer(%tile_0_3) {sym_name = "out2_buff_1"} : memref<512xi32> 
    %out2_prod_lock = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "out2_prod_lock"}
    %out2_cons_lock = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "out2_cons_lock"}
    %out1_buff_0 = aie.buffer(%tile_0_2) {sym_name = "out1_buff_0"} : memref<512xi32> 
    %out1_buff_1 = aie.buffer(%tile_0_2) {sym_name = "out1_buff_1"} : memref<512xi32> 
    %out1_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "out1_prod_lock"}
    %out1_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out1_cons_lock"}
    %in_from_memTile_1_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "in_from_memTile_1_buff_0"} : memref<1024xi32> 
    %in_from_memTile_1_buff_1 = aie.buffer(%mem_tile_0_1) {sym_name = "in_from_memTile_1_buff_1"} : memref<1024xi32> 
    %in_from_memTile_1_prod_lock = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "in_from_memTile_1_prod_lock"}
    %in_from_memTile_1_cons_lock = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "in_from_memTile_1_cons_lock"}
    %in_from_memTile_0_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "in_from_memTile_0_buff_0"} : memref<1024xi32> 
    %in_from_memTile_0_buff_1 = aie.buffer(%mem_tile_0_1) {sym_name = "in_from_memTile_0_buff_1"} : memref<1024xi32> 
    %in_from_memTile_0_prod_lock = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "in_from_memTile_0_prod_lock"}
    %in_from_memTile_0_cons_lock = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "in_from_memTile_0_cons_lock"}
    %in_to_memTile_prod_lock = aie.lock(%shim_noc_tile_0_0, 2) {init = 1 : i32, sym_name = "in_to_memTile_prod_lock"}
    %in_to_memTile_cons_lock = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "in_to_memTile_cons_lock"}
    %in_0_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "in_0_cons_buff_0"} : memref<1024xi32> 
    %in_0_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "in_0_cons_buff_1"} : memref<1024xi32> 
    %in_0_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in_0_cons_prod_lock"}
    %in_0_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in_0_cons_cons_lock"}
    %in_1_cons_buff_0 = aie.buffer(%tile_0_3) {sym_name = "in_1_cons_buff_0"} : memref<1024xi32> 
    %in_1_cons_buff_1 = aie.buffer(%tile_0_3) {sym_name = "in_1_cons_buff_1"} : memref<1024xi32> 
    %in_1_cons_prod_lock = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "in_1_cons_prod_lock"}
    %in_1_cons_cons_lock = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "in_1_cons_cons_lock"}
    %in_prod_lock = aie.lock(%shim_noc_tile_0_0, 0) {init = 1 : i32, sym_name = "in_prod_lock"}
    %in_cons_lock = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "in_cons_lock"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_3, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c9223372036854775806 step %c2 {
        aie.use_lock(%in_0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%out1_prod_lock, AcquireGreaterEqual, 1)
        %c0_3 = arith.constant 0 : index
        %c512_4 = arith.constant 512 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c512_4 step %c1_5 {
          %0 = memref.load %in_0_cons_buff_0[%arg1] : memref<1024xi32>
          %c1_i32 = arith.constant 1 : i32
          %1 = arith.addi %0, %c1_i32 : i32
          memref.store %1, %out1_buff_0[%arg1] : memref<512xi32>
        }
        aie.use_lock(%in_0_cons_prod_lock, Release, 1)
        aie.use_lock(%out1_cons_lock, Release, 1)
        aie.use_lock(%in_0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%out1_prod_lock, AcquireGreaterEqual, 1)
        %c0_6 = arith.constant 0 : index
        %c512_7 = arith.constant 512 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c512_7 step %c1_8 {
          %0 = memref.load %in_0_cons_buff_1[%arg1] : memref<1024xi32>
          %c1_i32 = arith.constant 1 : i32
          %1 = arith.addi %0, %c1_i32 : i32
          memref.store %1, %out1_buff_1[%arg1] : memref<512xi32>
        }
        aie.use_lock(%in_0_cons_prod_lock, Release, 1)
        aie.use_lock(%out1_cons_lock, Release, 1)
      }
      aie.use_lock(%in_0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%out1_prod_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %c512 = arith.constant 512 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg0 = %c0_1 to %c512 step %c1_2 {
        %0 = memref.load %in_0_cons_buff_0[%arg0] : memref<1024xi32>
        %c1_i32 = arith.constant 1 : i32
        %1 = arith.addi %0, %c1_i32 : i32
        memref.store %1, %out1_buff_0[%arg0] : memref<512xi32>
      }
      aie.use_lock(%in_0_cons_prod_lock, Release, 1)
      aie.use_lock(%out1_cons_lock, Release, 1)
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c9223372036854775806 step %c2 {
        aie.use_lock(%in_1_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%out2_prod_lock, AcquireGreaterEqual, 1)
        %c0_3 = arith.constant 0 : index
        %c512_4 = arith.constant 512 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c512_4 step %c1_5 {
          %0 = memref.load %in_1_cons_buff_0[%arg1] : memref<1024xi32>
          %c1_i32 = arith.constant 1 : i32
          %1 = arith.addi %0, %c1_i32 : i32
          memref.store %1, %out2_buff_0[%arg1] : memref<512xi32>
        }
        aie.use_lock(%in_1_cons_prod_lock, Release, 1)
        aie.use_lock(%out2_cons_lock, Release, 1)
        aie.use_lock(%in_1_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%out2_prod_lock, AcquireGreaterEqual, 1)
        %c0_6 = arith.constant 0 : index
        %c512_7 = arith.constant 512 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_6 to %c512_7 step %c1_8 {
          %0 = memref.load %in_1_cons_buff_1[%arg1] : memref<1024xi32>
          %c1_i32 = arith.constant 1 : i32
          %1 = arith.addi %0, %c1_i32 : i32
          memref.store %1, %out2_buff_1[%arg1] : memref<512xi32>
        }
        aie.use_lock(%in_1_cons_prod_lock, Release, 1)
        aie.use_lock(%out2_cons_lock, Release, 1)
      }
      aie.use_lock(%in_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%out2_prod_lock, AcquireGreaterEqual, 1)
      %c0_1 = arith.constant 0 : index
      %c512 = arith.constant 512 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg0 = %c0_1 to %c512 step %c1_2 {
        %0 = memref.load %in_1_cons_buff_0[%arg0] : memref<1024xi32>
        %c1_i32 = arith.constant 1 : i32
        %1 = arith.addi %0, %c1_i32 : i32
        memref.store %1, %out2_buff_0[%arg0] : memref<512xi32>
      }
      aie.use_lock(%in_1_cons_prod_lock, Release, 1)
      aie.use_lock(%out2_cons_lock, Release, 1)
      aie.end
    }
    aiex.runtime_sequence @sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>])
        aie.end
      } {issue_token = true}
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg2 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
    }
    aie.shim_dma_allocation @in(MM2S, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_0_cons_buff_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%in_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_0_cons_buff_1 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%in_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_buff_0 : memref<512xi32>, 0, 512)
      aie.use_lock(%out1_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out1_buff_1 : memref<512xi32>, 0, 512)
      aie.use_lock(%out1_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_1_cons_buff_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%in_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_1_cons_buff_1 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%in_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_buff_0 : memref<512xi32>, 0, 512)
      aie.use_lock(%out2_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out2_buff_1 : memref<512xi32>, 0, 512)
      aie.use_lock(%out2_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<1024xi32>, 0, 512)
      aie.use_lock(%out_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_1 : memref<1024xi32>, 0, 512)
      aie.use_lock(%out_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<1024xi32>, 512, 512)
      aie.use_lock(%out_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_1 : memref<1024xi32>, 512, 512)
      aie.use_lock(%out_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, 2)
      aie.dma_bd(%out_buff_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%out_prod_lock, Release, 2)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, 2)
      aie.dma_bd(%out_buff_1 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%out_prod_lock, Release, 2)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @out(S2MM, 0, 0)
  }
}

