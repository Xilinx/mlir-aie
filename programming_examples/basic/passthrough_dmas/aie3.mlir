// Shim -> Mem1 (ping-pong) -> Mem2 (ping-ping-pong-pong) -> Shim
// Unequal sizes: Mem1: 1024, Mem2: 256, 768
module {
  aie.device(npu1_2col) {
    memref.global "public" @in_fwd_cons : memref<1024xi32>
    memref.global "public" @in_fwd : memref<1024xi32>
    memref.global "public" @in_cons : memref<1024xi32>
    memref.global "public" @in : memref<1024xi32>
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %mem_tile_1_1 = aie.tile(1, 1)
    %in_fwd_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 1 : i32, sym_name = "in_fwd_cons_prod_lock_0"}
    %in_fwd_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "in_fwd_cons_cons_lock_0"}
    %in_mem_cons_buff_0 = aie.buffer(%mem_tile_0_1) {sym_name = "in_mem_cons_buff_0"} : memref<1024xi32> 
    %in_mem_cons_buff_1 = aie.buffer(%mem_tile_0_1) {sym_name = "in_mem_cons_buff_1"} : memref<1024xi32> 
    %in_mem_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "in_mem_cons_prod_lock_0"}
    %in_mem_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "in_mem_cons_cons_lock_0"}
    %in_cons_buff_0 = aie.buffer(%mem_tile_1_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "in_cons_buff_0"} : memref<256xi32> 
    %in_cons_buff_1 = aie.buffer(%mem_tile_1_1) {sym_name = "in_cons_buff_1"} : memref<768xi32> 
    %in_cons_buff_pong_0 = aie.buffer(%mem_tile_1_1) {sym_name = "in_cons_buff_pong_0"} : memref<256xi32> 
    %in_cons_buff_pong_1 = aie.buffer(%mem_tile_1_1) {sym_name = "in_cons_buff_pong_1"} : memref<768xi32> 
    %in_cons_prod_lock_0 = aie.lock(%mem_tile_1_1, 0) {init = 1 : i32, sym_name = "in_cons_prod_lock_0"}
    %in_cons_prod_lock_1 = aie.lock(%mem_tile_1_1, 1) {init = 1 : i32, sym_name = "in_cons_prod_lock_1"}
    %in_cons_cons_lock_0 = aie.lock(%mem_tile_1_1, 2) {init = 0 : i32, sym_name = "in_cons_cons_lock_0"}
    %in_cons_cons_lock_1 = aie.lock(%mem_tile_1_1, 3) {init = 0 : i32, sym_name = "in_cons_cons_lock_1"}
    %in_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 1 : i32, sym_name = "in_prod_lock_0"}
    %in_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "in_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aiex.runtime_sequence @sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @in_fwd {
        aie.dma_bd(%arg2 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
    aie.shim_dma_allocation @in(MM2S, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in_mem_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_mem_cons_buff_0 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in_mem_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:
      aie.use_lock(%in_mem_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_mem_cons_buff_1 : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in_mem_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:
      %1 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in_mem_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_mem_cons_buff_0 : memref<1024xi32>, 0, 1024) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%in_mem_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in_mem_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_mem_cons_buff_1 : memref<1024xi32>, 0, 1024) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%in_mem_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb8:
      aie.end
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:
      aie.use_lock(%in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_1 : memref<768xi32>, 256, 768) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb3
    ^bb3:
      aie.use_lock(%in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_pong_0 : memref<256xi32>, 0, 256) {bd_id = 2 : i32, next_bd_id = 3: i32}
      aie.use_lock(%in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb4:
      aie.use_lock(%in_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_pong_1 : memref<768xi32>, 256, 768) {bd_id = 3 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb5:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb6, ^bb10)
    ^bb6:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_0 : memref<256xi32>, 0, 256) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%in_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_1 : memref<768xi32>, 256, 768) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%in_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb4
      aie.use_lock(%in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_pong_0 : memref<256xi32>, 0, 256) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%in_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb9
    ^bb9:  // pred: ^bb4
      aie.use_lock(%in_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_pong_1 : memref<768xi32>, 256, 768) {bd_id = 7 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%in_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb6
    ^bb10:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @in_fwd(S2MM, 0, 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}

