module {
  aie.device(npu1_1col) {
    memref.global "public" @out_cons : memref<1xi32>
    memref.global "public" @out : memref<1xi32>
    memref.global "public" @out_join0_cons : memref<1xi32>
    memref.global "public" @out_join0 : memref<1xi32>
    memref.global "public" @out_join1_cons : memref<1xi32>
    memref.global "public" @out_join1 : memref<1xi32>
    memref.global "public" @in_split1_cons : memref<512xi32>
    memref.global "public" @in_split1 : memref<512xi32>
    memref.global "public" @in_split0_cons : memref<512xi32>
    memref.global "public" @in_split0 : memref<512xi32>
    memref.global "public" @in_cons : memref<512xi32>
    memref.global "public" @in : memref<512xi32>
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %out_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 1 : i32, sym_name = "out_cons_prod_lock"}
    %out_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "out_cons_cons_lock"}
    %out_buff_0 = aie.buffer(%tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "out_buff_0"} : memref<1xi32> 
    %out_buff_1 = aie.buffer(%tile_0_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "out_buff_1"} : memref<1xi32> 
    %out_prod_lock0 = aie.lock(%tile_0_1, 2) {init = 1 : i32, sym_name = "out_prod_lock0"}
    %out_cons_lock0 = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "out_cons_lock0"}
    %out_prod_lock1 = aie.lock(%tile_0_1, 4) {init = 1 : i32, sym_name = "out_prod_lock1"}
    %out_cons_lock1 = aie.lock(%tile_0_1, 5) {init = 0 : i32, sym_name = "out_cons_lock1"}
    %out_join0_buff_0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_join0_buff_0"} : memref<1xi32> 
    %out_join0_prod_lock = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "out_join0_prod_lock"}
    %out_join0_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out_join0_cons_lock"}
    %out_join1_buff_0 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_join1_buff_0"} : memref<1xi32> 
    %out_join1_prod_lock = aie.lock(%tile_0_3, 2) {init = 1 : i32, sym_name = "out_join1_prod_lock"}
    %out_join1_cons_lock = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "out_join1_cons_lock"}
    %in_split1_cons_buff_0 = aie.buffer(%tile_0_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "in_split1_cons_buff_0"} : memref<512xi32> 
    %in_split1_cons_prod_lock = aie.lock(%tile_0_3, 0) {init = 1 : i32, sym_name = "in_split1_cons_prod_lock"}
    %in_split1_cons_cons_lock = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "in_split1_cons_cons_lock"}
    %in_split0_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "in_split0_cons_buff_0"} : memref<512xi32> 
    %in_split0_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "in_split0_cons_prod_lock"}
    %in_split0_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in_split0_cons_cons_lock"}
    %in_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "in_cons_buff_0"} : memref<512xi32> 
    %in_cons_buff_1 = aie.buffer(%tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "in_cons_buff_1"} : memref<512xi32> 
    %in_cons_prod_lock0 = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "in_cons_prod_lock0"}
    %in_cons_cons_lock0 = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "in_cons_cons_lock0"}
    %in_cons_prod_lock1 = aie.lock(%tile_0_1, 6) {init = 1 : i32, sym_name = "in_cons_prod_lock1"}
    %in_cons_cons_lock1 = aie.lock(%tile_0_1, 7) {init = 0 : i32, sym_name = "in_cons_cons_lock1"}
    %in_prod_lock = aie.lock(%tile_0_0, 0) {init = 1 : i32, sym_name = "in_prod_lock"}
    %in_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "in_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 2)
    aie.flow(%tile_0_1, DMA : 2, %tile_0_0, DMA : 0)
    func.func private @reduce_max_vector(memref<512xi32>, memref<1xi32>, i32)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out_join0_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%in_split0_cons_cons_lock, AcquireGreaterEqual, 1)
      %c512_i32 = arith.constant 512 : i32
      func.call @reduce_max_vector(%in_split0_cons_buff_0, %out_join0_buff_0, %c512_i32) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%in_split0_cons_prod_lock, Release, 1)
      aie.use_lock(%out_join0_cons_lock, Release, 1)
      %2 = arith.addi %0, %c1 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.end
    } {link_with = "reduce_max.cc.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%out_join1_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%in_split1_cons_cons_lock, AcquireGreaterEqual, 1)
      %c512_i32 = arith.constant 512 : i32
      func.call @reduce_max_vector(%in_split1_cons_buff_0, %out_join1_buff_0, %c512_i32) : (memref<512xi32>, memref<1xi32>, i32) -> ()
      aie.use_lock(%in_split1_cons_prod_lock, Release, 1)
      aie.use_lock(%out_join1_cons_lock, Release, 1)
      %2 = arith.addi %0, %c1 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.end
    } {link_with = "reduce_max.cc.o"}
    aiex.runtime_sequence(%arg0: memref<512x512xi32>, %arg1: memref<512xi32>) {
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<512x512xi32>, 0, 262144, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<512xi32>, 0, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
    aie.shim_dma_allocation @in(MM2S, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in_cons_prod_lock0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_0 : memref<512xi32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in_cons_cons_lock0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in_cons_prod_lock1, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_1 : memref<512xi32>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in_cons_cons_lock1, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in_cons_cons_lock0, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_0 : memref<512xi32>, 0, 512) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in_cons_prod_lock0, Release, 1)
      aie.next_bd ^bb4
   // ^bb5:  // pred: ^bb4
   //   aie.use_lock(%in_cons_cons_lock0, AcquireGreaterEqual, 1)
   //   aie.dma_bd(%in_cons_buff_1 : memref<512xi32>, 0, 512) {bd_id = 3 : i32, next_bd_id = 2 : i32}
   //   aie.use_lock(%in_cons_prod_lock0, Release, 1)
   //   aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%in_cons_cons_lock1, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_cons_buff_1 : memref<512xi32>, 0, 512) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%in_cons_prod_lock1, Release, 1)
      aie.next_bd ^bb7
   // ^bb8:  // pred: ^bb7
   //   aie.use_lock(%in_cons_cons_lock1, AcquireGreaterEqual, 1)
   //   aie.dma_bd(%in_cons_buff_1 : memref<512xi32>, 512, 0) {bd_id = 25 : i32, next_bd_id = 24 : i32}
   //   aie.use_lock(%in_cons_prod_lock1, Release, 1)
   //   aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(S2MM, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%out_prod_lock1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_1 : memref<1xi32>, 0, 1) {bd_id = 26 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%out_cons_lock1, Release, 1)
      aie.next_bd ^bb10
    //^bb11:  // pred: ^bb10
    //  aie.use_lock(%out_prod_lock0, AcquireGreaterEqual, 1)
    //  aie.dma_bd(%out_buff_1 : memref<1xi32>, 1, 0) {bd_id = 27 : i32, next_bd_id = 26 : i32}
    //  aie.use_lock(%out_cons_lock0, Release, 1)
    //  aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%out_prod_lock0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<1xi32>, 0, 1) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_cons_lock0, Release, 1)
      aie.next_bd ^bb13
    //^bb14:  // pred: ^bb13
    //  aie.use_lock(%out_prod_lock1, AcquireGreaterEqual, 1)
    //  aie.dma_bd(%out_buff_1 : memref<1xi32>, 0, 1) {bd_id = 5 : i32, next_bd_id = 4 : i32}
    //  aie.use_lock(%out_cons_lock1, Release, 1)
    //  aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%out_cons_lock0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<1xi32>, 0, 1) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%out_prod_lock0, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%out_cons_lock1, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_1 : memref<1xi32>, 0, 1) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%out_prod_lock1, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%in_split0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_split0_cons_buff_0 : memref<512xi32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in_split0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%out_join0_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_join0_buff_0 : memref<1xi32>, 0, 1) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%out_join0_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%in_split1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_split1_cons_buff_0 : memref<512xi32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in_split1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%out_join1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_join1_buff_0 : memref<1xi32>, 0, 1) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%out_join1_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    aie.shim_dma_allocation @out(S2MM, 0, 0)
    aie.packet_flow(15) {
      aie.packet_source<%tile_0_0, Ctrl : 0>
      aie.packet_dest<%tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
