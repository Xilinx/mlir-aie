module {
  aie.device(npu1_1col) {
    memref.global "public" @mem_to_shim_cons : memref<8x20xi32>
    memref.global "public" @mem_to_shim : memref<8x20xi32>
    memref.global "public" @comp_to_mem_cons : memref<8x20xi32>
    memref.global "public" @comp_to_mem : memref<8x20xi32>
    memref.global "public" @mem_to_comp_cons : memref<8x20xi32>
    memref.global "public" @mem_to_comp : memref<8x20xi32>
    memref.global "public" @shim_to_mem_cons : memref<8x20xi32>
    memref.global "public" @shim_to_mem : memref<8x20xi32>
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %comp_to_mem_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "comp_to_mem_cons_buff_0"} : memref<8x20xi32> 
    %comp_to_mem_cons_buff_1 = aie.buffer(%tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "comp_to_mem_cons_buff_1"} : memref<8x20xi32> 
    %comp_to_mem_cons_prod_lock = aie.lock(%tile_0_1, 2) {init = 2 : i32, sym_name = "comp_to_mem_cons_prod_lock"}
    %comp_to_mem_cons_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "comp_to_mem_cons_cons_lock"}
    %comp_to_mem_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "comp_to_mem_buff_0"} : memref<8x20xi32> 
    %comp_to_mem_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "comp_to_mem_buff_1"} : memref<8x20xi32> 
    %comp_to_mem_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "comp_to_mem_prod_lock"}
    %comp_to_mem_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "comp_to_mem_cons_lock"}
    %mem_to_comp_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "mem_to_comp_cons_buff_0"} : memref<8x20xi32> 
    %mem_to_comp_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "mem_to_comp_cons_buff_1"} : memref<8x20xi32> 
    %mem_to_comp_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "mem_to_comp_cons_prod_lock"}
    %mem_to_comp_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "mem_to_comp_cons_cons_lock"}
    %shim_to_mem_cons_buff_0 = aie.buffer(%tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "shim_to_mem_cons_buff_0"} : memref<8x20xi32> 
    %shim_to_mem_cons_buff_1 = aie.buffer(%tile_0_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "shim_to_mem_cons_buff_1"} : memref<8x20xi32> 
    %shim_to_mem_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 2 : i32, sym_name = "shim_to_mem_cons_prod_lock"}
    %shim_to_mem_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "shim_to_mem_cons_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c20 = arith.constant 20 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9223372036854775806 = arith.constant 9223372036854775806 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb14
      %1 = arith.cmpi slt, %0, %c9223372036854775806 : index
      cf.cond_br %1, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      aie.use_lock(%mem_to_comp_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%comp_to_mem_prod_lock, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
      %3 = arith.cmpi slt, %2, %c8 : index
      cf.cond_br %3, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      cf.br ^bb5(%c0 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c20 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %6 = memref.load %mem_to_comp_cons_buff_0[%2, %4] : memref<8x20xi32>
      memref.store %6, %comp_to_mem_buff_0[%2, %4] : memref<8x20xi32>
      %7 = arith.addi %4, %c1 : index
      cf.br ^bb5(%7 : index)
    ^bb7:  // pred: ^bb5
      %8 = arith.addi %2, %c1 : index
      cf.br ^bb3(%8 : index)
    ^bb8:  // pred: ^bb3
      aie.use_lock(%mem_to_comp_cons_prod_lock, Release, 1)
      aie.use_lock(%comp_to_mem_cons_lock, Release, 1)
      aie.use_lock(%mem_to_comp_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%comp_to_mem_prod_lock, AcquireGreaterEqual, 1)
      cf.br ^bb9(%c0 : index)
    ^bb9(%9: index):  // 2 preds: ^bb8, ^bb13
      %10 = arith.cmpi slt, %9, %c8 : index
      cf.cond_br %10, ^bb10, ^bb14
    ^bb10:  // pred: ^bb9
      cf.br ^bb11(%c0 : index)
    ^bb11(%11: index):  // 2 preds: ^bb10, ^bb12
      %12 = arith.cmpi slt, %11, %c20 : index
      cf.cond_br %12, ^bb12, ^bb13
    ^bb12:  // pred: ^bb11
      %13 = memref.load %mem_to_comp_cons_buff_1[%9, %11] : memref<8x20xi32>
      memref.store %13, %comp_to_mem_buff_1[%9, %11] : memref<8x20xi32>
      %14 = arith.addi %11, %c1 : index
      cf.br ^bb11(%14 : index)
    ^bb13:  // pred: ^bb11
      %15 = arith.addi %9, %c1 : index
      cf.br ^bb9(%15 : index)
    ^bb14:  // pred: ^bb9
      aie.use_lock(%mem_to_comp_cons_prod_lock, Release, 1)
      aie.use_lock(%comp_to_mem_cons_lock, Release, 1)
      %16 = arith.addi %0, %c2 : index
      cf.br ^bb1(%16 : index)
    ^bb15:  // pred: ^bb1
      aie.use_lock(%mem_to_comp_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%comp_to_mem_prod_lock, AcquireGreaterEqual, 1)
      cf.br ^bb16(%c0 : index)
    ^bb16(%17: index):  // 2 preds: ^bb15, ^bb20
      %18 = arith.cmpi slt, %17, %c8 : index
      cf.cond_br %18, ^bb17, ^bb21
    ^bb17:  // pred: ^bb16
      cf.br ^bb18(%c0 : index)
    ^bb18(%19: index):  // 2 preds: ^bb17, ^bb19
      %20 = arith.cmpi slt, %19, %c20 : index
      cf.cond_br %20, ^bb19, ^bb20
    ^bb19:  // pred: ^bb18
      %21 = memref.load %mem_to_comp_cons_buff_0[%17, %19] : memref<8x20xi32>
      memref.store %21, %comp_to_mem_buff_0[%17, %19] : memref<8x20xi32>
      %22 = arith.addi %19, %c1 : index
      cf.br ^bb18(%22 : index)
    ^bb20:  // pred: ^bb18
      %23 = arith.addi %17, %c1 : index
      cf.br ^bb16(%23 : index)
    ^bb21:  // pred: ^bb16
      aie.use_lock(%mem_to_comp_cons_prod_lock, Release, 1)
      aie.use_lock(%comp_to_mem_cons_lock, Release, 1)
      aie.end
    }
    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[160, 0, 0, 0, -2147483648, 0, 0, 33554432]>
    aiex.runtime_sequence(%arg0: memref<160xi32>, %arg1: memref<160xi32>, %arg2: memref<160xi32>) {
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119316 : ui32, column = 0 : i32, row = 0 : i32, value = 1 : ui32}
      %1 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%1) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119296 : ui32, column = 0 : i32, mask = 3840 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483648 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.shim_dma_allocation @shim_to_mem(MM2S, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%shim_to_mem_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%shim_to_mem_cons_buff_0 : memref<8x20xi32>, 0, 160) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_to_mem_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%shim_to_mem_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%shim_to_mem_cons_buff_1 : memref<8x20xi32>, 0, 160) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%shim_to_mem_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%shim_to_mem_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%shim_to_mem_cons_buff_0 : memref<8x20xi32>, 0, 160, [<size = 1, stride = 160>, <size = 4, stride = 5>, <size = 8, stride = 20>, <size = 5, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%shim_to_mem_cons_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%shim_to_mem_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%shim_to_mem_cons_buff_1 : memref<8x20xi32>, 0, 160, [<size = 1, stride = 160>, <size = 4, stride = 5>, <size = 8, stride = 20>, <size = 5, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%shim_to_mem_cons_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%comp_to_mem_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%comp_to_mem_cons_buff_0 : memref<8x20xi32>, 0, 160) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%comp_to_mem_cons_cons_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%comp_to_mem_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%comp_to_mem_cons_buff_1 : memref<8x20xi32>, 0, 160) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%comp_to_mem_cons_cons_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%comp_to_mem_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%comp_to_mem_cons_buff_0 : memref<8x20xi32>, 0, 160) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%comp_to_mem_cons_prod_lock, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%comp_to_mem_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%comp_to_mem_cons_buff_1 : memref<8x20xi32>, 0, 160) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%comp_to_mem_cons_prod_lock, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%mem_to_comp_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_to_comp_cons_buff_0 : memref<8x20xi32>, 0, 160, [<size = 4, stride = 20>, <size = 2, stride = 80>, <size = 20, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%mem_to_comp_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%mem_to_comp_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%mem_to_comp_cons_buff_1 : memref<8x20xi32>, 0, 160, [<size = 4, stride = 20>, <size = 2, stride = 80>, <size = 20, stride = 1>]) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%mem_to_comp_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%comp_to_mem_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%comp_to_mem_buff_0 : memref<8x20xi32>, 0, 160) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%comp_to_mem_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%comp_to_mem_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%comp_to_mem_buff_1 : memref<8x20xi32>, 0, 160) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%comp_to_mem_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @mem_to_shim(S2MM, 0, 0)
    aie.packet_flow(15) {
      aie.packet_source<%tile_0_0, Ctrl : 0>
      aie.packet_dest<%tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}

