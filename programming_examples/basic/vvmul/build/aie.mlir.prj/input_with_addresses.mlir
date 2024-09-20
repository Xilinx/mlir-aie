module {
  aie.device(npu1_1col) {
    memref.global "public" @out_cons : memref<16xi32>
    memref.global "public" @out : memref<16xi32>
    memref.global "public" @in2_cons : memref<16xi32>
    memref.global "public" @in2 : memref<16xi32>
    memref.global "public" @in1_cons : memref<16xi32>
    memref.global "public" @in1 : memref<16xi32>
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %out_cons_prod_lock = aie.lock(%tile_0_0, 4) {init = 1 : i32, sym_name = "out_cons_prod_lock"}
    %out_cons_cons_lock = aie.lock(%tile_0_0, 5) {init = 0 : i32, sym_name = "out_cons_cons_lock"}
    %out_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "out_buff_0"} : memref<16xi32> 
    %out_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_buff_1"} : memref<16xi32> 
    %out_prod_lock = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "out_prod_lock"}
    %out_cons_lock = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "out_cons_lock"}
    %in2_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "in2_cons_buff_0"} : memref<16xi32> 
    %in2_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "in2_cons_buff_1"} : memref<16xi32> 
    %in2_cons_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "in2_cons_prod_lock"}
    %in2_cons_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "in2_cons_cons_lock"}
    %in2_prod_lock = aie.lock(%tile_0_0, 2) {init = 1 : i32, sym_name = "in2_prod_lock"}
    %in2_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "in2_cons_lock"}
    %in1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1088 : i32, mem_bank = 0 : i32, sym_name = "in1_cons_buff_0"} : memref<16xi32> 
    %in1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16448 : i32, mem_bank = 1 : i32, sym_name = "in1_cons_buff_1"} : memref<16xi32> 
    %in1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "in1_cons_prod_lock"}
    %in1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "in1_cons_cons_lock"}
    %in1_prod_lock = aie.lock(%tile_0_0, 0) {init = 1 : i32, sym_name = "in1_prod_lock"}
    %in1_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "in1_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb11
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb12
    ^bb2:  // pred: ^bb1
      %c0_0 = arith.constant 0 : index
      %c256 = arith.constant 256 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb10
      %3 = arith.cmpi slt, %2, %c256 : index
      cf.cond_br %3, ^bb4, ^bb11
    ^bb4:  // pred: ^bb3
      aie.use_lock(%in1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
      %c0_2 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1_3 = arith.constant 1 : index
      cf.br ^bb5(%c0_2 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c16 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %6 = memref.load %in1_cons_buff_0[%4] : memref<16xi32>
      %7 = memref.load %in2_cons_buff_0[%4] : memref<16xi32>
      %8 = arith.muli %6, %7 : i32
      memref.store %8, %out_buff_0[%4] : memref<16xi32>
      %9 = arith.addi %4, %c1_3 : index
      cf.br ^bb5(%9 : index)
    ^bb7:  // pred: ^bb5
      aie.use_lock(%in1_cons_prod_lock, Release, 1)
      aie.use_lock(%in2_cons_prod_lock, Release, 1)
      aie.use_lock(%out_cons_lock, Release, 1)
      aie.use_lock(%in1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%in2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%out_prod_lock, AcquireGreaterEqual, 1)
      %c0_4 = arith.constant 0 : index
      %c16_5 = arith.constant 16 : index
      %c1_6 = arith.constant 1 : index
      cf.br ^bb8(%c0_4 : index)
    ^bb8(%10: index):  // 2 preds: ^bb7, ^bb9
      %11 = arith.cmpi slt, %10, %c16_5 : index
      cf.cond_br %11, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      %12 = memref.load %in1_cons_buff_1[%10] : memref<16xi32>
      %13 = memref.load %in2_cons_buff_1[%10] : memref<16xi32>
      %14 = arith.muli %12, %13 : i32
      memref.store %14, %out_buff_1[%10] : memref<16xi32>
      %15 = arith.addi %10, %c1_6 : index
      cf.br ^bb8(%15 : index)
    ^bb10:  // pred: ^bb8
      aie.use_lock(%in1_cons_prod_lock, Release, 1)
      aie.use_lock(%in2_cons_prod_lock, Release, 1)
      aie.use_lock(%out_cons_lock, Release, 1)
      %16 = arith.addi %2, %c2 : index
      cf.br ^bb3(%16 : index)
    ^bb11:  // pred: ^bb3
      %17 = arith.addi %0, %c1 : index
      cf.br ^bb1(%17 : index)
    ^bb12:  // pred: ^bb1
      aie.end
    }
    aie.shim_dma_allocation @in1(MM2S, 0, 0)
    aiex.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {id = 0 : i64, metadata = @out} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {id = 1 : i64, metadata = @in1} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0, 1]) {id = 2 : i64, metadata = @in2} : memref<4096xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.shim_dma_allocation @in2(MM2S, 1, 0)
    aie.shim_dma_allocation @out(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%in1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_cons_buff_0 : memref<16xi32>, 0, 16) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%in1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%in1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in1_cons_buff_1 : memref<16xi32>, 0, 16) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%in1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%in2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_cons_buff_0 : memref<16xi32>, 0, 16) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%in2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%in2_cons_buff_1 : memref<16xi32>, 0, 16) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<16xi32>, 0, 16) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%out_prod_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%out_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_1 : memref<16xi32>, 0, 16) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%out_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.packet_flow(15) {
      aie.packet_source<%tile_0_0, Ctrl : 0>
      aie.packet_dest<%tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
