module {
  aie.device(npu1_1col) {
    memref.global "public" @objFifo_out0_cons : memref<16xi32>
    memref.global "public" @objFifo_out0 : memref<16xi32>
    memref.global "public" @objFifo_out1_cons : memref<8xi32>
    memref.global "public" @objFifo_out1 : memref<8xi32>
    memref.global "public" @objFifo_in1_cons : memref<8xi32>
    memref.global "public" @objFifo_in1 : memref<8xi32>
    memref.global "public" @objFifo_in0_cons : memref<16xi32>
    memref.global "public" @objFifo_in0 : memref<16xi32>
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %objFifo_out0_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "objFifo_out0_cons_prod_lock"}
    %objFifo_out0_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "objFifo_out0_cons_cons_lock"}
    %objFifo_out0_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "objFifo_out0_buff_0"} : memref<16xi32> 
    %objFifo_out0_buff_1 = aie.buffer(%tile_0_1) {address = 64 : i32, mem_bank = 0 : i32, sym_name = "objFifo_out0_buff_1"} : memref<16xi32> 
    %objFifo_out0_prod_lock = aie.lock(%tile_0_1, 2) {init = 2 : i32, sym_name = "objFifo_out0_prod_lock"}
    %objFifo_out0_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "objFifo_out0_cons_lock"}
    %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "objFifo_out1_buff_0"} : memref<8xi32> 
    %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "objFifo_out1_buff_1"} : memref<8xi32> 
    %objFifo_out1_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "objFifo_out1_prod_lock"}
    %objFifo_out1_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock"}
    %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "objFifo_in1_cons_buff_0"} : memref<8xi32> 
    %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "objFifo_in1_cons_buff_1"} : memref<8xi32> 
    %objFifo_in1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "objFifo_in1_cons_prod_lock"}
    %objFifo_in1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock"}
    %objFifo_in0_cons_buff_0 = aie.buffer(%tile_0_1) {address = 128 : i32, mem_bank = 0 : i32, sym_name = "objFifo_in0_cons_buff_0"} : memref<16xi32> 
    %objFifo_in0_cons_buff_1 = aie.buffer(%tile_0_1) {address = 192 : i32, mem_bank = 0 : i32, sym_name = "objFifo_in0_cons_buff_1"} : memref<16xi32> 
    %objFifo_in0_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 2 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
    %objFifo_in0_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
    %objFifo_in0_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "objFifo_in0_prod_lock"}
    %objFifo_in0_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_lock"}
    %switchbox_0_0 = aie.switchbox(%tile_0_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<North : 0, South : 2>
    }
    %shim_mux_0_0 = aie.shim_mux(%tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_1 = aie.switchbox(%tile_0_1) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, North : 0>
      aie.connect<North : 0, DMA : 1>
      aie.connect<DMA : 1, South : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_i32 = arith.constant 2 : i32
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c8 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)
      cf.br ^bb3(%c0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c8 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %4 = memref.load %objFifo_in1_cons_buff_0[%2] : memref<8xi32>
      %5 = arith.addi %4, %c2_i32 : i32
      memref.store %5, %objFifo_out1_buff_0[%2] : memref<8xi32>
      %6 = arith.addi %2, %c1 : index
      cf.br ^bb3(%6 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
      aie.use_lock(%objFifo_out1_cons_lock, Release, 1)
      aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)
      cf.br ^bb6(%c0 : index)
    ^bb6(%7: index):  // 2 preds: ^bb5, ^bb7
      %8 = arith.cmpi slt, %7, %c8 : index
      cf.cond_br %8, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      %9 = memref.load %objFifo_in1_cons_buff_1[%7] : memref<8xi32>
      %10 = arith.addi %9, %c2_i32 : i32
      memref.store %10, %objFifo_out1_buff_1[%7] : memref<8xi32>
      %11 = arith.addi %7, %c1 : index
      cf.br ^bb6(%11 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
      aie.use_lock(%objFifo_out1_cons_lock, Release, 1)
      %12 = arith.addi %0, %c2 : index
      cf.br ^bb1(%12 : index)
    ^bb9:  // pred: ^bb1
      aie.end
    }
    aie.shim_dma_allocation @objFifo_in0(MM2S, 0, 0)
    aiex.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<32xi32>, %arg2: memref<64xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c64_i64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c64_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @objFifo_out0} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c64_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @objFifo_in0} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @objFifo_out0}
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<16xi32>, 0, 16) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in0_cons_buff_1 : memref<16xi32>, 0, 16) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<16xi32>, 0, 16) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in0_cons_buff_1 : memref<16xi32>, 0, 16) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out0_buff_0 : memref<16xi32>, 0, 16) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out0_buff_1 : memref<16xi32>, 0, 16) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out0_buff_0 : memref<16xi32>, 0, 16) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out0_buff_1 : memref<16xi32>, 0, 16) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    aie.shim_dma_allocation @objFifo_out0(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<8xi32>, 0, 8) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_in1_cons_buff_1 : memref<8xi32>, 0, 8) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out1_buff_0 : memref<8xi32>, 0, 8) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%objFifo_out1_buff_1 : memref<8xi32>, 0, 8) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
  }
}

