module {
  aie.device(npu1_1col) {
    memref.global "public" @output_fifo_cons : memref<64xi32>
    memref.global "public" @output_fifo : memref<64xi32>
    memref.global "public" @input_fifo_cons : memref<64xi32>
    memref.global "public" @input_fifo : memref<64xi32>
    func.func private @sum_64_i32(memref<64xi32>, memref<64xi32>, memref<64xi32>)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %output_fifo_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "output_fifo_cons_prod_lock"}
    %output_fifo_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "output_fifo_cons_cons_lock"}
    %output_fifo_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "output_fifo_buff_0"} : memref<64xi32> 
    %output_fifo_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "output_fifo_buff_1"} : memref<64xi32> 
    %output_fifo_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock"}
    %output_fifo_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock"}
    %input_fifo_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "input_fifo_cons_buff_0"} : memref<64xi32> 
    %input_fifo_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "input_fifo_cons_buff_1"} : memref<64xi32> 
    %input_fifo_cons_buff_2 = aie.buffer(%tile_0_2) {address = 1280 : i32, mem_bank = 0 : i32, sym_name = "input_fifo_cons_buff_2"} : memref<64xi32> 
    %input_fifo_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 3 : i32, sym_name = "input_fifo_cons_prod_lock"}
    %input_fifo_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock"}
    %input_fifo_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock"}
    %input_fifo_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock"}
    %switchbox_0_0 = aie.switchbox(%tile_0_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<North : 0, South : 2>
    }
    %shim_mux_0_0 = aie.shim_mux(%tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%tile_0_1) {
      aie.connect<South : 0, North : 0>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c0_0 = arith.constant 0 : index
      cf.br ^bb1(%c0, %c0_0 : index, index)
    ^bb1(%0: index, %1: index):  // 2 preds: ^bb0, ^bb24
      %2 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %2, ^bb2, ^bb25
    ^bb2:  // pred: ^bb1
      %3 = arith.remsi %0, %c2 : index
      %4 = arith.cmpi eq, %3, %c0 : index
      %5 = arith.remsi %1, %c3 : index
      %6 = arith.index_cast %5 : index to i32
      cf.switch %6 : i32, [
        default: ^bb6,
        0: ^bb3,
        1: ^bb4,
        2: ^bb5
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb7(%input_fifo_cons_buff_0 : memref<64xi32>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb7(%input_fifo_cons_buff_1 : memref<64xi32>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb7(%input_fifo_cons_buff_2 : memref<64xi32>)
    ^bb6:  // pred: ^bb2
      cf.br ^bb7(%input_fifo_cons_buff_0 : memref<64xi32>)
    ^bb7(%7: memref<64xi32>):  // 4 preds: ^bb3, ^bb4, ^bb5, ^bb6
      %8 = arith.index_cast %5 : index to i32
      cf.switch %8 : i32, [
        default: ^bb11,
        0: ^bb8,
        1: ^bb9,
        2: ^bb10
      ]
    ^bb8:  // pred: ^bb7
      cf.br ^bb12(%input_fifo_cons_buff_1 : memref<64xi32>)
    ^bb9:  // pred: ^bb7
      cf.br ^bb12(%input_fifo_cons_buff_2 : memref<64xi32>)
    ^bb10:  // pred: ^bb7
      cf.br ^bb12(%input_fifo_cons_buff_0 : memref<64xi32>)
    ^bb11:  // pred: ^bb7
      cf.br ^bb12(%input_fifo_cons_buff_0 : memref<64xi32>)
    ^bb12(%9: memref<64xi32>):  // 4 preds: ^bb8, ^bb9, ^bb10, ^bb11
      cf.cond_br %4, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      cf.br ^bb15(%output_fifo_buff_0 : memref<64xi32>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb15(%output_fifo_buff_1 : memref<64xi32>)
    ^bb15(%10: memref<64xi32>):  // 2 preds: ^bb13, ^bb14
      cf.br ^bb16
    ^bb16:  // pred: ^bb15
      aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
      %11 = arith.cmpi eq, %0, %c0 : index
      %12 = arith.subi %c4294967295, %c1 : index
      %13 = arith.cmpi eq, %0, %12 : index
      cf.cond_br %11, ^bb17, ^bb18
    ^bb17:  // pred: ^bb16
      aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
      func.call @sum_64_i32(%7, %7, %10) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
      %c0_1 = arith.constant 0 : index
      cf.br ^bb23(%c0_1 : index)
    ^bb18:  // pred: ^bb16
      cf.cond_br %13, ^bb19, ^bb20
    ^bb19:  // pred: ^bb18
      aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 2)
      func.call @sum_64_i32(%7, %9, %10) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
      aie.use_lock(%input_fifo_cons_prod_lock, Release, 1)
      %c1_2 = arith.constant 1 : index
      cf.br ^bb21(%c1_2 : index)
    ^bb20:  // pred: ^bb18
      aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
      func.call @sum_64_i32(%7, %7, %10) : (memref<64xi32>, memref<64xi32>, memref<64xi32>) -> ()
      aie.use_lock(%input_fifo_cons_prod_lock, Release, 2)
      %c2_3 = arith.constant 2 : index
      cf.br ^bb21(%c2_3 : index)
    ^bb21(%14: index):  // 2 preds: ^bb19, ^bb20
      cf.br ^bb22
    ^bb22:  // pred: ^bb21
      cf.br ^bb23(%14 : index)
    ^bb23(%15: index):  // 2 preds: ^bb17, ^bb22
      cf.br ^bb24
    ^bb24:  // pred: ^bb23
      aie.use_lock(%output_fifo_cons_lock, Release, 1)
      %16 = arith.addi %0, %c1 : index
      cf.br ^bb1(%16, %15 : index, index)
    ^bb25:  // pred: ^bb1
      aie.end
    } {link_with = "kernel.o"}
    aie.shim_dma_allocation @input_fifo(MM2S, 0, 0)
    func.func @sequence(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, metadata = @input_fifo} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 2 : i64, metadata = @output_fifo} : memref<64xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
    aie.shim_dma_allocation @output_fifo(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_0 : memref<64xi32>, 0, 64) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_1 : memref<64xi32>, 0, 64) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_2 : memref<64xi32>, 0, 64) {bd_id = 2 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb4:  // 2 preds: ^bb0, ^bb6
      %1 = aie.dma_start(MM2S, 0, ^bb5, ^bb7)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_0 : memref<64xi32>, 0, 64) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // no predecessors
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_1 : memref<64xi32>, 0, 64) {bd_id = 4 : i32}
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb7:  // pred: ^bb4
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

