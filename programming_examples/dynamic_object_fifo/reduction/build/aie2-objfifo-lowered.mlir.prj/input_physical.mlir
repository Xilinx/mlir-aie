module {
  aie.device(npu1_1col) {
    memref.global "public" @output_fifo_cons : memref<10xi32>
    memref.global "public" @output_fifo : memref<10xi32>
    memref.global "public" @input_fifo_cons : memref<10xi32>
    memref.global "public" @input_fifo : memref<10xi32>
    func.func private @sum_10_i32(memref<10xi32>, memref<10xi32>, memref<10xi32>)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %output_fifo_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "output_fifo_cons_prod_lock"}
    %output_fifo_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "output_fifo_cons_cons_lock"}
    %output_fifo_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "output_fifo_buff_0"} : memref<10xi32> 
    %output_fifo_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "output_fifo_buff_1"} : memref<10xi32> 
    %output_fifo_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock"}
    %output_fifo_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock"}
    %input_fifo_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "input_fifo_cons_buff_0"} : memref<10xi32> 
    %input_fifo_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "input_fifo_cons_buff_1"} : memref<10xi32> 
    %input_fifo_cons_buff_2 = aie.buffer(%tile_0_2) {address = 1064 : i32, mem_bank = 0 : i32, sym_name = "input_fifo_cons_buff_2"} : memref<10xi32> 
    %input_fifo_cons_buff_3 = aie.buffer(%tile_0_2) {address = 16424 : i32, mem_bank = 1 : i32, sym_name = "input_fifo_cons_buff_3"} : memref<10xi32> 
    %input_fifo_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 4 : i32, sym_name = "input_fifo_cons_prod_lock"}
    %input_fifo_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock"}
    %input_fifo_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock"}
    %input_fifo_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock"}
    %switchbox_0_0 = aie.switchbox(%tile_0_0) {
      aie.connect<South : 3, North : 1>
      aie.connect<North : 0, South : 2>
    }
    %shim_mux_0_0 = aie.shim_mux(%tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%tile_0_1) {
      aie.connect<South : 1, North : 1>
      aie.connect<North : 0, South : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<DMA : 0, South : 0>
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c0_0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      cf.br ^bb1(%c0, %c0_0, %c0_1 : index, index, index)
    ^bb1(%0: index, %1: index, %2: index):  // 2 preds: ^bb0, ^bb14
      %3 = arith.cmpi slt, %0, %c10 : index
      cf.cond_br %3, ^bb2, ^bb15
    ^bb2:  // pred: ^bb1
      %4 = arith.remsi %1, %c2 : index
      %5 = arith.remsi %2, %c2 : index
      %6 = arith.index_cast %4 : index to i32
      cf.switch %6 : i32, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%input_fifo_cons_buff_0 : memref<10xi32>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%input_fifo_cons_buff_2 : memref<10xi32>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%input_fifo_cons_buff_0 : memref<10xi32>)
    ^bb6(%7: memref<10xi32>):  // 3 preds: ^bb3, ^bb4, ^bb5
      %8 = arith.index_cast %4 : index to i32
      cf.switch %8 : i32, [
        default: ^bb9,
        0: ^bb7,
        1: ^bb8
      ]
    ^bb7:  // pred: ^bb6
      cf.br ^bb10(%input_fifo_cons_buff_1 : memref<10xi32>)
    ^bb8:  // pred: ^bb6
      cf.br ^bb10(%input_fifo_cons_buff_3 : memref<10xi32>)
    ^bb9:  // pred: ^bb6
      cf.br ^bb10(%input_fifo_cons_buff_1 : memref<10xi32>)
    ^bb10(%9: memref<10xi32>):  // 3 preds: ^bb7, ^bb8, ^bb9
      %10 = arith.index_cast %5 : index to i32
      cf.switch %10 : i32, [
        default: ^bb13,
        0: ^bb11,
        1: ^bb12
      ]
    ^bb11:  // pred: ^bb10
      cf.br ^bb14(%output_fifo_buff_0 : memref<10xi32>)
    ^bb12:  // pred: ^bb10
      cf.br ^bb14(%output_fifo_buff_1 : memref<10xi32>)
    ^bb13:  // pred: ^bb10
      cf.br ^bb14(%output_fifo_buff_1 : memref<10xi32>)
    ^bb14(%11: memref<10xi32>):  // 3 preds: ^bb11, ^bb12, ^bb13
      aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 2)
      func.call @sum_10_i32(%7, %9, %11) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
      %c1_2 = arith.constant 1 : index
      aie.use_lock(%output_fifo_cons_lock, Release, 1)
      %c1_3 = arith.constant 1 : index
      aie.use_lock(%input_fifo_cons_prod_lock, Release, 2)
      %12 = arith.addi %1, %c1_2 : index
      %13 = arith.addi %2, %c1_3 : index
      %14 = arith.addi %0, %c1 : index
      cf.br ^bb1(%14, %12, %13 : index, index, index)
    ^bb15:  // pred: ^bb1
      aie.end
    } {link_with = "kernel.o"}
    aie.shim_dma_allocation @input_fifo(MM2S, 0, 0)
    aiex.runtime_sequence @sequence(%arg0: memref<100xi32>, %arg1: memref<50xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 100][0, 0, 0, 1]) {id = 0 : i64, metadata = @input_fifo} : memref<100xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 50][0, 0, 0, 1]) {id = 2 : i64, metadata = @output_fifo} : memref<50xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.shim_dma_allocation @output_fifo(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb4
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_0 : memref<10xi32>, 0, 10) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_1 : memref<10xi32>, 0, 10) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_2 : memref<10xi32>, 0, 10) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb4:  // pred: ^bb3
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_3 : memref<10xi32>, 0, 10) {bd_id = 3 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb5:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_0 : memref<10xi32>, 0, 10) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_1 : memref<10xi32>, 0, 10) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // pred: ^bb5
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

