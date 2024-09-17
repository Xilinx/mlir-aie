module {
  aie.device(npu1_1col) {
    memref.global "public" @output_fifo_cons : memref<64xi32>
    memref.global "public" @output_fifo : memref<64xi32>
    memref.global "public" @input_fifo_cons : memref<64xi32>
    memref.global "public" @input_fifo : memref<64xi32>
    func.func private @passthrough_64_i32(memref<64xi32>, memref<64xi32>)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %output_fifo_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "output_fifo_cons_prod_lock"}
    %output_fifo_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "output_fifo_cons_cons_lock"}
    %output_fifo_buff_0 = aie.buffer(%tile_0_2) {sym_name = "output_fifo_buff_0"} : memref<64xi32> 
    %output_fifo_buff_1 = aie.buffer(%tile_0_2) {sym_name = "output_fifo_buff_1"} : memref<64xi32> 
    %output_fifo_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock"}
    %output_fifo_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock"}
    %input_fifo_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "input_fifo_cons_buff_0"} : memref<64xi32> 
    %input_fifo_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "input_fifo_cons_buff_1"} : memref<64xi32> 
    %input_fifo_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "input_fifo_cons_prod_lock"}
    %input_fifo_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock"}
    %input_fifo_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock"}
    %input_fifo_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<2xindex> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      memref.store %c0, %buffer_0_2[%c0_0] : memref<2xindex>
      %c1 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      memref.store %c0, %buffer_0_2[%c1] : memref<2xindex>
      %c0_2 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1_3 = arith.constant 1 : index
      scf.for %arg0 = %c0_2 to %c4294967295 step %c1_3 {
        aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
        %0 = memref.load %buffer_0_2[%c1] : memref<2xindex>
        %1 = scf.index_switch %0 -> memref<64xi32> 
        case 0 {
          scf.yield %input_fifo_cons_buff_0 : memref<64xi32>
        }
        case 1 {
          scf.yield %input_fifo_cons_buff_1 : memref<64xi32>
        }
        default {
          scf.yield %input_fifo_cons_buff_0 : memref<64xi32>
        }
        aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
        %2 = memref.load %buffer_0_2[%c0_0] : memref<2xindex>
        %3 = scf.index_switch %2 -> memref<64xi32> 
        case 0 {
          scf.yield %output_fifo_buff_0 : memref<64xi32>
        }
        case 1 {
          scf.yield %output_fifo_buff_1 : memref<64xi32>
        }
        default {
          scf.yield %output_fifo_buff_0 : memref<64xi32>
        }
        func.call @passthrough_64_i32(%1, %3) : (memref<64xi32>, memref<64xi32>) -> ()
        aie.use_lock(%output_fifo_cons_lock, Release, 1)
        %4 = memref.load %buffer_0_2[%c0_0] : memref<2xindex>
        %c1_4 = arith.constant 1 : index
        %5 = arith.addi %4, %c1_4 : index
        %6 = arith.remsi %5, %c2 : index
        memref.store %6, %buffer_0_2[%c0_0] : memref<2xindex>
        aie.use_lock(%input_fifo_cons_prod_lock, Release, 1)
        %7 = memref.load %buffer_0_2[%c1] : memref<2xindex>
        %c1_5 = arith.constant 1 : index
        %8 = arith.addi %7, %c1_5 : index
        %9 = arith.remsi %8, %c2_1 : index
        memref.store %9, %buffer_0_2[%c1] : memref<2xindex>
      }
      aie.end
    } {link_with = "kernel.o"}
    aie.shim_dma_allocation @input_fifo(MM2S, 0, 0)
    aiex.runtime_sequence(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, metadata = @input_fifo} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 2 : i64, metadata = @output_fifo} : memref<64xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.shim_dma_allocation @output_fifo(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_0 : memref<64xi32>, 0, 64)
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_1 : memref<64xi32>, 0, 64)
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_0 : memref<64xi32>, 0, 64)
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_1 : memref<64xi32>, 0, 64)
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
  }
}

