// aie-opt --aie-objectFifo-stateful-transform ../programming_examples/dynamic_object_fifo/sliding_window/aie2.mlir 
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
    %output_fifo_buff_0 = aie.buffer(%tile_0_2) {sym_name = "output_fifo_buff_0"} : memref<10xi32> 
    %output_fifo_buff_1 = aie.buffer(%tile_0_2) {sym_name = "output_fifo_buff_1"} : memref<10xi32> 
    %output_fifo_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock"}
    %output_fifo_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock"}
    %input_fifo_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32> 
    %input_fifo_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32> 
    %input_fifo_cons_buff_2 = aie.buffer(%tile_0_2) {sym_name = "input_fifo_cons_buff_2"} : memref<10xi32> 
    %input_fifo_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 3 : i32, sym_name = "input_fifo_cons_prod_lock"}
    %input_fifo_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock"}
    %input_fifo_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock"}
    %input_fifo_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index

      aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
      func.call @sum_10_i32(%input_fifo_cons_buff_0, %input_fifo_cons_buff_0, %output_fifo_buff_0) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
      // %rel_in = arith.constant 0 : index // # released objects
      aie.use_lock(%output_fifo_cons_lock, Release, 1)
      // %rel_out = arith.constant 1 : index // # released objects

      %init_input_fifo_counter = arith.constant 0 : index
      %init_output_fifo_counter = arith.constant 1 : index // because of above release
      scf.for %arg0 = %c0 to %c8 step %c1 iter_args(%input_fifo_counter = %init_input_fifo_counter, %output_fifo_counter = %init_output_fifo_counter) -> (index, index) {
        %mod_depth_in = arith.remsi %input_fifo_counter, %c3 : index
        %mod_depth_out = arith.remsi %output_fifo_counter, %c2 : index

        %input_0 = scf.index_switch %mod_depth_in -> memref<10xi32>
        case 0 {
          scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
        }
        case 1 {
          scf.yield %input_fifo_cons_buff_1 : memref<10xi32>
        }
        case 2 {
          scf.yield %input_fifo_cons_buff_2 : memref<10xi32>
        }
        default {
          scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
        }

        %input_1 = scf.index_switch %mod_depth_in -> memref<10xi32>
        case 0 {
          scf.yield %input_fifo_cons_buff_1 : memref<10xi32>
        }
        case 1 {
          scf.yield %input_fifo_cons_buff_2 : memref<10xi32>
        }
        case 2 {
          scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
        }
        default {
          scf.yield %input_fifo_cons_buff_1 : memref<10xi32>
        }

        %output_fifo_buff = scf.index_switch %mod_depth_out -> memref<10xi32>
        case 0 {
          scf.yield %output_fifo_buff_0 : memref<10xi32>
        }
        case 1 {
          scf.yield %output_fifo_buff_1 : memref<10xi32>
        }
        default {
          scf.yield %output_fifo_buff_0 : memref<10xi32>
        }

        aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)

        aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
        func.call @sum_10_i32(%input_0, %input_1, %output_fifo_buff) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
        aie.use_lock(%input_fifo_cons_prod_lock, Release, 1)
        %rel_in = arith.constant 1 : index // # released objects
        
        aie.use_lock(%output_fifo_cons_lock, Release, 1)
        %rel_out = arith.constant 1 : index // # released objects

        %sum_in = arith.addi %input_fifo_counter, %rel_in : index
        %sum_out = arith.addi %output_fifo_counter, %rel_out : index
        scf.yield %sum_in, %sum_out : index, index
      }
      aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
      // TODO: the loop needs to give some information about the last release numbers, we cannot use its bounds
      // as they may change at runtime
      func.call @sum_10_i32(%input_fifo_cons_buff_2, %input_fifo_cons_buff_0, %output_fifo_buff_1) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
      aie.use_lock(%input_fifo_cons_prod_lock, Release, 2)
      //%rel_in = arith.constant 1 : index // # released objects
      aie.use_lock(%output_fifo_cons_lock, Release, 1)
      //%rel_out = arith.constant 1 : index // # released objects

      aie.end
    } {link_with = "kernel.o"}
    aie.shim_dma_allocation @input_fifo(MM2S, 0, 0)
    aiex.runtime_sequence(%arg0: memref<100xi32>, %arg1: memref<100xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 100][0, 0, 0, 1]) {id = 0 : i64, metadata = @input_fifo} : memref<100xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 100][0, 0, 0, 1]) {id = 2 : i64, metadata = @output_fifo} : memref<100xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.shim_dma_allocation @output_fifo(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_0 : memref<10xi32>, 0, 10)
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_1 : memref<10xi32>, 0, 10)
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_2 : memref<10xi32>, 0, 10)
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb4:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb5, ^bb7)
    ^bb5:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_0 : memref<10xi32>, 0, 10)
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb4
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_1 : memref<10xi32>, 0, 10)
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb3
      aie.end
    }
  }
}