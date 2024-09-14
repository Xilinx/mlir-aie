// 4 steps
// 1. Counters and map to the respective for loops
// 2. Respective counters added as iteration arguments to the nested loops
// 3. ADDI statement before the loop : to bring previous iteration count of the inner loop
// 4. ADDI statement after the loop : to update the iteration count of the inner loop to the outer loop.
module {
  aie.device(npu1_1col) {
    memref.global "public" @output_fifo_cons : memref<10xi32>
    memref.global "public" @output_fifo : memref<10xi32>
    memref.global "public" @input_fifo_cons : memref<10xi32>
    memref.global "public" @input_fifo : memref<10xi32>
    func.func private @passthrough_10_i32(memref<10xi32>, memref<10xi32>, index)
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
    %input_fifo_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "input_fifo_cons_prod_lock"}
    %input_fifo_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock"}
    %input_fifo_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock"}
    %input_fifo_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c5 = arith.constant 5 : index
      %c0_0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      // So, a map of all the forLoops with necessary counter arguments based on nesting inside the loop.
      %0:2 = scf.for %arg0 = %c0 to %c5 step %c1 iter_args(%arg1 = %c0_0,%arg2 = %c0_1) -> (index, index) { // Number of iter_args depend on number of nested loops as well
        %c2 = arith.constant 2 : index
        %2 = arith.remsi %arg1, %c2 : index  // Since this is outer loop related, they depend on %arg1
        %3 = scf.index_switch %2 -> memref<10xi32> 
        case 0 {
          scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
        }
        case 1 {
          scf.yield %input_fifo_cons_buff_1 : memref<10xi32>
        }
        default {
          scf.yield %input_fifo_cons_buff_0 : memref<10xi32>
        }
        aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
        %c0_4 = arith.constant 0 : index
        %450 = arith.addi %c0_4, %arg2 : index // Add the previous iterations of inner loop to know exact start point
        %6 = scf.for %arg3 = %c0 to %c5 step %c1 iter_args(%arg4 = %450) -> (index) { // Using second argument here directly, since second counter is the one that was created for this loop
          %c2_7 = arith.constant 2 : index
          %9 = arith.remsi %arg4, %c2_7 : index
          %10 = scf.index_switch %9 -> memref<10xi32> 
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
          func.call @passthrough_10_i32(%3, %10, %2) : (memref<10xi32>, memref<10xi32>, index) -> ()
          aie.use_lock(%output_fifo_cons_lock, Release, 1)
          %c1_8 = arith.constant 1 : index
          %11 = arith.addi %arg4, %c1_8 : index
          scf.yield %11 : index
        }
        aie.use_lock(%input_fifo_cons_prod_lock, Release, 1)
        %c1_5 = arith.constant 1 : index
        %7 = arith.addi %arg1, %c1_5 : index
        %550 = arith.addi %c5, %450 : index // Add the number of iterations the inner loop went through to the outer loop
        scf.yield %7, %550 : index, index
      }
      aie.end
    } {link_with = "kernel.o"}
    aie.shim_dma_allocation @input_fifo(MM2S, 0, 0)
    aiex.runtime_sequence(%arg0: memref<10xi32>, %arg1: memref<10xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 50][0, 0, 0, 1]) {id = 0 : i64, metadata = @input_fifo} : memref<10xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 250][0, 0, 0, 1]) {id = 2 : i64, metadata = @output_fifo} : memref<10xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.shim_dma_allocation @output_fifo(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_0 : memref<10xi32>, 0, 10)
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_1 : memref<10xi32>, 0, 10)
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_0 : memref<10xi32>, 0, 10)
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_1 : memref<10xi32>, 0, 10)
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
  }
}