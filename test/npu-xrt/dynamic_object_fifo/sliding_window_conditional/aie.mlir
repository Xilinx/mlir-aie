module {
  aie.device(npu1_1col) {
    memref.global "public" @output_fifo_cons : memref<10xi32>
    memref.global "public" @output_fifo : memref<10xi32>
    memref.global "public" @input_fifo_cons : memref<10xi32>
    memref.global "public" @input_fifo : memref<10xi32>
    func.func private @add_10_i32(memref<10xi32>, memref<10xi32>, memref<10xi32>)
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
    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<2xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      %c0_0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %buffer_0_2[%c0_0] : memref<2xi32>
      %c1 = arith.constant 1 : index
      %c3_i32 = arith.constant 3 : i32
      memref.store %c0_i32, %buffer_0_2[%c1] : memref<2xi32>
      %c0_1 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg0 = %c0_1 to %c10 step %c1_2 {
        aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
        %0 = memref.load %buffer_0_2[%c0_0] : memref<2xi32>
        %1 = arith.index_cast %0 : i32 to index
        %2 = scf.index_switch %1 -> memref<10xi32> 
        case 0 {
          scf.yield %output_fifo_buff_0 : memref<10xi32>
        }
        case 1 {
          scf.yield %output_fifo_buff_1 : memref<10xi32>
        }
        default {
          scf.yield %output_fifo_buff_0 : memref<10xi32>
        }
        %3 = arith.cmpi eq, %arg0, %c0_1 : index
        %4 = arith.subi %c10, %c1_2 : index
        %5 = arith.cmpi eq, %arg0, %4 : index
        scf.if %3 {
          aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
          %8 = memref.load %buffer_0_2[%c1] : memref<2xi32>
          %9 = arith.index_cast %8 : i32 to index
          %10 = scf.index_switch %9 -> memref<10xi32> 
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
          func.call @add_10_i32(%10, %10, %2) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
        } else {
          scf.if %5 {
            aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 2)
            %8 = memref.load %buffer_0_2[%c1] : memref<2xi32>
            %9 = arith.index_cast %8 : i32 to index
            %10 = scf.index_switch %9 -> memref<10xi32> 
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
            %11 = memref.load %buffer_0_2[%c1] : memref<2xi32>
            %12 = arith.index_cast %11 : i32 to index
            %13 = scf.index_switch %12 -> memref<10xi32> 
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
            func.call @add_10_i32(%10, %13, %2) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
            aie.use_lock(%input_fifo_cons_prod_lock, Release, 2)
            %14 = memref.load %buffer_0_2[%c1] : memref<2xi32>
            %c2_4 = arith.constant 2 : i32
            %15 = arith.addi %14, %c2_4 : i32
            %16 = arith.remsi %15, %c3_i32 : i32
            memref.store %16, %buffer_0_2[%c1] : memref<2xi32>
          } else {
            %8 = memref.load %buffer_0_2[%c1] : memref<2xi32>
            %9 = arith.index_cast %8 : i32 to index
            %10 = scf.index_switch %9 -> memref<10xi32> 
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
            %11 = memref.load %buffer_0_2[%c1] : memref<2xi32>
            %12 = arith.index_cast %11 : i32 to index
            %13 = scf.index_switch %12 -> memref<10xi32> 
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
            func.call @add_10_i32(%10, %13, %2) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
            aie.use_lock(%input_fifo_cons_prod_lock, Release, 1)
            %14 = memref.load %buffer_0_2[%c1] : memref<2xi32>
            %c1_4 = arith.constant 1 : i32
            %15 = arith.addi %14, %c1_4 : i32
            %16 = arith.remsi %15, %c3_i32 : i32
            memref.store %16, %buffer_0_2[%c1] : memref<2xi32>
          }
        }
        aie.use_lock(%output_fifo_cons_lock, Release, 1)
        %6 = memref.load %buffer_0_2[%c0_0] : memref<2xi32>
        %c1_3 = arith.constant 1 : i32
        %7 = arith.addi %6, %c1_3 : i32
        %8 = arith.remsi %7, %c2_i32 : i32
        memref.store %8, %buffer_0_2[%c0_0] : memref<2xi32>
      }
      aie.end
    } {link_with = "kernel.o"}
    aie.shim_dma_allocation @input_fifo(MM2S, 0, 0)
    aiex.runtime_sequence(%arg0: memref<10xi32>, %arg1: memref<10xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 100][0, 0, 0, 1]) {id = 0 : i64, metadata = @input_fifo} : memref<10xi32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 100][0, 0, 0, 1]) {id = 2 : i64, metadata = @output_fifo} : memref<10xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    aie.shim_dma_allocation @output_fifo(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb3
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
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_0 : memref<10xi32>, 0, 10)
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_1 : memref<10xi32>, 0, 10)
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb4
      aie.end
    }
  }
}
