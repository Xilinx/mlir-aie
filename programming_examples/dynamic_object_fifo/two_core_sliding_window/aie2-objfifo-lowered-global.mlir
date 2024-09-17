module {
  aie.device(npu1_1col) {
    memref.global "public" @output_fifo_cons : memref<10xi32>
    memref.global "public" @output_fifo : memref<10xi32>
    memref.global "public" @inter_fifo_cons : memref<10xi32>
    memref.global "public" @inter_fifo : memref<10xi32>
    memref.global "public" @input_fifo_cons : memref<10xi32>
    memref.global "public" @input_fifo : memref<10xi32>
    func.func private @passthrough_10_i32(memref<10xi32>, memref<10xi32>)
    func.func private @sum_10_i32(memref<10xi32>, memref<10xi32>, memref<10xi32>)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    %output_fifo_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "output_fifo_cons_prod_lock"}
    %output_fifo_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "output_fifo_cons_cons_lock"}
    %output_fifo_buff_0 = aie.buffer(%tile_0_4) {sym_name = "output_fifo_buff_0"} : memref<10xi32> 
    %output_fifo_buff_1 = aie.buffer(%tile_0_4) {sym_name = "output_fifo_buff_1"} : memref<10xi32> 
    %output_fifo_prod_lock = aie.lock(%tile_0_4, 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock"}
    %output_fifo_cons_lock = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock"}
    %inter_fifo_cons_buff_0 = aie.buffer(%tile_0_4) {sym_name = "inter_fifo_cons_buff_0"} : memref<10xi32> 
    %inter_fifo_cons_buff_1 = aie.buffer(%tile_0_4) {sym_name = "inter_fifo_cons_buff_1"} : memref<10xi32> 
    %inter_fifo_cons_buff_2 = aie.buffer(%tile_0_4) {sym_name = "inter_fifo_cons_buff_2"} : memref<10xi32> 
    %inter_fifo_cons_prod_lock = aie.lock(%tile_0_4, 0) {init = 3 : i32, sym_name = "inter_fifo_cons_prod_lock"}
    %inter_fifo_cons_cons_lock = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "inter_fifo_cons_cons_lock"}
    %inter_fifo_buff_0 = aie.buffer(%tile_0_2) {sym_name = "inter_fifo_buff_0"} : memref<10xi32> 
    %inter_fifo_buff_1 = aie.buffer(%tile_0_2) {sym_name = "inter_fifo_buff_1"} : memref<10xi32> 
    %inter_fifo_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "inter_fifo_prod_lock"}
    %inter_fifo_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "inter_fifo_cons_lock"}
    %input_fifo_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "input_fifo_cons_buff_0"} : memref<10xi32> 
    %input_fifo_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "input_fifo_cons_buff_1"} : memref<10xi32> 
    %input_fifo_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "input_fifo_cons_prod_lock"}
    %input_fifo_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock"}
    %input_fifo_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock"}
    %input_fifo_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_4, DMA : 0, %tile_0_0, DMA : 0)
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
      %c1_3 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      scf.for %arg0 = %c0_2 to %c10 step %c1_3 {
        aie.use_lock(%inter_fifo_prod_lock, AcquireGreaterEqual, 1)
        %0 = memref.load %buffer_0_2[%c0_0] : memref<2xindex>
        %1 = scf.index_switch %0 -> memref<10xi32> 
        case 0 {
          scf.yield %inter_fifo_buff_0 : memref<10xi32>
        }
        case 1 {
          scf.yield %inter_fifo_buff_1 : memref<10xi32>
        }
        default {
          scf.yield %inter_fifo_buff_0 : memref<10xi32>
        }
        aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
        %2 = memref.load %buffer_0_2[%c1] : memref<2xindex>
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
        func.call @passthrough_10_i32(%3, %1) : (memref<10xi32>, memref<10xi32>) -> ()
        aie.use_lock(%input_fifo_cons_prod_lock, Release, 1)
        %4 = memref.load %buffer_0_2[%c1] : memref<2xindex>
        %c1_4 = arith.constant 1 : index
        %5 = arith.addi %4, %c1_4 : index
        %6 = arith.remsi %5, %c2_1 : index
        memref.store %6, %buffer_0_2[%c1] : memref<2xindex>
        aie.use_lock(%inter_fifo_cons_lock, Release, 1)
        %7 = memref.load %buffer_0_2[%c0_0] : memref<2xindex>
        %c1_5 = arith.constant 1 : index
        %8 = arith.addi %7, %c1_5 : index
        %9 = arith.remsi %8, %c2 : index
        memref.store %9, %buffer_0_2[%c0_0] : memref<2xindex>
      }
      aie.end
    } {link_with = "kernel.o"}
    %buffer_0_4 = aie.buffer(%tile_0_4) : memref<2xindex> 
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      memref.store %c0, %buffer_0_4[%c0_0] : memref<2xindex>
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      memref.store %c0, %buffer_0_4[%c1] : memref<2xindex>
      %c0_1 = arith.constant 0 : index
      %c1_2 = arith.constant 1 : index
      %c9 = arith.constant 9 : index
      aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
      %0 = memref.load %buffer_0_4[%c0_0] : memref<2xindex>
      %1 = scf.index_switch %0 -> memref<10xi32> 
      case 0 {
        scf.yield %output_fifo_buff_0 : memref<10xi32>
      }
      case 1 {
        scf.yield %output_fifo_buff_1 : memref<10xi32>
      }
      default {
        scf.yield %output_fifo_buff_0 : memref<10xi32>
      }
      aie.use_lock(%inter_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
      %2 = memref.load %buffer_0_4[%c1] : memref<2xindex>
      %3 = scf.index_switch %2 -> memref<10xi32> 
      case 0 {
        scf.yield %inter_fifo_cons_buff_0 : memref<10xi32>
      }
      case 1 {
        scf.yield %inter_fifo_cons_buff_1 : memref<10xi32>
      }
      case 2 {
        scf.yield %inter_fifo_cons_buff_2 : memref<10xi32>
      }
      default {
        scf.yield %inter_fifo_cons_buff_0 : memref<10xi32>
      }
      func.call @sum_10_i32(%3, %3, %1) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
      aie.use_lock(%output_fifo_cons_lock, Release, 1)
      %4 = memref.load %buffer_0_4[%c0_0] : memref<2xindex>
      %c1_3 = arith.constant 1 : index
      %5 = arith.addi %4, %c1_3 : index
      %6 = arith.remsi %5, %c2 : index
      memref.store %6, %buffer_0_4[%c0_0] : memref<2xindex>
      scf.for %arg0 = %c0_1 to %c9 step %c1_2 {
        aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
        %19 = memref.load %buffer_0_4[%c0_0] : memref<2xindex>
        %20 = scf.index_switch %19 -> memref<10xi32> 
        case 0 {
          scf.yield %output_fifo_buff_0 : memref<10xi32>
        }
        case 1 {
          scf.yield %output_fifo_buff_1 : memref<10xi32>
        }
        default {
          scf.yield %output_fifo_buff_0 : memref<10xi32>
        }
        aie.use_lock(%inter_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
        %21 = memref.load %buffer_0_4[%c1] : memref<2xindex>
        %22 = scf.index_switch %21 -> memref<10xi32> 
        case 0 {
          scf.yield %inter_fifo_cons_buff_0 : memref<10xi32>
        }
        case 1 {
          scf.yield %inter_fifo_cons_buff_1 : memref<10xi32>
        }
        case 2 {
          scf.yield %inter_fifo_cons_buff_2 : memref<10xi32>
        }
        default {
          scf.yield %inter_fifo_cons_buff_0 : memref<10xi32>
        }
        %23 = memref.load %buffer_0_4[%c1] : memref<2xindex>
        %24 = scf.index_switch %23 -> memref<10xi32> 
        case 0 {
          scf.yield %inter_fifo_cons_buff_1 : memref<10xi32>
        }
        case 1 {
          scf.yield %inter_fifo_cons_buff_2 : memref<10xi32>
        }
        case 2 {
          scf.yield %inter_fifo_cons_buff_0 : memref<10xi32>
        }
        default {
          scf.yield %inter_fifo_cons_buff_1 : memref<10xi32>
        }
        func.call @sum_10_i32(%22, %24, %20) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
        aie.use_lock(%inter_fifo_cons_prod_lock, Release, 1)
        %25 = memref.load %buffer_0_4[%c1] : memref<2xindex>
        %c1_6 = arith.constant 1 : index
        %26 = arith.addi %25, %c1_6 : index
        %27 = arith.remsi %26, %c3 : index
        memref.store %27, %buffer_0_4[%c1] : memref<2xindex>
        aie.use_lock(%output_fifo_cons_lock, Release, 1)
        %28 = memref.load %buffer_0_4[%c0_0] : memref<2xindex>
        %c1_7 = arith.constant 1 : index
        %29 = arith.addi %28, %c1_7 : index
        %30 = arith.remsi %29, %c2 : index
        memref.store %30, %buffer_0_4[%c0_0] : memref<2xindex>
      }
      aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
      %7 = memref.load %buffer_0_4[%c0_0] : memref<2xindex>
      %8 = scf.index_switch %7 -> memref<10xi32> 
      case 0 {
        scf.yield %output_fifo_buff_0 : memref<10xi32>
      }
      case 1 {
        scf.yield %output_fifo_buff_1 : memref<10xi32>
      }
      default {
        scf.yield %output_fifo_buff_0 : memref<10xi32>
      }
      aie.use_lock(%inter_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
      %9 = memref.load %buffer_0_4[%c1] : memref<2xindex>
      %10 = scf.index_switch %9 -> memref<10xi32> 
      case 0 {
        scf.yield %inter_fifo_cons_buff_0 : memref<10xi32>
      }
      case 1 {
        scf.yield %inter_fifo_cons_buff_1 : memref<10xi32>
      }
      case 2 {
        scf.yield %inter_fifo_cons_buff_2 : memref<10xi32>
      }
      default {
        scf.yield %inter_fifo_cons_buff_0 : memref<10xi32>
      }
      %11 = memref.load %buffer_0_4[%c1] : memref<2xindex>
      %12 = scf.index_switch %11 -> memref<10xi32> 
      case 0 {
        scf.yield %inter_fifo_cons_buff_1 : memref<10xi32>
      }
      case 1 {
        scf.yield %inter_fifo_cons_buff_2 : memref<10xi32>
      }
      case 2 {
        scf.yield %inter_fifo_cons_buff_0 : memref<10xi32>
      }
      default {
        scf.yield %inter_fifo_cons_buff_1 : memref<10xi32>
      }
      func.call @sum_10_i32(%10, %12, %8) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
      aie.use_lock(%inter_fifo_cons_prod_lock, Release, 2)
      %13 = memref.load %buffer_0_4[%c1] : memref<2xindex>
      %c2_4 = arith.constant 2 : index
      %14 = arith.addi %13, %c2_4 : index
      %15 = arith.remsi %14, %c3 : index
      memref.store %15, %buffer_0_4[%c1] : memref<2xindex>
      aie.use_lock(%output_fifo_cons_lock, Release, 1)
      %16 = memref.load %buffer_0_4[%c0_0] : memref<2xindex>
      %c1_5 = arith.constant 1 : index
      %17 = arith.addi %16, %c1_5 : index
      %18 = arith.remsi %17, %c2 : index
      memref.store %18, %buffer_0_4[%c0_0] : memref<2xindex>
      aie.end
    } {link_with = "kernel.o"}
    aie.shim_dma_allocation @input_fifo(MM2S, 0, 0)
    aiex.runtime_sequence(%arg0: memref<10xi32>, %arg1: memref<10xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 100][0, 0, 0, 1]) {id = 0 : i64, metadata = @input_fifo} : memref<10xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 100][0, 0, 0, 1]) {id = 2 : i64, metadata = @output_fifo} : memref<10xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
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
      aie.use_lock(%inter_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inter_fifo_buff_0 : memref<10xi32>, 0, 10)
      aie.use_lock(%inter_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%inter_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inter_fifo_buff_1 : memref<10xi32>, 0, 10)
      aie.use_lock(%inter_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @output_fifo(S2MM, 0, 0)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      aie.use_lock(%inter_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inter_fifo_cons_buff_0 : memref<10xi32>, 0, 10)
      aie.use_lock(%inter_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inter_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inter_fifo_cons_buff_1 : memref<10xi32>, 0, 10)
      aie.use_lock(%inter_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%inter_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inter_fifo_cons_buff_2 : memref<10xi32>, 0, 10)
      aie.use_lock(%inter_fifo_cons_cons_lock, Release, 1)
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

