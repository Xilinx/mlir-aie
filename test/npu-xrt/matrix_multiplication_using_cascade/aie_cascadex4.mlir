//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1) {
    func.func private @matmul_scalar_put_4x1x4_4x4x4_i32_i32(memref<1x4x4x4xi32, 2 : i32>, memref<4x1x4x4xi32, 2 : i32>, memref<4x4x4x4xi32, 2 : i32>) attributes {link_with = "mm.o"}
    func.func private @matmul_scalar_put_get_4x1x4_4x4x4_i32_i32(memref<1x4x4x4xi32, 2 : i32>, memref<4x1x4x4xi32, 2 : i32>, memref<4x4x4x4xi32, 2 : i32>) attributes {link_with = "mm.o"}
    func.func private @matmul_scalar_get_4x1x4_4x4x4_i32_i32(memref<1x4x4x4xi32, 2 : i32>, memref<4x1x4x4xi32, 2 : i32>, memref<4x4x4x4xi32, 2 : i32>) attributes {link_with = "mm.o"}
    // <trace>
    func.func private @event_0() attributes {link_with = "mm.o"}
    func.func private @event_1() attributes {link_with = "mm.o"}
    func.func private @flush_trace() attributes {link_with = "mm.o"}
    // </trace>
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    // <trace>
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    // </trace>
    %lock_1_1 = aie.lock(%tile_1_1, 1) {init = 4 : i32}
    %lock_1_1_0 = aie.lock(%tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%tile_0_1, 3) {init = 4 : i32}
    %lock_0_1_1 = aie.lock(%tile_0_1, 2) {init = 0 : i32}
    %lock_0_1_2 = aie.lock(%tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_3 = aie.lock(%tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_5 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_6 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_7 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_8 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_9 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_10 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_11 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_12 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_13 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_14 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_15 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_16 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_17 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %buf14 = aie.buffer(%tile_0_1) {mem_bank = 0 : i32, sym_name = "buf14"} : memref<16x16xi32, 1 : i32>
    %buf13 = aie.buffer(%tile_1_1) {mem_bank = 0 : i32, sym_name = "buf13"} : memref<16x16xi32, 1 : i32>
    %buf12 = aie.buffer(%tile_0_1) {mem_bank = 0 : i32, sym_name = "buf12"} : memref<16x16xi32, 1 : i32>
    %buf11 = aie.buffer(%tile_3_2) {mem_bank = 0 : i32, sym_name = "buf11"} : memref<1x4x4x4xi32, 2 : i32>
    %buf10 = aie.buffer(%tile_3_2) {mem_bank = 0 : i32, sym_name = "buf10"} : memref<4x1x4x4xi32, 2 : i32>
    %buf9 = aie.buffer(%tile_3_2) {mem_bank = 0 : i32, sym_name = "buf9"} : memref<4x4x4x4xi32, 2 : i32>
    %buf8 = aie.buffer(%tile_2_2) {mem_bank = 0 : i32, sym_name = "buf8"} : memref<1x4x4x4xi32, 2 : i32>
    %buf7 = aie.buffer(%tile_2_2) {mem_bank = 0 : i32, sym_name = "buf7"} : memref<4x1x4x4xi32, 2 : i32>
    %buf6 = aie.buffer(%tile_2_2) {mem_bank = 0 : i32, sym_name = "buf6"} : memref<4x4x4x4xi32, 2 : i32>
    %buf5 = aie.buffer(%tile_1_2) {mem_bank = 0 : i32, sym_name = "buf5"} : memref<1x4x4x4xi32, 2 : i32>
    %buf4 = aie.buffer(%tile_1_2) {mem_bank = 0 : i32, sym_name = "buf4"} : memref<4x1x4x4xi32, 2 : i32>
    %buf3 = aie.buffer(%tile_1_2) {mem_bank = 0 : i32, sym_name = "buf3"} : memref<4x4x4x4xi32, 2 : i32>
    %buf2 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf2"} : memref<1x4x4x4xi32, 2 : i32>
    %buf1 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf1"} : memref<4x1x4x4xi32, 2 : i32>
    %buf0 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf0"} : memref<4x4x4x4xi32, 2 : i32>
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, %c1_ul1)
      aie.dma_bd(%buf2 : memref<1x4x4x4xi32, 2 : i32> offset = 0 len = 64)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(%lock_0_2_6, Release, %c1_ul2)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, %c1_ul3)
      aie.dma_bd(%buf1 : memref<4x1x4x4xi32, 2 : i32> offset = 0 len = 64)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(%lock_0_2_4, Release, %c1_ul4)
      aie.next_bd ^bb4
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c1_ul5 = arith.constant 1 : i32
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, %c1_ul5)
      %c1_ul6 = arith.constant 1 : i32
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, %c1_ul6)
      // <trace>
      func.call @event_0() : () -> ()
      // </trace>
      scf.for %arg0 = %c0 to %c4 step %c1 {
        scf.for %arg1 = %c0 to %c4 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              memref.store %c0_i32, %buf0[%arg0, %arg1, %arg2, %arg3] : memref<4x4x4x4xi32, 2 : i32>
            }
          }
        }
      }
      // <trace>
      func.call @event_1() : () -> ()
      // </trace>
      func.call @matmul_scalar_put_4x1x4_4x4x4_i32_i32(%buf2, %buf1, %buf0) : (memref<1x4x4x4xi32, 2 : i32>, memref<4x1x4x4xi32, 2 : i32>, memref<4x4x4x4xi32, 2 : i32>) -> ()
      %c1_ul7 = arith.constant 1 : i32
      aie.use_lock(%lock_0_2_5, Release, %c1_ul7)
      %c1_ul8 = arith.constant 1 : i32
      aie.use_lock(%lock_0_2, Release, %c1_ul8)
      // <trace>
      func.call @flush_trace() : () -> ()
      // </trace>
      cf.br ^bb1
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c1_ul9 = arith.constant 1 : i32
      aie.use_lock(%lock_1_2_8, AcquireGreaterEqual, %c1_ul9)
      aie.dma_bd(%buf5 : memref<1x4x4x4xi32, 2 : i32> offset = 0 len = 64)
      %c1_ul10 = arith.constant 1 : i32
      aie.use_lock(%lock_1_2_9, Release, %c1_ul10)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      %c1_ul11 = arith.constant 1 : i32
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, %c1_ul11)
      aie.dma_bd(%buf4 : memref<4x1x4x4xi32, 2 : i32> offset = 0 len = 64)
      %c1_ul12 = arith.constant 1 : i32
      aie.use_lock(%lock_1_2_7, Release, %c1_ul12)
      aie.next_bd ^bb4
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c0_i32 = arith.constant 0 : i32
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c1_ul13 = arith.constant 1 : i32
      aie.use_lock(%lock_1_2_9, AcquireGreaterEqual, %c1_ul13)
      %c1_ul14 = arith.constant 1 : i32
      aie.use_lock(%lock_1_2_7, AcquireGreaterEqual, %c1_ul14)
      // <trace>
      func.call @event_0() : () -> ()
      // </trace>
      scf.for %arg0 = %c0 to %c4 step %c1 {
        scf.for %arg1 = %c0 to %c4 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              memref.store %c0_i32, %buf3[%arg0, %arg1, %arg2, %arg3] : memref<4x4x4x4xi32, 2 : i32>
            }
          }
        }
      }
      // <trace>
      func.call @event_1() : () -> ()
      // </trace>
      func.call @matmul_scalar_put_get_4x1x4_4x4x4_i32_i32(%buf5, %buf4, %buf3) : (memref<1x4x4x4xi32, 2 : i32>, memref<4x1x4x4xi32, 2 : i32>, memref<4x4x4x4xi32, 2 : i32>) -> ()
      %c1_ul15 = arith.constant 1 : i32
      aie.use_lock(%lock_1_2_8, Release, %c1_ul15)
      %c1_ul16 = arith.constant 1 : i32
      aie.use_lock(%lock_1_2, Release, %c1_ul16)
      // <trace>
      func.call @flush_trace() : () -> ()
      // </trace>
      cf.br ^bb1
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c1_ul17 = arith.constant 1 : i32
      aie.use_lock(%lock_2_2_11, AcquireGreaterEqual, %c1_ul17)
      aie.dma_bd(%buf8 : memref<1x4x4x4xi32, 2 : i32> offset = 0 len = 64)
      %c1_ul18 = arith.constant 1 : i32
      aie.use_lock(%lock_2_2_12, Release, %c1_ul18)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      %c1_ul19 = arith.constant 1 : i32
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, %c1_ul19)
      aie.dma_bd(%buf7 : memref<4x1x4x4xi32, 2 : i32> offset = 0 len = 64)
      %c1_ul20 = arith.constant 1 : i32
      aie.use_lock(%lock_2_2_10, Release, %c1_ul20)
      aie.next_bd ^bb4
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c0_i32 = arith.constant 0 : i32
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c1_ul21 = arith.constant 1 : i32
      aie.use_lock(%lock_2_2_12, AcquireGreaterEqual, %c1_ul21)
      %c1_ul22 = arith.constant 1 : i32
      aie.use_lock(%lock_2_2_10, AcquireGreaterEqual, %c1_ul22)
      // <trace>
      func.call @event_0() : () -> ()
      // </trace>
      scf.for %arg0 = %c0 to %c4 step %c1 {
        scf.for %arg1 = %c0 to %c4 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              memref.store %c0_i32, %buf6[%arg0, %arg1, %arg2, %arg3] : memref<4x4x4x4xi32, 2 : i32>
            }
          }
        }
      }
      // <trace>
      func.call @event_1() : () -> ()
      // </trace>
      func.call @matmul_scalar_put_get_4x1x4_4x4x4_i32_i32(%buf8, %buf7, %buf6) : (memref<1x4x4x4xi32, 2 : i32>, memref<4x1x4x4xi32, 2 : i32>, memref<4x4x4x4xi32, 2 : i32>) -> ()
      %c1_ul23 = arith.constant 1 : i32
      aie.use_lock(%lock_2_2_11, Release, %c1_ul23)
      %c1_ul24 = arith.constant 1 : i32
      aie.use_lock(%lock_2_2, Release, %c1_ul24)
      // <trace>
      func.call @flush_trace() : () -> ()
      // </trace>
      cf.br ^bb1
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c1_ul25 = arith.constant 1 : i32
      aie.use_lock(%lock_3_2_14, AcquireGreaterEqual, %c1_ul25)
      aie.dma_bd(%buf11 : memref<1x4x4x4xi32, 2 : i32> offset = 0 len = 64)
      %c1_ul26 = arith.constant 1 : i32
      aie.use_lock(%lock_3_2_15, Release, %c1_ul26)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      %c1_ul27 = arith.constant 1 : i32
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, %c1_ul27)
      aie.dma_bd(%buf10 : memref<4x1x4x4xi32, 2 : i32> offset = 0 len = 64)
      %c1_ul28 = arith.constant 1 : i32
      aie.use_lock(%lock_3_2_13, Release, %c1_ul28)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      %c1_ul29 = arith.constant 1 : i32
      aie.use_lock(%lock_3_2_17, AcquireGreaterEqual, %c1_ul29)
      aie.dma_bd(%buf9 : memref<4x4x4x4xi32, 2 : i32> offset = 0 len = 256 sizes = [16, 4, 4] strides = [4, 64, 1])
      %c1_ul30 = arith.constant 1 : i32
      aie.use_lock(%lock_3_2_16, Release, %c1_ul30)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c0_i32 = arith.constant 0 : i32
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c1_ul31 = arith.constant 1 : i32
      aie.use_lock(%lock_3_2_16, AcquireGreaterEqual, %c1_ul31)
      %c1_ul32 = arith.constant 1 : i32
      aie.use_lock(%lock_3_2_15, AcquireGreaterEqual, %c1_ul32)
      %c1_ul33 = arith.constant 1 : i32
      aie.use_lock(%lock_3_2_13, AcquireGreaterEqual, %c1_ul33)
      // <trace>
      func.call @event_0() : () -> ()
      // </trace>
      scf.for %arg0 = %c0 to %c4 step %c1 {
        scf.for %arg1 = %c0 to %c4 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              memref.store %c0_i32, %buf9[%arg0, %arg1, %arg2, %arg3] : memref<4x4x4x4xi32, 2 : i32>
            }
          }
        }
      }
      // <trace>
      func.call @event_1() : () -> ()
      // </trace>
      func.call @matmul_scalar_get_4x1x4_4x4x4_i32_i32(%buf11, %buf10, %buf9) : (memref<1x4x4x4xi32, 2 : i32>, memref<4x1x4x4xi32, 2 : i32>, memref<4x4x4x4xi32, 2 : i32>) -> ()
      %c1_ul34 = arith.constant 1 : i32
      aie.use_lock(%lock_3_2_17, Release, %c1_ul34)
      %c1_ul35 = arith.constant 1 : i32
      aie.use_lock(%lock_3_2_14, Release, %c1_ul35)
      %c1_ul36 = arith.constant 1 : i32
      aie.use_lock(%lock_3_2, Release, %c1_ul36)
      // <trace>
      func.call @flush_trace() : () -> ()
      // </trace>
      cf.br ^bb1
    }
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_1_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_1_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 2, %tile_2_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 3, %tile_3_2, DMA : 0)
    aie.flow(%tile_1_1, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%tile_1_1, DMA : 1, %tile_1_2, DMA : 1)
    aie.flow(%tile_1_1, DMA : 2, %tile_2_2, DMA : 1)
    aie.flow(%tile_1_1, DMA : 3, %tile_3_2, DMA : 1)
    aie.flow(%tile_3_2, DMA : 0, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_1, DMA : 4, %tile_0_0, DMA : 0)
    aie.cascade_flow(%tile_0_2, %tile_1_2)
    aie.cascade_flow(%tile_1_2, %tile_2_2)
    aie.cascade_flow(%tile_2_2, %tile_3_2)
    // <trace>
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(1) {
      aie.packet_source<%tile_1_2, Trace : 0>
      aie.packet_dest<%tile_1_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(2) {
      aie.packet_source<%tile_2_2, Trace : 0>
      aie.packet_dest<%tile_2_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(3) {
      aie.packet_source<%tile_3_2, Trace : 0>
      aie.packet_dest<%tile_3_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(4) {
      aie.packet_source<%tile_0_1, Trace : 0>
      aie.packet_dest<%tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(5) {
      aie.packet_source<%tile_1_1, Trace : 0>
      aie.packet_dest<%tile_1_0, DMA : 1>
    } {keep_pkt_header = true}
    // </trace>
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb13, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c4_ul37 = arith.constant 4 : i32
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c4_ul37)
      aie.dma_bd(%buf14 : memref<16x16xi32, 1 : i32> offset = 0 len = 256)
      %c4_ul38 = arith.constant 4 : i32
      aie.use_lock(%lock_0_1_1, Release, %c4_ul38)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      %c1_ul39 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1_2, AcquireGreaterEqual, %c1_ul39)
      aie.dma_bd(%buf12 : memref<16x16xi32, 1 : i32> offset = 0 len = 256)
      %c1_ul40 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1_3, Release, %c1_ul40)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb5
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
    ^bb6:  // 2 preds: ^bb3, ^bb4
      %c1_ul41 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, %c1_ul41)
      aie.dma_bd(%buf14 : memref<16x16xi32, 1 : i32> offset = 0 len = 64 sizes = [4, 4, 4] strides = [64, 16, 1])
      %c1_ul42 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1, Release, %c1_ul42)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb7
      %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb5, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb5, ^bb6
      %c1_ul43 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, %c1_ul43)
      aie.dma_bd(%buf14 : memref<16x16xi32, 1 : i32> offset = 4 len = 64 sizes = [4, 4, 4] strides = [64, 16, 1])
      %c1_ul44 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1, Release, %c1_ul44)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb9
      %4 = aie.dma_start(MM2S, 2, ^bb10, ^bb7, repeat_count = 1)
    ^bb10:  // 2 preds: ^bb7, ^bb8
      %c1_ul45 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, %c1_ul45)
      aie.dma_bd(%buf14 : memref<16x16xi32, 1 : i32> offset = 8 len = 64 sizes = [4, 4, 4] strides = [64, 16, 1])
      %c1_ul46 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1, Release, %c1_ul46)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb0
      %5 = aie.dma_start(MM2S, 3, ^bb12, ^bb9, repeat_count = 1)
    ^bb12:  // 2 preds: ^bb9, ^bb10
      %c1_ul47 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, %c1_ul47)
      aie.dma_bd(%buf14 : memref<16x16xi32, 1 : i32> offset = 12 len = 64 sizes = [4, 4, 4] strides = [64, 16, 1])
      %c1_ul48 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1, Release, %c1_ul48)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb0
      %6 = aie.dma_start(MM2S, 4, ^bb14, ^bb11, repeat_count = 1)
    ^bb14:  // 2 preds: ^bb3, ^bb4
      %c1_ul49 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1_3, AcquireGreaterEqual, %c1_ul49)
      aie.dma_bd(%buf12 : memref<16x16xi32, 1 : i32> offset = 0 len = 256)
      %c1_ul50 = arith.constant 1 : i32
      aie.use_lock(%lock_0_1_2, Release, %c1_ul50)
      aie.next_bd ^bb14
    }
    %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb9, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %c4_ul51 = arith.constant 4 : i32
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, %c4_ul51)
      aie.dma_bd(%buf13 : memref<16x16xi32, 1 : i32> offset = 0 len = 256)
      %c4_ul52 = arith.constant 4 : i32
      aie.use_lock(%lock_1_1_0, Release, %c4_ul52)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      %c1_ul53 = arith.constant 1 : i32
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, %c1_ul53)
      aie.dma_bd(%buf13 : memref<16x16xi32, 1 : i32> offset = 0 len = 64 sizes = [4, 4, 4] strides = [4, 16, 1])
      %c1_ul54 = arith.constant 1 : i32
      aie.use_lock(%lock_1_1, Release, %c1_ul54)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb7
      %2 = aie.dma_start(MM2S, 1, ^bb6, ^bb3, repeat_count = 1)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      %c1_ul55 = arith.constant 1 : i32
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, %c1_ul55)
      aie.dma_bd(%buf13 : memref<16x16xi32, 1 : i32> offset = 64 len = 64 sizes = [4, 4, 4] strides = [4, 16, 1])
      %c1_ul56 = arith.constant 1 : i32
      aie.use_lock(%lock_1_1, Release, %c1_ul56)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb9
      %3 = aie.dma_start(MM2S, 2, ^bb8, ^bb5, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      %c1_ul57 = arith.constant 1 : i32
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, %c1_ul57)
      aie.dma_bd(%buf13 : memref<16x16xi32, 1 : i32> offset = 128 len = 64 sizes = [4, 4, 4] strides = [4, 16, 1])
      %c1_ul58 = arith.constant 1 : i32
      aie.use_lock(%lock_1_1, Release, %c1_ul58)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb0
      %4 = aie.dma_start(MM2S, 3, ^bb10, ^bb7, repeat_count = 1)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      %c1_ul59 = arith.constant 1 : i32
      aie.use_lock(%lock_1_1_0, AcquireGreaterEqual, %c1_ul59)
      aie.dma_bd(%buf13 : memref<16x16xi32, 1 : i32> offset = 192 len = 64 sizes = [4, 4, 4] strides = [4, 16, 1])
      %c1_ul60 = arith.constant 1 : i32
      aie.use_lock(%lock_1_1, Release, %c1_ul60)
      aie.next_bd ^bb10
    }
    aie.shim_dma_allocation @airMemcpyId12 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @airMemcpyId4 (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @airMemcpyId5 (%tile_0_0, MM2S, 1)
    aie.runtime_sequence(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>, %arg2: memref<16x16xi32>) {
      // <trace>
      %cst_npu_0 = arith.constant 212992 : i32
      %cst_npu_1 = arith.constant 31232 : i32
      aiex.npu.write32(%cst_npu_0, %cst_npu_1) {column = 3 : i32, row = 2 : i32} : i32, i32 // [14:8] reset event: 122(BROADCAST_15)
      %cst_npu_2 = arith.constant 213200 : i32
      %cst_npu_3 = arith.constant 7995392 : i32
      aiex.npu.write32(%cst_npu_2, %cst_npu_3) {column = 3 : i32, row = 2 : i32} : i32, i32 // [22:16] start event: 122(BROADCAST_15)
      %cst_npu_4 = arith.constant 213204 : i32
      %cst_npu_5 = arith.constant 3 : i32
      aiex.npu.write32(%cst_npu_4, %cst_npu_5) {column = 3 : i32, row = 2 : i32} : i32, i32 // packet_type: 0(core), packet_id: 3
      %cst_npu_6 = arith.constant 213216 : i32
      %cst_npu_7 = arith.constant 1260527873 : i32
      aiex.npu.write32(%cst_npu_6, %cst_npu_7) {column = 3 : i32, row = 2 : i32} : i32, i32 // events: 0x4B(port0 run) 22(event1) 21(event0) 01(true)
      %cst_npu_8 = arith.constant 213220 : i32
      %cst_npu_9 = arith.constant 757865039 : i32
      aiex.npu.write32(%cst_npu_8, %cst_npu_9) {column = 3 : i32, row = 2 : i32} : i32, i32 // events: 0x2D(lock release) 2C(lock acquire) 1A(lock stall) 4F(port1 run)
      %cst_npu_10 = arith.constant 261888 : i32
      %cst_npu_11 = arith.constant 289 : i32
      aiex.npu.write32(%cst_npu_10, %cst_npu_11) {column = 3 : i32, row = 2 : i32} : i32, i32 // [13:8] port1 MM2S-0+1, [5:0] port0 S2MM-0+1
      %cst_npu_12 = arith.constant 261892 : i32
      %cst_npu_13 = arith.constant 0 : i32
      aiex.npu.write32(%cst_npu_12, %cst_npu_13) {column = 3 : i32, row = 2 : i32} : i32, i32
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 8192 : i32, buffer_offset = 25600 : i32, column = 3 : i32, row = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, ddr_id = 2 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 3: i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %cst_npu_14 = arith.constant 119308 : i32
      %cst_npu_15 = arith.constant 15 : i32
      aiex.npu.write32(%cst_npu_14, %cst_npu_15) {column = 3 : i32, row = 0 : i32} : i32, i32

      %cst_npu_16 = arith.constant 212992 : i32
      %cst_npu_17 = arith.constant 31232 : i32
      aiex.npu.write32(%cst_npu_16, %cst_npu_17) {column = 2 : i32, row = 2 : i32} : i32, i32 // [14:8] reset event: 122(BROADCAST_15)
      %cst_npu_18 = arith.constant 213200 : i32
      %cst_npu_19 = arith.constant 7995392 : i32
      aiex.npu.write32(%cst_npu_18, %cst_npu_19) {column = 2 : i32, row = 2 : i32} : i32, i32 // [22:16] start event: 122(BROADCAST_15)
      %cst_npu_20 = arith.constant 213204 : i32
      %cst_npu_21 = arith.constant 2 : i32
      aiex.npu.write32(%cst_npu_20, %cst_npu_21) {column = 2 : i32, row = 2 : i32} : i32, i32 // packet_type: 0(core), packet_id: 2
      %cst_npu_22 = arith.constant 213216 : i32
      %cst_npu_23 = arith.constant 1260527873 : i32
      aiex.npu.write32(%cst_npu_22, %cst_npu_23) {column = 2 : i32, row = 2 : i32} : i32, i32 // events: 0x4B(port0 run) 22(event1) 21(event0) 01(true)
      %cst_npu_24 = arith.constant 213220 : i32
      %cst_npu_25 = arith.constant 757865039 : i32
      aiex.npu.write32(%cst_npu_24, %cst_npu_25) {column = 2 : i32, row = 2 : i32} : i32, i32 // events: 0x2D(lock release) 2C(lock acquire) 1A(lock stall) 4F(port1 run)
      %cst_npu_26 = arith.constant 261888 : i32
      %cst_npu_27 = arith.constant 289 : i32
      aiex.npu.write32(%cst_npu_26, %cst_npu_27) {column = 2 : i32, row = 2 : i32} : i32, i32 // [13:8] port1 MM2S-0+1, [5:0] port0 S2MM-0+1
      %cst_npu_28 = arith.constant 261892 : i32
      %cst_npu_29 = arith.constant 0 : i32
      aiex.npu.write32(%cst_npu_28, %cst_npu_29) {column = 2 : i32, row = 2 : i32} : i32, i32
      aiex.npu.writebd {bd_id = 14 : i32, buffer_length = 8192 : i32, buffer_offset = 17408 : i32, column = 2 : i32, row = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, ddr_id = 2 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 2: i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %cst_npu_30 = arith.constant 119308 : i32
      %cst_npu_31 = arith.constant 14 : i32
      aiex.npu.write32(%cst_npu_30, %cst_npu_31) {column = 2 : i32, row = 0 : i32} : i32, i32

      %cst_npu_32 = arith.constant 212992 : i32
      %cst_npu_33 = arith.constant 31232 : i32
      aiex.npu.write32(%cst_npu_32, %cst_npu_33) {column = 1 : i32, row = 2 : i32} : i32, i32 // [14:8] reset event: 122(BROADCAST_15)
      %cst_npu_34 = arith.constant 213200 : i32
      %cst_npu_35 = arith.constant 7995392 : i32
      aiex.npu.write32(%cst_npu_34, %cst_npu_35) {column = 1 : i32, row = 2 : i32} : i32, i32 // [22:16] start event: 122(BROADCAST_15)
      %cst_npu_36 = arith.constant 213204 : i32
      %cst_npu_37 = arith.constant 1 : i32
      aiex.npu.write32(%cst_npu_36, %cst_npu_37) {column = 1 : i32, row = 2 : i32} : i32, i32 // packet_type: 0(core), packet_id: 1
      %cst_npu_38 = arith.constant 213216 : i32
      %cst_npu_39 = arith.constant 1260527873 : i32
      aiex.npu.write32(%cst_npu_38, %cst_npu_39) {column = 1 : i32, row = 2 : i32} : i32, i32 // events: 0x4B(port0 run) 22(event1) 21(event0) 01(true)
      %cst_npu_40 = arith.constant 213220 : i32
      %cst_npu_41 = arith.constant 757865039 : i32
      aiex.npu.write32(%cst_npu_40, %cst_npu_41) {column = 1 : i32, row = 2 : i32} : i32, i32 // events: 0x2D(lock release) 2C(lock acquire) 1A(lock stall) 4F(port1 run)
      %cst_npu_42 = arith.constant 261888 : i32
      %cst_npu_43 = arith.constant 289 : i32
      aiex.npu.write32(%cst_npu_42, %cst_npu_43) {column = 1 : i32, row = 2 : i32} : i32, i32 // [13:8] port1 MM2S-0+1, [5:0] port0 S2MM-0+1
      %cst_npu_44 = arith.constant 261892 : i32
      %cst_npu_45 = arith.constant 0 : i32
      aiex.npu.write32(%cst_npu_44, %cst_npu_45) {column = 1 : i32, row = 2 : i32} : i32, i32
      aiex.npu.writebd {bd_id = 13 : i32, buffer_length = 8192 : i32, buffer_offset = 9216 : i32, column = 1 : i32, row = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, ddr_id = 2 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 1: i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %cst_npu_46 = arith.constant 119308 : i32
      %cst_npu_47 = arith.constant 13 : i32
      aiex.npu.write32(%cst_npu_46, %cst_npu_47) {column = 1 : i32, row = 0 : i32} : i32, i32

      %cst_npu_48 = arith.constant 212992 : i32
      %cst_npu_49 = arith.constant 31232 : i32
      aiex.npu.write32(%cst_npu_48, %cst_npu_49) {column = 0 : i32, row = 2 : i32} : i32, i32 // [14:8] reset event: 122(BROADCAST_15)
      %cst_npu_50 = arith.constant 213200 : i32
      %cst_npu_51 = arith.constant 7995392 : i32
      aiex.npu.write32(%cst_npu_50, %cst_npu_51) {column = 0 : i32, row = 2 : i32} : i32, i32 // [22:16] start event: 122(BROADCAST_15)
      %cst_npu_52 = arith.constant 213204 : i32
      %cst_npu_53 = arith.constant 0 : i32
      aiex.npu.write32(%cst_npu_52, %cst_npu_53) {column = 0 : i32, row = 2 : i32} : i32, i32 // packet_type: 0(core), packet_id: 0
      %cst_npu_54 = arith.constant 213216 : i32
      %cst_npu_55 = arith.constant 1260527873 : i32
      aiex.npu.write32(%cst_npu_54, %cst_npu_55) {column = 0 : i32, row = 2 : i32} : i32, i32 // events: 0x4B(port0 run) 22(event1) 21(event0) 01(true)
      %cst_npu_56 = arith.constant 213220 : i32
      %cst_npu_57 = arith.constant 757865039 : i32
      aiex.npu.write32(%cst_npu_56, %cst_npu_57) {column = 0 : i32, row = 2 : i32} : i32, i32 // events: 0x2D(lock release) 2C(lock acquire) 1A(lock stall) 4F(port1 run)
      %cst_npu_58 = arith.constant 261888 : i32
      %cst_npu_59 = arith.constant 289 : i32
      aiex.npu.write32(%cst_npu_58, %cst_npu_59) {column = 0 : i32, row = 2 : i32} : i32, i32 // [13:8] port1 MM2S-0+1, [5:0] port0 S2MM-0+1
      %cst_npu_60 = arith.constant 261892 : i32
      %cst_npu_61 = arith.constant 0 : i32
      aiex.npu.write32(%cst_npu_60, %cst_npu_61) {column = 0 : i32, row = 2 : i32} : i32, i32
      aiex.npu.writebd {bd_id = 12 : i32, buffer_length = 8192 : i32, buffer_offset = 1024 : i32, column = 0 : i32, row = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, ddr_id = 2 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0: i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %cst_npu_62 = arith.constant 119308 : i32
      %cst_npu_63 = arith.constant 12 : i32
      aiex.npu.write32(%cst_npu_62, %cst_npu_63) {column = 0 : i32, row = 0 : i32} : i32, i32

      %cst_npu_64 = arith.constant 606208 : i32
      %cst_npu_65 = arith.constant 40192 : i32
      aiex.npu.write32(%cst_npu_64, %cst_npu_65) {column = 1 : i32, row = 1 : i32} : i32, i32 // [15:8] reset event: 157(BROADCAST_15)
      %cst_npu_66 = arith.constant 606416 : i32
      %cst_npu_67 = arith.constant 10289152 : i32
      aiex.npu.write32(%cst_npu_66, %cst_npu_67) {column = 1 : i32, row = 1 : i32} : i32, i32 // [23:16] start event: 157(BROADCAST_15)
      %cst_npu_68 = arith.constant 606420 : i32
      %cst_npu_69 = arith.constant 12293 : i32
      aiex.npu.write32(%cst_npu_68, %cst_npu_69) {column = 1 : i32, row = 1 : i32} : i32, i32 // [14:12] packet_type: 3(mem_tile), [4:0] packet_id: 5
      %cst_npu_70 = arith.constant 606432 : i32
      %cst_npu_71 = arith.constant 336 : i32
      aiex.npu.write32(%cst_npu_70, %cst_npu_71) {column = 1 : i32, row = 1 : i32} : i32, i32 // events: 0x00 00 01(true) 50(port0 run)
      %cst_npu_72 = arith.constant 606436 : i32
      %cst_npu_73 = arith.constant 1415076960 : i32
      aiex.npu.write32(%cst_npu_72, %cst_npu_73) {column = 1 : i32, row = 1 : i32} : i32, i32 // events: 0x54(port1 run) 58(port2 run) 5C(port3 run) 60(port4 run)
      %cst_npu_74 = arith.constant 724736 : i32
      %cst_npu_75 = arith.constant 33620000 : i32
      aiex.npu.write32(%cst_npu_74, %cst_npu_75) {column = 1 : i32, row = 1 : i32} : i32, i32 // [29:24] port3 MM2S-2, [21:16] port2 MM2S-1, [13:8] port1 MM2S-0, [5:0] port0 S2MM-0
      %cst_npu_76 = arith.constant 724740 : i32
      %cst_npu_77 = arith.constant 3 : i32
      aiex.npu.write32(%cst_npu_76, %cst_npu_77) {column = 1 : i32, row = 1 : i32} : i32, i32 // [5:0] port4 MM2S-3
      aiex.npu.writebd {bd_id = 11 : i32, buffer_length = 8192 : i32, buffer_offset = 9216 : i32, column = 1 : i32, row = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, ddr_id = 2 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 5: i32, packet_type = 3 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %cst_npu_78 = arith.constant 119308 : i32
      %cst_npu_79 = arith.constant 11 : i32
      aiex.npu.write32(%cst_npu_78, %cst_npu_79) {column = 1 : i32, row = 0 : i32} : i32, i32

      %cst_npu_80 = arith.constant 606208 : i32
      %cst_npu_81 = arith.constant 40192 : i32
      aiex.npu.write32(%cst_npu_80, %cst_npu_81) {column = 0 : i32, row = 1 : i32} : i32, i32 // [15:8] reset event: 157(BROADCAST_15)
      %cst_npu_82 = arith.constant 606416 : i32
      %cst_npu_83 = arith.constant 10289152 : i32
      aiex.npu.write32(%cst_npu_82, %cst_npu_83) {column = 0 : i32, row = 1 : i32} : i32, i32 // [23:16] start event: 157(BROADCAST_15)
      %cst_npu_84 = arith.constant 606420 : i32
      %cst_npu_85 = arith.constant 12292 : i32
      aiex.npu.write32(%cst_npu_84, %cst_npu_85) {column = 0 : i32, row = 1 : i32} : i32, i32 // [14:12] packet_type: 3(mem_tile), [4:0] packet_id: 4
      %cst_npu_86 = arith.constant 606432 : i32
      %cst_npu_87 = arith.constant 760239192 : i32
      aiex.npu.write32(%cst_npu_86, %cst_npu_87) {column = 0 : i32, row = 1 : i32} : i32, i32 // events: 0x2D(lock release) 50(port0 run) 0x54(port1 run) 58(port2 run)
      %cst_npu_88 = arith.constant 606436 : i32
      %cst_npu_89 = arith.constant 1549821032 : i32
      aiex.npu.write32(%cst_npu_88, %cst_npu_89) {column = 0 : i32, row = 1 : i32} : i32, i32 // events: 5C(port3 run) 60(port4 run) 64(port5 run) 68(port6 run)
      %cst_npu_90 = arith.constant 724736 : i32
      %cst_npu_91 = arith.constant 33620000 : i32
      aiex.npu.write32(%cst_npu_90, %cst_npu_91) {column = 0 : i32, row = 1 : i32} : i32, i32 // [29:24] port3 MM2S-2, [21:16] port2 MM2S-1, [13:8] port1 MM2S-0, [5:0] port0 S2MM-0
      %cst_npu_92 = arith.constant 724740 : i32
      %cst_npu_93 = arith.constant 270595 : i32
      aiex.npu.write32(%cst_npu_92, %cst_npu_93) {column = 0 : i32, row = 1 : i32} : i32, i32 // [21:16] port6 MM2S-4, [13:8] port5 S2MM-1, [5:0] port4 MM2S-3
      aiex.npu.writebd {bd_id = 10 : i32, buffer_length = 8192 : i32, buffer_offset = 1024 : i32, column = 0 : i32, row = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, ddr_id = 2 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 4: i32, packet_type = 3 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %cst_npu_94 = arith.constant 119308 : i32
      %cst_npu_95 = arith.constant 10 : i32
      aiex.npu.write32(%cst_npu_94, %cst_npu_95) {column = 0 : i32, row = 0 : i32} : i32, i32

      %cst_npu_96 = arith.constant 212992 : i32
      %cst_npu_97 = arith.constant 32512 : i32
      aiex.npu.write32(%cst_npu_96, %cst_npu_97) {column = 0 : i32, row = 0 : i32} : i32, i32 // [14:8] reset event: 127(USER_EVENT_1)
      %cst_npu_98 = arith.constant 213068 : i32
      %cst_npu_99 = arith.constant 127 : i32
      aiex.npu.write32(%cst_npu_98, %cst_npu_99) {column = 0 : i32, row = 0 : i32} : i32, i32 // [6:0] broadcast 15: 127(USER_EVENT_1)
      %cst_npu_100 = arith.constant 213000 : i32
      %cst_npu_101 = arith.constant 127 : i32
      aiex.npu.write32(%cst_npu_100, %cst_npu_101) {column = 0 : i32, row = 0 : i32} : i32, i32 // event generate [6:0]: 127(USER_EVENT_1)

      // </trace>
      memref.assume_alignment %arg0, 64 : memref<16x16xi32>
      memref.assume_alignment %arg1, 64 : memref<16x16xi32>
      memref.assume_alignment %arg2, 64 : memref<16x16xi32>
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 16, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<16x16xi32>
      aiex.npu.dma_memcpy_nd (%arg1[0, 0, 0, 0][1, 1, 16, 16][0, 0, 16, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<16x16xi32>
      aiex.npu.dma_memcpy_nd (%arg2[0, 0, 0, 0][1, 1, 16, 16][0, 0, 16, 1]) {id = 2 : i64, metadata = @airMemcpyId12, issue_token = true} : memref<16x16xi32>
      aiex.npu.dma_wait {symbol = @airMemcpyId12}
    }
  } {sym_name = "segment_0"}
}
