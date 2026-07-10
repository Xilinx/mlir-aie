//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)

    %input_0_2_lock0 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "input_0_2_lock0"}
    %input_0_2_lock2 = aie.lock(%tile_0_2, 2) {init = 0 : i32, sym_name = "input_0_2_lock2"}
    %output_0_2_lock4 = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "output_0_2_lock4"}
    %output_0_2_lock5 = aie.lock(%tile_0_2, 5) {init = 1 : i32, sym_name = "output_0_2_lock5"}

    %input_0_2_buffer = aie.buffer(%tile_0_2) {sym_name = "input_0_2_buffer"} : memref<8xi32>
    %output_0_2_buffer = aie.buffer(%tile_0_2) {sym_name = "output_0_2_buffer"} : memref<8xi32>

    %input_0_3_lock0 = aie.lock(%tile_0_3, 0) {init = 0 : i32, sym_name = "input_0_3_lock0"}
    %input_0_3_lock2 = aie.lock(%tile_0_3, 2) {init = 0 : i32, sym_name = "input_0_3_lock2"}
    %output_0_3_lock4 = aie.lock(%tile_0_3, 4) {init = 0 : i32, sym_name = "output_0_3_lock4"}
    %output_0_3_lock5 = aie.lock(%tile_0_3, 5) {init = 1 : i32, sym_name = "output_0_3_lock5"}

    %input_0_3_buffer = aie.buffer(%tile_0_3) {sym_name = "input_0_3_buffer"} : memref<8xi32>
    %output_0_3_buffer = aie.buffer(%tile_0_3) {sym_name = "output_0_3_buffer"} : memref<8xi32>

    %input_0_4_lock0 = aie.lock(%tile_0_4, 0) {init = 0 : i32, sym_name = "input_0_4_lock0"}
    %input_0_4_lock2 = aie.lock(%tile_0_4, 2) {init = 0 : i32, sym_name = "input_0_4_lock2"}
    %output_0_4_lock4 = aie.lock(%tile_0_4, 4) {init = 0 : i32, sym_name = "output_0_4_lock4"}
    %output_0_4_lock5 = aie.lock(%tile_0_4, 5) {init = 1 : i32, sym_name = "output_0_4_lock5"}

    %input_0_4_buffer = aie.buffer(%tile_0_4) {sym_name = "input_0_4_buffer"} : memref<8xi32>
    %output_0_4_buffer = aie.buffer(%tile_0_4) {sym_name = "output_0_4_buffer"} : memref<8xi32>

    %input_0_5_lock0 = aie.lock(%tile_0_5, 0) {init = 0 : i32, sym_name = "input_0_5_lock0"}
    %input_0_5_lock2 = aie.lock(%tile_0_5, 2) {init = 0 : i32, sym_name = "input_0_5_lock2"}
    %output_0_5_lock4 = aie.lock(%tile_0_5, 4) {init = 0 : i32, sym_name = "output_0_5_lock4"}
    %output_0_5_lock5 = aie.lock(%tile_0_5, 5) {init = 1 : i32, sym_name = "output_0_5_lock5"}

    %input_0_5_buffer = aie.buffer(%tile_0_5) {sym_name = "input_0_5_buffer"} : memref<8xi32>
    %output_0_5_buffer = aie.buffer(%tile_0_5) {sym_name = "output_0_5_buffer"} : memref<8xi32>

    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 1)
    aie.flow(%tile_0_3, DMA : 0, %tile_1_0, DMA : 0)
    aie.flow(%tile_0_4, DMA : 0, %tile_1_0, DMA : 1)
    aie.flow(%tile_0_5, DMA : 0, %tile_2_0, DMA : 0)

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c3_i32 = arith.constant 3 : i32
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      // initialize to i + 3
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %arg1_i32 = arith.index_cast %arg1 : index to i32
        %1 = arith.addi %arg1_i32, %c3_i32 : i32
        memref.store %1, %input_0_2_buffer[%arg1] : memref<8xi32>
      }
      %c4294967295 = arith.constant 4294967295 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%input_0_2_lock0, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 4
          %1 = memref.load %input_0_2_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_2_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_0_2_lock0, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 5
          %1 = memref.load %input_0_2_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_2_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_0_2_lock2, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 6
          %1 = memref.load %input_0_2_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_2_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_0_2_lock2, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 7
          %1 = memref.load %input_0_2_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_2_buffer[%arg1] : memref<8xi32>
        }
        // write to output buffer
        aie.use_lock(%output_0_2_lock5, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
            %1 = memref.load %input_0_2_buffer[%arg1] : memref<8xi32>
            memref.store %1, %output_0_2_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%output_0_2_lock4, Release, 1)
      }
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%output_0_2_lock4, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_0_2_buffer : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%output_0_2_lock5, Release, 1)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c3_i32 = arith.constant 3 : i32
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      // initialize to i + 3
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %arg1_i32 = arith.index_cast %arg1 : index to i32
        %1 = arith.addi %arg1_i32, %c3_i32 : i32
        memref.store %1, %input_0_3_buffer[%arg1] : memref<8xi32>
      }
      %c4294967295 = arith.constant 4294967295 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%input_0_3_lock0, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 4
          %1 = memref.load %input_0_3_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_3_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_0_3_lock0, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 5
          %1 = memref.load %input_0_3_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_3_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_0_3_lock2, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 6
          %1 = memref.load %input_0_3_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_3_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_0_3_lock2, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 7
          %1 = memref.load %input_0_3_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_3_buffer[%arg1] : memref<8xi32>
        }
        // write to output buffer
        aie.use_lock(%output_0_3_lock5, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
            %1 = memref.load %input_0_3_buffer[%arg1] : memref<8xi32>
            memref.store %1, %output_0_3_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%output_0_3_lock4, Release, 1)
      }
      aie.end
    }

    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%output_0_3_lock4, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_0_3_buffer : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%output_0_3_lock5, Release, 1)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c3_i32 = arith.constant 3 : i32
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      // initialize to i + 3
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %arg1_i32 = arith.index_cast %arg1 : index to i32
        %1 = arith.addi %arg1_i32, %c3_i32 : i32
        memref.store %1, %input_0_4_buffer[%arg1] : memref<8xi32>
      }
      %c4294967295 = arith.constant 4294967295 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%input_0_4_lock0, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 4
          %1 = memref.load %input_0_4_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_4_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_0_4_lock0, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 5
          %1 = memref.load %input_0_4_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_4_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_0_4_lock2, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 6
          %1 = memref.load %input_0_4_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_4_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_0_4_lock2, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 7
          %1 = memref.load %input_0_4_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_4_buffer[%arg1] : memref<8xi32>
        }
        // write to output buffer
        aie.use_lock(%output_0_4_lock5, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
            %1 = memref.load %input_0_4_buffer[%arg1] : memref<8xi32>
            memref.store %1, %output_0_4_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%output_0_4_lock4, Release, 1)
      }
      aie.end
    }

    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%output_0_4_lock4, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_0_4_buffer : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%output_0_4_lock5, Release, 1)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c3_i32 = arith.constant 3 : i32
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      // initialize to i + 3
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %arg1_i32 = arith.index_cast %arg1 : index to i32
        %1 = arith.addi %arg1_i32, %c3_i32 : i32
        memref.store %1, %input_0_5_buffer[%arg1] : memref<8xi32>
      }
      %c4294967295 = arith.constant 4294967295 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%input_0_5_lock0, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 4
          %1 = memref.load %input_0_5_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_5_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_0_5_lock0, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 5
          %1 = memref.load %input_0_5_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_5_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_0_5_lock2, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 6
          %1 = memref.load %input_0_5_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_5_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_0_5_lock2, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 7
          %1 = memref.load %input_0_5_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_0_5_buffer[%arg1] : memref<8xi32>
        }
        // write to output buffer
        aie.use_lock(%output_0_5_lock5, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
            %1 = memref.load %input_0_5_buffer[%arg1] : memref<8xi32>
            memref.store %1, %output_0_5_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%output_0_5_lock4, Release, 1)
      }
      aie.end
    }

    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%output_0_5_lock4, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_0_5_buffer : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%output_0_5_lock5, Release, 1)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    aie.shim_dma_allocation @ctrlin0 (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @ctrlin1 (%tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @ctrl0 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @out1 (%tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @out2 (%tile_1_0, S2MM, 1)
    aie.shim_dma_allocation @out3 (%tile_2_0, S2MM, 0)

    aie.runtime_sequence @seq(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %arg2: memref<32xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %c4_i64 = arith.constant 4 : i64
      %c6_i64 = arith.constant 6 : i64
      %c8_i64 = arith.constant 8 : i64
      %c10_i64 = arith.constant 10 : i64
      %c12_i64 = arith.constant 12 : i64
      %c14_i64 = arith.constant 14 : i64
      %c16_i64 = arith.constant 16 : i64
      %c24_i64 = arith.constant 24 : i64

      // start reading output
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 2 : i64, issue_token = true, metadata = @out0} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c8_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 3 : i64, issue_token = true, metadata = @out1} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c16_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 4 : i64, issue_token = true, metadata = @out2} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c24_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 5 : i64, issue_token = true, metadata = @out3} : memref<32xi32>

      // write bd0
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 27, pkt_type = 1>) {id = 6 : i64, issue_token = true, metadata = @ctrlin0} : memref<8xi32>
      %cst_npu_0 = arith.constant 0 : i32
      %cst_npu_1 = arith.constant 0 : i32
      %cst_npu_2 = arith.constant 1 : i32
      %cst_npu_3 = arith.constant 0 : i32
      %cst_npu_4 = arith.constant 1 : i32
      %cst_npu_5 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_0, %cst_npu_1, %cst_npu_2, %cst_npu_3, %cst_npu_4, %cst_npu_5) : i32, i32, i32, i32, i32, i32
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c4_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 29, pkt_type = 1>) {id = 7 : i64, issue_token = true, metadata = @ctrlin1} : memref<8xi32>
      %cst_npu_6 = arith.constant 0 : i32
      %cst_npu_7 = arith.constant 0 : i32
      %cst_npu_8 = arith.constant 1 : i32
      %cst_npu_9 = arith.constant 1 : i32
      %cst_npu_10 = arith.constant 1 : i32
      %cst_npu_11 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_6, %cst_npu_7, %cst_npu_8, %cst_npu_9, %cst_npu_10, %cst_npu_11) : i32, i32, i32, i32, i32, i32
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c8_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 30, pkt_type = 1>) {id = 8 : i64, issue_token = true, metadata = @ctrlin1} : memref<8xi32>
      %cst_npu_12 = arith.constant 0 : i32
      %cst_npu_13 = arith.constant 0 : i32
      %cst_npu_14 = arith.constant 1 : i32
      %cst_npu_15 = arith.constant 1 : i32
      %cst_npu_16 = arith.constant 1 : i32
      %cst_npu_17 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_12, %cst_npu_13, %cst_npu_14, %cst_npu_15, %cst_npu_16, %cst_npu_17) : i32, i32, i32, i32, i32, i32
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c12_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 31, pkt_type = 1>) {id = 9 : i64, issue_token = true, metadata = @ctrlin1} : memref<8xi32>
      %cst_npu_18 = arith.constant 0 : i32
      %cst_npu_19 = arith.constant 0 : i32
      %cst_npu_20 = arith.constant 1 : i32
      %cst_npu_21 = arith.constant 1 : i32
      %cst_npu_22 = arith.constant 1 : i32
      %cst_npu_23 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_18, %cst_npu_19, %cst_npu_20, %cst_npu_21, %cst_npu_22, %cst_npu_23) : i32, i32, i32, i32, i32, i32

      // patch bd0 address for packet 1, push to mm2s_0_task_queue, wait
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c2_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 27, pkt_type = 1>) {id = 6 : i64, issue_token = true, metadata = @ctrlin0} : memref<8xi32>
      %cst_npu_24 = arith.constant 0 : i32
      %cst_npu_25 = arith.constant 0 : i32
      %cst_npu_26 = arith.constant 1 : i32
      %cst_npu_27 = arith.constant 0 : i32
      %cst_npu_28 = arith.constant 1 : i32
      %cst_npu_29 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_24, %cst_npu_25, %cst_npu_26, %cst_npu_27, %cst_npu_28, %cst_npu_29) : i32, i32, i32, i32, i32, i32
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c6_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 29, pkt_type = 1>) {id = 7 : i64, issue_token = true, metadata = @ctrlin1} : memref<8xi32>
      %cst_npu_30 = arith.constant 0 : i32
      %cst_npu_31 = arith.constant 0 : i32
      %cst_npu_32 = arith.constant 1 : i32
      %cst_npu_33 = arith.constant 1 : i32
      %cst_npu_34 = arith.constant 1 : i32
      %cst_npu_35 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_30, %cst_npu_31, %cst_npu_32, %cst_npu_33, %cst_npu_34, %cst_npu_35) : i32, i32, i32, i32, i32, i32
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c10_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 30, pkt_type = 1>) {id = 8 : i64, issue_token = true, metadata = @ctrlin1} : memref<8xi32>
      %cst_npu_36 = arith.constant 0 : i32
      %cst_npu_37 = arith.constant 0 : i32
      %cst_npu_38 = arith.constant 1 : i32
      %cst_npu_39 = arith.constant 1 : i32
      %cst_npu_40 = arith.constant 1 : i32
      %cst_npu_41 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_36, %cst_npu_37, %cst_npu_38, %cst_npu_39, %cst_npu_40, %cst_npu_41) : i32, i32, i32, i32, i32, i32
      aiex.npu.dma_memcpy_nd(%arg1[%c0_i64, %c0_i64, %c0_i64, %c14_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 31, pkt_type = 1>) {id = 9 : i64, issue_token = true, metadata = @ctrlin1} : memref<8xi32>
      %cst_npu_42 = arith.constant 0 : i32
      %cst_npu_43 = arith.constant 0 : i32
      %cst_npu_44 = arith.constant 1 : i32
      %cst_npu_45 = arith.constant 1 : i32
      %cst_npu_46 = arith.constant 1 : i32
      %cst_npu_47 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_42, %cst_npu_43, %cst_npu_44, %cst_npu_45, %cst_npu_46, %cst_npu_47) : i32, i32, i32, i32, i32, i32

      // wait for dma output
      aiex.npu.dma_wait {symbol = @out0}
      aiex.npu.dma_wait {symbol = @out1}
      aiex.npu.dma_wait {symbol = @out2}
      aiex.npu.dma_wait {symbol = @out3}
    }
  }
}