//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_4col) {
    memref.global "public" @ctrlin0 : memref<8xi32>
    memref.global "public" @ctrlin1 : memref<8xi32>
    memref.global "public" @out0 : memref<8xi32>
    memref.global "public" @out1 : memref<8xi32>
    memref.global "public" @out2 : memref<8xi32>
    memref.global "public" @out3 : memref<8xi32>
    memref.global "public" @ctrl0 : memref<8xi32>

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
      aie.dma_bd(%output_0_2_buffer : memref<8xi32>, 0, 8)
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
      aie.dma_bd(%output_0_3_buffer : memref<8xi32>, 0, 8)
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
      aie.dma_bd(%output_0_4_buffer : memref<8xi32>, 0, 8)
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
      aie.dma_bd(%output_0_5_buffer : memref<8xi32>, 0, 8)
      aie.use_lock(%output_0_5_lock5, Release, 1)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    aie.shim_dma_allocation @ctrlin0(MM2S, 0, 0)
    aie.shim_dma_allocation @ctrlin1(MM2S, 1, 0)
    aie.shim_dma_allocation @ctrl0(S2MM, 0, 0)
    aie.shim_dma_allocation @out0(S2MM, 1, 0)
    aie.shim_dma_allocation @out1(S2MM, 0, 1)
    aie.shim_dma_allocation @out2(S2MM, 1, 1)
    aie.shim_dma_allocation @out3(S2MM, 0, 2)

    aiex.runtime_sequence @seq(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %arg2: memref<32xi32>) {
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
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 2 : i64, issue_token = true, metadata = @out0} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[%c0_i64, %c0_i64, %c0_i64, %c8_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 3 : i64, issue_token = true, metadata = @out1} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[%c0_i64, %c0_i64, %c0_i64, %c16_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 4 : i64, issue_token = true, metadata = @out2} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[%c0_i64, %c0_i64, %c0_i64, %c24_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 5 : i64, issue_token = true, metadata = @out3} : memref<32xi32>

      // write bd0
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 27, pkt_type = 1>) {id = 6 : i64, issue_token = true, metadata = @ctrlin0} : memref<8xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[%c0_i64, %c0_i64, %c0_i64, %c4_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 29, pkt_type = 1>) {id = 7 : i64, issue_token = true, metadata = @ctrlin1} : memref<8xi32>
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[%c0_i64, %c0_i64, %c0_i64, %c8_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 30, pkt_type = 1>) {id = 8 : i64, issue_token = true, metadata = @ctrlin1} : memref<8xi32>
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[%c0_i64, %c0_i64, %c0_i64, %c12_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 31, pkt_type = 1>) {id = 9 : i64, issue_token = true, metadata = @ctrlin1} : memref<8xi32>
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // patch bd0 address for packet 1, push to mm2s_0_task_queue, wait
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[%c0_i64, %c0_i64, %c0_i64, %c2_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 27, pkt_type = 1>) {id = 6 : i64, issue_token = true, metadata = @ctrlin0} : memref<8xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[%c0_i64, %c0_i64, %c0_i64, %c6_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 29, pkt_type = 1>) {id = 7 : i64, issue_token = true, metadata = @ctrlin1} : memref<8xi32>
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[%c0_i64, %c0_i64, %c0_i64, %c10_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 30, pkt_type = 1>) {id = 8 : i64, issue_token = true, metadata = @ctrlin1} : memref<8xi32>
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[%c0_i64, %c0_i64, %c0_i64, %c14_i64] [%c1_i64, %c1_i64, %c1_i64, %c2_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64], packet = <pkt_id = 31, pkt_type = 1>) {id = 9 : i64, issue_token = true, metadata = @ctrlin1} : memref<8xi32>
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // wait for dma output
      aiex.npu.dma_wait {symbol = @out0}
      aiex.npu.dma_wait {symbol = @out1}
      aiex.npu.dma_wait {symbol = @out2}
      aiex.npu.dma_wait {symbol = @out3}
    }
  }
}
