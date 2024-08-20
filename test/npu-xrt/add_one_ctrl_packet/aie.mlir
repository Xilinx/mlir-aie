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
  aie.device(npu1_1col) {
    memref.global "public" @out0 : memref<8xi32>
    memref.global "public" @ctrl0 : memref<8xi32>

    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}

    %input_lock0 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "input_lock0"}
    %input_lock2 = aie.lock(%tile_0_2, 2) {init = 0 : i32, sym_name = "input_lock2"}
    %output_lock4 = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "output_lock4"}
    %output_lock5 = aie.lock(%tile_0_2, 5) {init = 1 : i32, sym_name = "output_lock5"}

    %input_buffer = aie.buffer(%tile_0_2) {sym_name = "input_buffer"} : memref<8xi32>
    %output_buffer = aie.buffer(%tile_0_2) {sym_name = "output_buffer"} : memref<8xi32>
    %other_buffer = aie.buffer(%tile_0_2) {sym_name = "other_buffer"} : memref<8xi32>

    aie.packet_flow(0x1) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, Ctrl : 0>
    }
    aie.packet_flow(0x2) {
      aie.packet_source<%tile_0_2, Ctrl : 0>
      aie.packet_dest<%tile_0_0, DMA : 0>
    }
    aie.packet_flow(0x3) {
      aie.packet_source<%tile_0_2, DMA : 0>
      aie.packet_dest<%tile_0_0, DMA : 1>
    }

    aie.flow(%tile_0_0, DMA : 1, %tile_0_2, DMA : 1)

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
        memref.store %1, %input_buffer[%arg1] : memref<8xi32>
        memref.store %c1_i32, %other_buffer[%arg1] : memref<8xi32>
      }
      %c4294967295 = arith.constant 4294967295 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        aie.use_lock(%input_lock0, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 4
          %1 = memref.load %input_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_lock0, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 5
          %1 = memref.load %input_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_lock2, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 6
          %1 = memref.load %input_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%input_lock2, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
          // 7
          %1 = memref.load %input_buffer[%arg1] : memref<8xi32>
          %2 = arith.addi %1, %c1_i32 : i32
          memref.store %2, %input_buffer[%arg1] : memref<8xi32>
        }
        // write to output buffer
        aie.use_lock(%output_lock5, AcquireGreaterEqual, 1)
        scf.for %arg1 = %c0 to %c8 step %c1 {
            %1 = memref.load %input_buffer[%arg1] : memref<8xi32>
            memref.store %1, %output_buffer[%arg1] : memref<8xi32>
            %2 = arith.addi %1, %c1_i32 : i32
            memref.store %2, %other_buffer[%arg1] : memref<8xi32>
        }
        aie.use_lock(%output_lock4, Release, 1)
      }
      aie.end
    }

    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%output_lock4, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_buffer : memref<8xi32>, 0, 8) {packet = #aie.packet_info<pkt_id = 3, pkt_type = 0>}
      aie.use_lock(%output_lock5, Release, 1)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    aie.shim_dma_allocation @ctrl0(S2MM, 0, 0)
    aie.shim_dma_allocation @out0(S2MM, 1, 0)

    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[2, 0, 0x40090000, 0, 0x40000000, 0, 0, 0x2000000]>
    aiex.runtime_sequence @seq(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %arg2: memref<8xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %c8_i64 = arith.constant 8 : i64

      // set Ctrl_Pkt_Tlast_Error_Enable=0 in Module_Clock_Control register
      // aiex.npu.maskwrite32 {address = 0x00060000 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32, mask = 0x8 : ui32}

      // start reading output
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @ctrl0} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 2 : i64, issue_token = true, metadata = @out0} : memref<8xi32>

      // write bd0
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 0x1d000 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>

      // patch bd0 address for packet 0, push to mm2s_0_task_queue, wait
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 0x1d210 : ui32, column = 0 : i32, row = 0 : i32, mask = 0x00000F00 : ui32, value = 0x400 : ui32}
      aiex.npu.write32 {address = 0x1d214 : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // patch bd0 address for packet 1, push to mm2s_0_task_queue, wait
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 8 : i32}
      aiex.npu.maskwrite32 {address = 0x1d210 : ui32, column = 0 : i32, row = 0 : i32, mask = 0x00000F00 : ui32, value = 0x400 : ui32}
      aiex.npu.write32 {address = 0x1d214 : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // wait for dma output
      aiex.npu.dma_wait {symbol = @out0}

      // patch bd0 length and address for packet 2, push to mm2s_0_task_queue, wait
      aiex.npu.write32 {address = 0x1d000 : ui32, column = 0 : i32, row = 0 : i32, value = 1 : ui32}
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 16 : i32}
      aiex.npu.write32 {address = 0x1d214 : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // patch bd0 address for packet 3, push to mm2s_0_task_queue, wait
      aiex.npu.address_patch {addr = 0x1d004 : ui32, arg_idx = 1 : i32, arg_plus = 20 : i32}
      aiex.npu.write32 {address = 0x1d214 : ui32, column = 0 : i32, row = 0 : i32, value = 0x80000000 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

      // wait for control port output
      aiex.npu.dma_wait {symbol = @ctrl0}
    }
  }
}
