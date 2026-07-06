//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %tile_0_2 = aie.tile(0, 2)

    %input_lock0 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "input_lock0"}
    %input_lock2 = aie.lock(%tile_0_2, 2) {init = 0 : i32, sym_name = "input_lock2"}
    %output_lock4 = aie.lock(%tile_0_2, 4) {init = 0 : i32, sym_name = "output_lock4"}
    %output_lock5 = aie.lock(%tile_0_2, 5) {init = 1 : i32, sym_name = "output_lock5"}

    %input_buffer = aie.buffer(%tile_0_2) {sym_name = "input_buffer"} : memref<8xi32>
    %output_buffer = aie.buffer(%tile_0_2) {sym_name = "output_buffer"} : memref<8xi32>
    %other_buffer = aie.buffer(%tile_0_2) {sym_name = "other_buffer"} : memref<8xi32>

    aie.packet_flow(0x1) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, TileControl : 0>
    }
    aie.packet_flow(0x2) {
      aie.packet_source<%tile_0_2, TileControl : 0>
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

    aie.shim_dma_allocation @ctrl0 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 1)

    memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[2, 0, 0x40090000, 0, 0x40000000, 0, 0, 0x2000000]>
    aie.runtime_sequence @seq(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %arg2: memref<8xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %c8_i64 = arith.constant 8 : i64

      // set Ctrl_Pkt_Tlast_Error_Enable=0 in Module_Clock_Control register
      // aiex.npu.maskwrite32 {address = 0x00060000 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32, mask = 0x8 : ui32}

      // start reading output
      aiex.npu.dma_memcpy_nd(%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, issue_token = true, metadata = @ctrl0} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(%arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c8_i64] [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 2 : i64, issue_token = true, metadata = @out0} : memref<8xi32>

      // write bd0
      %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
      aiex.npu.blockwrite(%0) {address = 0x1d000 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>

      // patch bd0 address for packet 0, push to mm2s_0_task_queue, wait
      %cst_npu_0 = arith.constant 0 : i32
      aiex.npu.address_patch(%cst_npu_0 : i32) {addr = 0x1d004 : ui32, arg_idx = 1 : i32}
      %cst_npu_1 = arith.constant 0x1d210 : i32
      %cst_npu_2 = arith.constant 0x400 : i32
      %cst_npu_3 = arith.constant 0x00000F00 : i32
      aiex.npu.maskwrite32(%cst_npu_1, %cst_npu_2, %cst_npu_3) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %cst_npu_4 = arith.constant 0x1d214 : i32
      %cst_npu_5 = arith.constant 0x80000000 : i32
      aiex.npu.write32(%cst_npu_4, %cst_npu_5) {column = 0 : i32, row = 0 : i32} : i32, i32
      %cst_npu_6 = arith.constant 0 : i32
      %cst_npu_7 = arith.constant 0 : i32
      %cst_npu_8 = arith.constant 1 : i32
      %cst_npu_9 = arith.constant 0 : i32
      %cst_npu_10 = arith.constant 1 : i32
      %cst_npu_11 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_6, %cst_npu_7, %cst_npu_8, %cst_npu_9, %cst_npu_10, %cst_npu_11) : i32, i32, i32, i32, i32, i32

      // patch bd0 address for packet 1, push to mm2s_0_task_queue, wait
      %cst_npu_12 = arith.constant 8 : i32
      aiex.npu.address_patch(%cst_npu_12 : i32) {addr = 0x1d004 : ui32, arg_idx = 1 : i32}
      %cst_npu_13 = arith.constant 0x1d214 : i32
      %cst_npu_14 = arith.constant 0x80000000 : i32
      aiex.npu.write32(%cst_npu_13, %cst_npu_14) {column = 0 : i32, row = 0 : i32} : i32, i32
      %cst_npu_15 = arith.constant 0 : i32
      %cst_npu_16 = arith.constant 0 : i32
      %cst_npu_17 = arith.constant 1 : i32
      %cst_npu_18 = arith.constant 0 : i32
      %cst_npu_19 = arith.constant 1 : i32
      %cst_npu_20 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_15, %cst_npu_16, %cst_npu_17, %cst_npu_18, %cst_npu_19, %cst_npu_20) : i32, i32, i32, i32, i32, i32

      // wait for dma output
      aiex.npu.dma_wait {symbol = @out0}

      // patch bd0 length and address for packet 2, push to mm2s_0_task_queue, wait
      %cst_npu_21 = arith.constant 0x1d000 : i32
      %cst_npu_22 = arith.constant 1 : i32
      aiex.npu.write32(%cst_npu_21, %cst_npu_22) {column = 0 : i32, row = 0 : i32} : i32, i32
      %cst_npu_23 = arith.constant 16 : i32
      aiex.npu.address_patch(%cst_npu_23 : i32) {addr = 0x1d004 : ui32, arg_idx = 1 : i32}
      %cst_npu_24 = arith.constant 0x1d214 : i32
      %cst_npu_25 = arith.constant 0x80000000 : i32
      aiex.npu.write32(%cst_npu_24, %cst_npu_25) {column = 0 : i32, row = 0 : i32} : i32, i32
      %cst_npu_26 = arith.constant 0 : i32
      %cst_npu_27 = arith.constant 0 : i32
      %cst_npu_28 = arith.constant 1 : i32
      %cst_npu_29 = arith.constant 0 : i32
      %cst_npu_30 = arith.constant 1 : i32
      %cst_npu_31 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_26, %cst_npu_27, %cst_npu_28, %cst_npu_29, %cst_npu_30, %cst_npu_31) : i32, i32, i32, i32, i32, i32

      // patch bd0 address for packet 3, push to mm2s_0_task_queue, wait
      %cst_npu_32 = arith.constant 20 : i32
      aiex.npu.address_patch(%cst_npu_32 : i32) {addr = 0x1d004 : ui32, arg_idx = 1 : i32}
      %cst_npu_33 = arith.constant 0x1d214 : i32
      %cst_npu_34 = arith.constant 0x80000000 : i32
      aiex.npu.write32(%cst_npu_33, %cst_npu_34) {column = 0 : i32, row = 0 : i32} : i32, i32
      %cst_npu_35 = arith.constant 0 : i32
      %cst_npu_36 = arith.constant 0 : i32
      %cst_npu_37 = arith.constant 1 : i32
      %cst_npu_38 = arith.constant 0 : i32
      %cst_npu_39 = arith.constant 1 : i32
      %cst_npu_40 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_35, %cst_npu_36, %cst_npu_37, %cst_npu_38, %cst_npu_39, %cst_npu_40) : i32, i32, i32, i32, i32, i32

      // wait for control port output
      aiex.npu.dma_wait {symbol = @ctrl0}
    }
  }
}
