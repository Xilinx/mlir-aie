//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Vector-scalar multiplication with event trace functionality on NPU.
// This tests basic trace configuration and data capture
//
// trace components:
// 1. aie.packet_flow - Routes trace packets from compute tiles to shim DMA
// 2. aiex.npu.write32 - Configures trace control registers
// 3. aiex.npu.writebd - Sets up buffer descriptor for trace data capture
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    // External kernel function declaration
    func.func private @vector_scalar_mul_aie_scalar(memref<1024xi32>, memref<1024xi32>, memref<1xi32>, i32)

    // Tile declarations
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // ObjectFIFOs for data movement
    aie.objectfifo @in(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @infactor(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1xi32>>
    aie.objectfifo @out(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>

    // Core computation
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @infactor(Consume, 1) : !aie.objectfifosubview<memref<1xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
        %c0_0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4 step %c1_1 {
          %2 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<1024xi32>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
          %4 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @vector_scalar_mul_aie_scalar(%5, %3, %1, %c1024_i32) : (memref<1024xi32>, memref<1024xi32>, memref<1xi32>, i32) -> ()
          aie.objectfifo.release @in(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
        aie.objectfifo.release @infactor(Consume, 1)
      }
      aie.end
    } {link_with = "vector_scalar_mul.o"}

    // ========================================================================
    // Trace Packet Flow Configuration
    // ========================================================================

    // Packet flows to route trace data from compute tile to shim DMA
    // Flow 1: Route trace with id=1 from compute tile (0,2) to shim tile (0,0) DMA channel 1
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}

    // Flow 2: Route trace with id=2 from shim tile itself to DMA channel 1
    aie.packet_flow(2) {
      aie.packet_source<%shim_noc_tile_0_0, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}

    // Runtime sequence with trace configuration
    aiex.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<1xi32>, %arg2: memref<4096xi32>) {

      // ========================================================================
      // Trace Control Register Configuration
      // ========================================================================

      // Configure trace unit for compute tile (0,2)
      // Address 213200 (0x340D0): Trace_Control_0
      // Value enables trace with specific event selection
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 2038038528 : ui32}

      // Address 213204 (0x340D4): Trace_Control_1
      // Value configures trace mode and packet generation
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 1 : ui32}

      // Address 213216 (0x340E0): Trace_Event_0
      // Configures which events to trace (events 0-3)
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 1260724769 : ui32}

      // Address 213220 (0x340E4): Trace_Event_1
      // Configures which events to trace (events 4-7)
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 439168079 : ui32}

      // Address 261888 (0x3FF00): Stream_Switch_Event_Port_Selection_0
      // Select Stream Switch Ports for event generation
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 289 : ui32}
      // Address 261892 (0x3FF04): Stream_Switch_Event_Port_Selection_1
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}

      // Address 212992 (0x34000): Timer_Control
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}

      // Configure trace unit for shim tile (0,0)
      // Address 213200 (0x340D0): Trace_Control_0
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 0 : i32, value = 2122252288 : ui32}
      // Address 213204 (0x340D4): Trace_Control_1
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 0 : i32, value = 8194 : ui32}
      // Address 213216 (0x340E0): Trace_Event_0
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 0 : i32, value = 370151182 : ui32}
      // Address 213220 (0x340E4): Trace_Event_1
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 0 : i32, value = 522065943 : ui32}
      // Address 212992 (0x34000): Timer_Control
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 0 : i32, value = 32512 : ui32}

      // ========================================================================
      // Trace Buffer Descriptor and DMA Configuration
      // ========================================================================

      // Configure buffer descriptor 15 for trace data capture
      aiex.npu.writebd {
        bd_id = 15 : i32,
        buffer_length = 8192 : i32,      // 8KB trace buffer
        buffer_offset = 0 : i32,
        burst_length = 64 : i32,
        column = 0 : i32,
        d0_size = 0 : i32,
        d0_stride = 0 : i32,
        d0_zero_after = 0 : i32,
        d0_zero_before = 0 : i32,
        d1_size = 0 : i32,
        d1_stride = 0 : i32,
        d1_zero_after = 0 : i32,
        d1_zero_before = 0 : i32,
        d2_size = 0 : i32,
        d2_stride = 0 : i32,
        d2_zero_after = 0 : i32,
        d2_zero_before = 0 : i32,
        enable_packet = 1 : i32,         // Enable packet mode for trace
        iteration_current = 0 : i32,
        iteration_size = 0 : i32,
        iteration_stride = 0 : i32,
        lock_acq_enable = 0 : i32,
        lock_acq_id = 0 : i32,
        lock_acq_val = 0 : i32,
        lock_rel_id = 0 : i32,
        lock_rel_val = 0 : i32,
        next_bd = 0 : i32,
        out_of_order_id = 0 : i32,
        packet_id = 0 : i32,
        packet_type = 0 : i32,
        row = 0 : i32,
        use_next_bd = 0 : i32,
        valid_bd = 1 : i32
      }

      // Patch the trace buffer address (arg_idx = 4 corresponds to the 5th XRT buffer)
      // Address 119268 (0x1D1E4): Buffer descriptor 15 address field
      aiex.npu.address_patch {addr = 119268 : ui32, arg_idx = 4 : i32, arg_plus = 0 : i32}

      // Configure DMA channel 1 for trace data transfer
      // Address 119304 (0x1D208): DMA_S2MM_1_Control
      aiex.npu.maskwrite32 {address = 119304 : ui32, column = 0 : i32, mask = 7936 : ui32, row = 0 : i32, value = 3840 : ui32}
      // Address 119308 (0x1D20C): DMA_S2MM_1_Queue
      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483663 : ui32}

      // Start trace control
      // Address 212992 (0x34000): Timer_Control
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 0 : i32, value = 32512 : ui32}
      // Address 213068 (0x3404C): Event_Broadcast_15
      aiex.npu.write32 {address = 213068 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      // Address 213000 (0x34008): Event_Generate
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}

      // ========================================================================
      // Kernel Data Transfer
      // ========================================================================

      // Configure DMA tasks for input, factor, and output
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}

      %1 = aiex.dma_configure_task_for @infactor {
        aie.dma_bd(%arg1 : memref<1xi32>, 0, 1, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}

      %2 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg2 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}

      // Start and await data transfer tasks
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%2)

      // ========================================================================
      // Trace Epilogue
      // ========================================================================

      // Flush trace data by writing trace done event
      // Address 213064 (0x34048): Event_Broadcast_14
      aiex.npu.write32 {address = 213064 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
      // Address 213000 (0x34008): Event_Generate
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
    }
  }
}
