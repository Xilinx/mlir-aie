//===- aie_new_trace.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This is a version using the NEW PROTOTYPE trace syntax (aie.trace operations).
// Compare with aie_trace.mlir to see the difference between low-level manual
// configuration and high-level declarative trace API.
//
// This example uses:
// - aie.trace operation for declarative trace configuration
// - aie.trace.event for specifying events to capture
// - aie.trace.start_config in runtime sequence
//
// The passes aie-trace-to-config and aiex-inline-trace-config will lower this
// to the same aiex.npu.write32 operations as aie_trace.mlir.
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
    } {link_with = "scale.o"}

    // ========================================================================
    // TRACE CONFIGURATION
    // ========================================================================

    // Trace configuration for compute tile (0,2) - core events
    aie.trace @core_trace(%tile_0_2) {
      // Set trace mode (Event-Time captures timestamps)
      aie.trace.mode "Event-Time"

      // Configure packet routing (ID and type for packet-switched routing)
      aie.trace.packet id=1 type=core

      // Specify which events to capture (up to 8 events)
      // These are the same events as in the manual version
      aie.trace.event<"INSTR_EVENT_0">        // User event 0 (start marker)
      aie.trace.event<"INSTR_EVENT_1">        // User event 1 (end marker)
      aie.trace.event<"INSTR_VECTOR">         // Vector instructions
      aie.trace.event<"MEMORY_STALL">         // Memory access stalls
      aie.trace.event<"STREAM_STALL">         // Stream buffer stalls
      aie.trace.event<"LOCK_STALL">           // Lock acquisition stalls
      aie.trace.event<"PORT_RUNNING_1">       // DMA:0 slave port running
      aie.trace.event<"PORT_IDLE_1">       // DMA:1 master port running
      aie.trace.port<0> port=DMA channel=0 direction=S2MM
      aie.trace.port<1> port=DMA channel=0 direction=MM2S

      // Specify start/stop control (broadcast events)
      aie.trace.start event=<"BROADCAST_15">
      aie.trace.stop event=<"BROADCAST_14">
    }

    // Trace configuration for compute tile (0,2) - memory events
    aie.trace @mem_trace(%tile_0_2) {
      // Set trace mode (Event-Time captures timestamps)
      aie.trace.mode "Event-Time"

      // Configure packet routing (ID and type for packet-switched routing)
      aie.trace.packet id=3 type=mem

      // Specify which events to capture (up to 8 events)
      // These are the same events as in the manual version
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.event<"DMA_S2MM_1_START_TASK">
      aie.trace.event<"DMA_MM2S_0_START_TASK">
      aie.trace.event<"DMA_S2MM_0_FINISHED_TASK">
      aie.trace.event<"DMA_S2MM_1_FINISHED_TASK">
      aie.trace.event<"DMA_MM2S_0_FINISHED_TASK">
      aie.trace.event<"DMA_S2MM_0_STREAM_STARVATION">
      aie.trace.event<"DMA_S2MM_1_STREAM_STARVATION">

      // Specify start/stop control (broadcast events)
      aie.trace.start event=<"BROADCAST_15">
      aie.trace.stop event=<"BROADCAST_14">
    }

    // Trace configuration for shim tile (0,0)
    // Captures DMA activity at the interface to DDR
    aie.trace @shim_trace(%shim_noc_tile_0_0) {
      aie.trace.packet id=2 type=shimtile

      // Shim DMA events
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.event<"DMA_S2MM_1_START_TASK">
      aie.trace.event<"DMA_MM2S_0_START_TASK">
      aie.trace.event<"DMA_S2MM_0_FINISHED_TASK">
      aie.trace.event<"DMA_S2MM_1_FINISHED_TASK">
      aie.trace.event<"DMA_MM2S_0_FINISHED_TASK">
      aie.trace.event<"DMA_S2MM_0_STREAM_STARVATION">
      aie.trace.event<"DMA_S2MM_1_STREAM_STARVATION">

      aie.trace.start event=<"TRUE">
      aie.trace.stop event=<"NONE">
    }

    // Packet flows to route trace data (same as before)
    // These define the routing but the trace config is separate
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(3) {
      aie.packet_source<%tile_0_2, Trace : 1>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}

    aie.packet_flow(2) {
      aie.packet_source<%shim_noc_tile_0_0, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}

    // ========================================================================
    // RUNTIME SEQUENCE WITH TRACE ACTIVATION
    // ========================================================================

    // Runtime sequence with trace configuration
    aiex.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<1xi32>, %arg2: memref<4096xi32>) {

      // ========================================================================
      // TRACE INITIALIZATION (NEW API)
      // ========================================================================

      // Start trace configuration for core tile
      // This will be lowered to the aiex.npu.write32 operations automatically
      aie.trace.start_config @core_trace
      aie.trace.start_config @mem_trace

      // Start trace configuration for shim tile
      aie.trace.start_config @shim_trace

      // Address 212992 (0x34000): Timer_Control
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}

      // Configure trace buffer descriptor (still manual for now)
      // TODO: This could be automated based on trace configuration
      aiex.npu.writebd {
        bd_id = 15 : i32,
        buffer_length = 8192 : i32,
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
        enable_packet = 1 : i32,
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

      // Patch trace buffer address
      aiex.npu.address_patch {addr = 119268 : ui32, arg_idx = 4 : i32, arg_plus = 0 : i32}

      // Configure DMA channel for trace
      aiex.npu.maskwrite32 {address = 119304 : ui32, column = 0 : i32, mask = 7936 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483663 : ui32}

      // Start trace control
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}

      // ========================================================================
      // DATA TRANSFER CONFIGURATION (unchanged)
      // ========================================================================

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

      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%2)

      // ========================================================================
      // TRACE COMPLETION (unchanged)
      // ========================================================================

      aiex.npu.write32 {address = 213064 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
    }
  }
}
