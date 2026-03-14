//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This example uses:
// - aie.trace operation for declarative trace configuration
// - aie.trace.event for specifying events to capture
// - aie.trace.start_config in runtime sequence
//
// This will be incrementally lowered by trace passes
//===----------------------------------------------------------------------===//

module {
  aie.device(npu2_1col) {
    // External kernel function declaration
    func.func private @vector_scalar_mul_aie_scalar(memref<1024xi32>, memref<1024xi32>, memref<1xi32>, i32) attributes {link_with = "vector_scalar_mul.o"}

    // Tile declarations
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    // ObjectFIFOs for data movement
    aie.objectfifo @in(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @in_fwd(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in] -> [@in_fwd]([] [0])

    aie.objectfifo @infactor(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1xi32>>
    aie.objectfifo @infactor_fwd(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1xi32>>
    aie.objectfifo.link [@infactor] -> [@infactor_fwd]([] [0])

    aie.objectfifo @out(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out_fwd(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@out] -> [@out_fwd]([] [0])

    // Core computation
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @infactor_fwd(Consume, 1) : !aie.objectfifosubview<memref<1xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
        %c0_0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4 step %c1_1 {
          %2 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<1024xi32>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
          %4 = aie.objectfifo.acquire @in_fwd(Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @vector_scalar_mul_aie_scalar(%5, %3, %1, %c1024_i32) : (memref<1024xi32>, memref<1024xi32>, memref<1xi32>, i32) -> ()
          aie.objectfifo.release @in_fwd(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
        aie.objectfifo.release @infactor_fwd(Consume, 1)
      }
      aie.end
    }

    // ========================================================================
    // TRACE CONFIGURATION
    // ========================================================================

    // Trace configuration for compute tile (0,2) - core events
    aie.trace @core_trace(%tile_0_2) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type=core

      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.event<"INSTR_EVENT_1">
      aie.trace.event<"INSTR_VECTOR">
      aie.trace.event<"PORT_RUNNING_0">
      aie.trace.event<"PORT_RUNNING_1">
      aie.trace.event<"INSTR_LOCK_ACQUIRE_REQ">
      aie.trace.event<"INSTR_LOCK_RELEASE_REQ">
      aie.trace.event<"LOCK_STALL">

      aie.trace.port<0> port=DMA channel=0 direction=S2MM
      aie.trace.port<1> port=DMA channel=0 direction=MM2S

      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    // Trace configuration for compute tile (0,2) - memory events
    aie.trace @mem_trace(%tile_0_2) {
      aie.trace.packet id=2 type=mem

      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.event<"DMA_S2MM_1_START_TASK">
      aie.trace.event<"DMA_MM2S_0_START_TASK">
      aie.trace.event<"DMA_S2MM_0_FINISHED_TASK">
      aie.trace.event<"DMA_S2MM_1_FINISHED_TASK">
      aie.trace.event<"DMA_MM2S_0_FINISHED_TASK">
      aie.trace.event<"DMA_S2MM_0_STREAM_STARVATION">
      aie.trace.event<"DMA_S2MM_1_STREAM_STARVATION">

      aie.trace.start event=<"BROADCAST_15">
      aie.trace.stop event=<"BROADCAST_14">
    }

    // Trace configuration for mem tile (0, 1)
    aie.trace @memtile_trace(%mem_tile_0_1) {
      aie.trace.packet id=3 type=memtile

      aie.trace.event<"PORT_RUNNING_0">
      aie.trace.event<"PORT_RUNNING_1">
      aie.trace.event<"PORT_RUNNING_2">
      aie.trace.event<"PORT_RUNNING_3">
      aie.trace.event<"PORT_RUNNING_4">
      aie.trace.event<"PORT_RUNNING_5">
      aie.trace.event<"PORT_RUNNING_6">
      aie.trace.event<"PORT_RUNNING_7">

      aie.trace.port<0> port=DMA channel=0 direction=MM2S
      aie.trace.port<1> port=DMA channel=1 direction=MM2S
      aie.trace.port<2> port=DMA channel=0 direction=S2MM
      aie.trace.port<3> port=DMA channel=1 direction=S2MM
      aie.trace.port<4> port=DMA channel=2 direction=S2MM
      aie.trace.port<5> port=DMA channel=3 direction=S2MM
      aie.trace.port<6> port=DMA channel=4 direction=S2MM
      aie.trace.port<7> port=DMA channel=5 direction=S2MM

      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    // Trace configuration for shim tile (0,0)
    aie.trace @shim_trace(%shim_noc_tile_0_0) {
      aie.trace.packet id=4 type=shimtile

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

    // ========================================================================
    // RUNTIME SEQUENCE WITH TRACE ACTIVATION
    // ========================================================================

    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<1xi32>, %arg2: memref<4096xi32>) {

      // ========================================================================
      // TRACE INITIALIZATION
      // ========================================================================

      aie.trace.start_config @core_trace
      aie.trace.start_config @mem_trace
      aie.trace.start_config @memtile_trace
      aie.trace.start_config @shim_trace

      // ========================================================================
      // DATA TRANSFER CONFIGURATION
      // ========================================================================

      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}

      %1 = aiex.dma_configure_task_for @infactor {
        aie.dma_bd(%arg1 : memref<1xi32>, 0, 1, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}

      %2 = aiex.dma_configure_task_for @out_fwd {
        aie.dma_bd(%arg2 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%1)
      aiex.dma_await_task(%2)
    }
  }
}
