//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// End-to-end test for distribute-channels and lateral-routing in the
// -aie-insert-trace-flows pass.
//
// Design: 2-column NPU1 (Phoenix/AIE2). Column 0 has an active core
// performing vector-scalar multiplication. Column 1 is spare (no core).
//
// Two traces on the compute tile:
//   @core_trace  -- core execution events (INSTR_EVENT_0/1, LOCK_STALL, ...)
//   @mem_trace   -- memory DMA events (S2MM/MM2S start/finish)
//
// With -aie-insert-trace-flows="distribute-channels=true lateral-routing=true":
//   - Lateral routing redirects traces from column 0 shim to column 1 shim
//   - Channel distribution splits the two traces across S2MM channels 0 and 1
//   - Both channels share arg_idx=4, split by offset within a 2x trace buffer
//     (channel 0 at offset 0, channel 1 at offset buffer_size)
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_2col) {
    // External kernel function declaration
    func.func private @vector_scalar_mul_aie_scalar(
        memref<1024xi32>, memref<1024xi32>, memref<1xi32>, i32
    ) attributes {link_with = "vector_scalar_mul.o"}

    // Tile declarations: column 0 active, column 1 spare
    %shim_0 = aie.tile(0, 0)
    %mem_0  = aie.tile(0, 1)
    %core_0 = aie.tile(0, 2)

    // ObjectFIFOs for data movement (column 0 only)
    aie.objectfifo @in(%shim_0, {%mem_0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @in_fwd(%mem_0, {%core_0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in] -> [@in_fwd]([] [0])

    aie.objectfifo @infactor(%shim_0, {%mem_0}, 2 : i32) : !aie.objectfifo<memref<1xi32>>
    aie.objectfifo @infactor_fwd(%mem_0, {%core_0}, 2 : i32) : !aie.objectfifo<memref<1xi32>>
    aie.objectfifo.link [@infactor] -> [@infactor_fwd]([] [0])

    aie.objectfifo @out(%core_0, {%mem_0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out_fwd(%mem_0, {%shim_0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@out] -> [@out_fwd]([] [0])

    // Core computation (column 0, row 2)
    %c = aie.core(%core_0) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c_inf step %c1 {
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
          func.call @vector_scalar_mul_aie_scalar(%5, %3, %1, %c1024_i32)
              : (memref<1024xi32>, memref<1024xi32>, memref<1xi32>, i32) -> ()
          aie.objectfifo.release @in_fwd(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
        aie.objectfifo.release @infactor_fwd(Consume, 1)
      }
      aie.end
    }

    // ====================================================================
    // TRACE CONFIGURATION
    // ====================================================================

    // Trace 1: Core execution events on compute tile (0,2)
    aie.trace @core_trace(%core_0) {
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

    // Trace 2: Memory DMA events on compute tile (0,2)
    aie.trace @mem_trace(%core_0) {
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

    // ====================================================================
    // RUNTIME SEQUENCE WITH TRACE ACTIVATION
    // ====================================================================

    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<1xi32>,
                         %arg2: memref<4096xi32>) {

      // Trace: 8192 bytes per channel, 2 channels = 16384 total at arg_idx=4.
      // The distribute pass splits this into two 8192-byte regions by offset.
      aie.trace.host_config buffer_size = 8192

      aie.trace.start_config @core_trace
      aie.trace.start_config @mem_trace

      // Data transfers
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi32>, 0, 4096,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 4096, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}

      %1 = aiex.dma_configure_task_for @infactor {
        aie.dma_bd(%arg1 : memref<1xi32>, 0, 1,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 1, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}

      %2 = aiex.dma_configure_task_for @out_fwd {
        aie.dma_bd(%arg2 : memref<4096xi32>, 0, 4096,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 4096, stride = 1>])
          {burst_length = 0 : i32}
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
