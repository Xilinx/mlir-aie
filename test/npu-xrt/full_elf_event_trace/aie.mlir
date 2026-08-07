// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A trivial full-ELF trace design, structured with the aiex.configure /
// aiex.run reconfigure syntax (a top-level @main runtime sequence configures
// the @trace_dev device and invokes its runtime sequence).
//
// The core asserts INSTR_EVENT_0 once (event0()) at entry and INSTR_EVENT_1
// once (event1()) at exit, around a small passthrough loop. The trace
// configuration also captures the surrounding lock events so the trace unit
// accumulates enough packets to flush the two single INSTR_EVENT_0/1 markers
// out to the shim DMA (a lone event0/event1 pair by itself never fills a trace
// packet and would be lost).
//
// `reuse_output_buffer = true` is required here: aiex.run passes a fixed
// argument list to @trace_seq, so trace lowering cannot append a fresh
// trace-buffer argument (that would desync the aiex.run arity). Instead the
// trace data is written into the tail of the output buffer, past the 64 i32
// output words.

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%in : memref<64xi32>, %out : memref<64xi32>) {
      aiex.configure @trace_dev {
        aiex.run @trace_seq(%in, %out) : (memref<64xi32>, memref<64xi32>)
      }
    }
  }

  aie.device(npu2) @trace_dev {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 1 : i32) : !aie.objectfifo<memref<64xi32>>

    aie.trace @core_trace(%tile_0_2) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.event<"INSTR_EVENT_1">
      aie.trace.event<"INSTR_VECTOR">
      aie.trace.event<"INSTR_LOCK_ACQUIRE_REQ">
      aie.trace.event<"INSTR_LOCK_RELEASE_REQ">
      aie.trace.event<"LOCK_STALL">
      aie.trace.event<"PORT_RUNNING_0">
      aie.trace.event<"PORT_RUNNING_1">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %si = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>
      %ei = aie.objectfifo.subview.access %si[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
      %so = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<64xi32>>
      %eo = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
      aie.event(0)
      scf.for %i = %c0 to %c64 step %c1 {
        %v = memref.load %ei[%i] : memref<64xi32>
        memref.store %v, %eo[%i] : memref<64xi32>
      }
      aie.event(1)
      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @trace_seq(%in : memref<64xi32>, %out : memref<64xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32, reuse_output_buffer = true}
      aie.trace.start_config @core_trace
      %ti = aiex.dma_configure_task_for @of_in {
        aie.dma_bd(%in : memref<64xi32> offset = 0 len = 64)
        aie.end
      } {}
      %to = aiex.dma_configure_task_for @of_out {
        aie.dma_bd(%out : memref<64xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti)
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
    }
  }
}
