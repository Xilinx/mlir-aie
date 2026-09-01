//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Two traced designs behind aiex.configure. @dev_a emits INSTR_EVENT_0 and
// @dev_b emits INSTR_EVENT_1. Each device declares two runtime sequences that
// differ only in the runtime parameter, which sets how many events the core
// emits. @main dispatches six runs, so -aie-fuse-trace-buffers must give each
// run its own slice of one trace buffer.
//
// The runs alternate between the devices. Two runs of one device in a row share
// a trace buffer descriptor, because loading the PDI of the device that is
// already loaded reconfigures nothing, and the second run then appends to the
// first run's slice.
//
// Run order and event count per run:
//   @seq_a1 ->  7000 x INSTR_EVENT_0
//   @seq_b1 ->  8000 x INSTR_EVENT_1
//   @seq_a2 ->  9000 x INSTR_EVENT_0
//   @seq_b2 -> 10000 x INSTR_EVENT_1
//   @seq_a2 ->  9000 x INSTR_EVENT_0
//   @seq_b2 -> 10000 x INSTR_EVENT_1
//
//===----------------------------------------------------------------------===//

module {

  aie.device(npu2_1col) @main {

    // Each run writes the parameter it received into its own four-element
    // window of %out, which pins a slice to the run that produced it.
    aie.runtime_sequence @sequence(%out: memref<24xi32>) {

      aiex.configure @dev_a {
        %v0 = memref.subview %out[0] [4] [1] : memref<24xi32> to memref<4xi32, strided<[1], offset: 0>>
        %a0 = memref.reinterpret_cast %v0 to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 0>> to memref<4xi32>
        aiex.run @seq_a1 (%a0) : (memref<4xi32>)
      }

      aiex.configure @dev_b {
        %v1 = memref.subview %out[4] [4] [1] : memref<24xi32> to memref<4xi32, strided<[1], offset: 4>>
        %a1 = memref.reinterpret_cast %v1 to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 4>> to memref<4xi32>
        aiex.run @seq_b1 (%a1) : (memref<4xi32>)
      }

      aiex.configure @dev_a {
        %v2 = memref.subview %out[8] [4] [1] : memref<24xi32> to memref<4xi32, strided<[1], offset: 8>>
        %a2 = memref.reinterpret_cast %v2 to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 8>> to memref<4xi32>
        aiex.run @seq_a2 (%a2) : (memref<4xi32>)
      }

      aiex.configure @dev_b {
        %v3 = memref.subview %out[12] [4] [1] : memref<24xi32> to memref<4xi32, strided<[1], offset: 12>>
        %a3 = memref.reinterpret_cast %v3 to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 12>> to memref<4xi32>
        aiex.run @seq_b2 (%a3) : (memref<4xi32>)
      }

      aiex.configure @dev_a {
        %v4 = memref.subview %out[16] [4] [1] : memref<24xi32> to memref<4xi32, strided<[1], offset: 16>>
        %a4 = memref.reinterpret_cast %v4 to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 16>> to memref<4xi32>
        aiex.run @seq_a2 (%a4) : (memref<4xi32>)
      }

      aiex.configure @dev_b {
        %v5 = memref.subview %out[20] [4] [1] : memref<24xi32> to memref<4xi32, strided<[1], offset: 20>>
        %a5 = memref.reinterpret_cast %v5 to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 20>> to memref<4xi32>
        aiex.run @seq_b2 (%a5) : (memref<4xi32>)
      }

    }

  }

  aie.device(npu2_1col) @dev_a {

    func.func private @emit_events_0(i32) attributes {link_with = "kernel.o"}

    %shim_a = aie.tile(0, 0)
    %core_a = aie.tile(0, 2)

    %rtp_a = aie.buffer(%core_a) {sym_name = "rtp_a"} : memref<1xi32>
    %sync_a = aie.lock(%core_a, 0) {init = 0 : i32, sym_name = "sync_a"}

    aie.objectfifo @out_a (%core_a, {%shim_a}, 1 : i32) : !aie.objectfifo<memref<4xi32>>

    aie.core(%core_a) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %cmax = arith.constant 0xFFFFFE : index

      scf.for %iter = %c0 to %cmax step %c1 {
        // The runtime sequence writes the parameter, then arms this lock.
        // AcquireGreaterEqual takes the lock back down to 0, so the next
        // iteration waits for the next dispatch.
        aie.use_lock(%sync_a, AcquireGreaterEqual, %c1_i32)
        %n = memref.load %rtp_a[%c0] : memref<1xi32>
        %elem = aie.objectfifo.acquire @out_a (Produce, 1) : memref<4xi32>
        func.call @emit_events_0(%n) : (i32) -> ()
        memref.store %n, %elem[%c0] : memref<4xi32>
        aie.objectfifo.release @out_a (Produce, 1)
      }
      aie.end
    }

    aie.trace @trace_a(%core_a) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.event<"INSTR_EVENT_1">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence @seq_a1(%out: memref<4xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_a
      %n = arith.constant 7000 : i32
      aiex.npu.rtp_write(@rtp_a, 0, %n) : i32
      aiex.set_lock(%sync_a, 1)
      %t = aiex.dma_configure_task_for @out_a {
        aie.dma_bd(%out : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }

    aie.runtime_sequence @seq_a2(%out: memref<4xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_a
      %n = arith.constant 9000 : i32
      aiex.npu.rtp_write(@rtp_a, 0, %n) : i32
      aiex.set_lock(%sync_a, 1)
      %t = aiex.dma_configure_task_for @out_a {
        aie.dma_bd(%out : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }

  }

  aie.device(npu2_1col) @dev_b {

    func.func private @emit_events_1(i32) attributes {link_with = "kernel.o"}

    %shim_b = aie.tile(0, 0)
    %core_b = aie.tile(0, 2)

    %rtp_b = aie.buffer(%core_b) {sym_name = "rtp_b"} : memref<1xi32>
    %sync_b = aie.lock(%core_b, 0) {init = 0 : i32, sym_name = "sync_b"}

    aie.objectfifo @out_b (%core_b, {%shim_b}, 1 : i32) : !aie.objectfifo<memref<4xi32>>

    aie.core(%core_b) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %cmax = arith.constant 0xFFFFFE : index

      scf.for %iter = %c0 to %cmax step %c1 {
        aie.use_lock(%sync_b, AcquireGreaterEqual, %c1_i32)
        %n = memref.load %rtp_b[%c0] : memref<1xi32>
        %elem = aie.objectfifo.acquire @out_b (Produce, 1) : memref<4xi32>
        func.call @emit_events_1(%n) : (i32) -> ()
        memref.store %n, %elem[%c0] : memref<4xi32>
        aie.objectfifo.release @out_b (Produce, 1)
      }
      aie.end
    }

    aie.trace @trace_b(%core_b) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.event<"INSTR_EVENT_1">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence @seq_b1(%out: memref<4xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_b
      %n = arith.constant 8000 : i32
      aiex.npu.rtp_write(@rtp_b, 0, %n) : i32
      aiex.set_lock(%sync_b, 1)
      %t = aiex.dma_configure_task_for @out_b {
        aie.dma_bd(%out : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }

    aie.runtime_sequence @seq_b2(%out: memref<4xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_b
      %n = arith.constant 10000 : i32
      aiex.npu.rtp_write(@rtp_b, 0, %n) : i32
      aiex.set_lock(%sync_b, 1)
      %t = aiex.dma_configure_task_for @out_b {
        aie.dma_bd(%out : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }

  }

}
