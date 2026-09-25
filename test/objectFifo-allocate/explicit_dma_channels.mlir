// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --verify-diagnostics %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An explicit flow or packet flow ending at a tile's DMA, and a task the
// runtime sequence configures on a channel by index, each hold that DMA channel
// even though no DMA body in the device starts it. Allocation works around
// them rather than handing the same channel to an objectFIFO.

// CHECK-LABEL: @reserved_by_flow_and_task
// CHECK: aie.objectfifo.dma_endpoint @in_fifo(%{{.*}}) fills @p {channelIndex = 2 : i32}
// CHECK: aie.objectfifo.dma_endpoint @out_fifo(%{{.*}}) drains @p {channelIndex = 2 : i32}
module @reserved_by_flow_and_task {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %ring = aie.buffer(%mem) {sym_name = "ring"} : memref<64xi32>
    // S2MM 0 and MM2S 0 through circuit flows, MM2S 1 through a packet flow.
    aie.flow(%shim, DMA : 0, %mem, DMA : 0)
    aie.flow(%mem, DMA : 0, %core, DMA : 0)
    aie.packet_flow(1) {
      aie.packet_source<%mem, DMA : 1>
      aie.packet_dest<%core, DMA : 1>
    }
    aie.objectfifo.pool @p(%mem) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @in_fifo(%mem) fills @p
    aie.objectfifo.dma_endpoint @out_fifo(%mem) drains @p
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // S2MM 1 through the runtime sequence only.
      %t = aiex.dma_configure_task(%mem, S2MM, 1) {
        aie.dma_bd(%ring : memref<64xi32> len = 64)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}

// -----

// A pinned endpoint may name a flow's channel: it is then the DMA draining
// that flow, which is also how allocation's own output reads when it reruns.
// CHECK-LABEL: @pinned_on_flow
// CHECK: aie.objectfifo.dma_endpoint @pinned(%{{.*}}) fills @p {channelIndex = 3 : i32}
// CHECK: aie.objectfifo.dma_endpoint @free(%{{.*}}) fills @p {channelIndex = 1 : i32}
module @pinned_on_flow {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.flow(%shim, DMA : 0, %mem, DMA : 0)
    aie.flow(%shim, DMA : 1, %mem, DMA : 3)
    aie.objectfifo.pool @p(%mem) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @pinned(%mem) fills @p {channelIndex = 3 : i32}
    aie.objectfifo.dma_endpoint @free(%mem) fills @p
  }
}

// -----

// A pinned endpoint on a runtime task's channel names the task.
module @pinned_on_task {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %ring = aie.buffer(%mem) {sym_name = "ring"} : memref<64xi32>
    aie.objectfifo.pool @p(%mem) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error @+1 {{pinned MM2S DMA channel 5 is out of range or already in use on this tile}}
    aie.objectfifo.dma_endpoint @out_fifo(%mem) drains @p {channelIndex = 5 : i32}
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // expected-note @+1 {{pre-existing aiex.dma_configure_task reserves DMA channel 5}}
      %t = aiex.dma_configure_task(%mem, MM2S, 5) {
        aie.dma_bd(%ring : memref<64xi32> len = 64)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}
