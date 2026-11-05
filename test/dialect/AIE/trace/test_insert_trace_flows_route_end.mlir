//===- test_insert_trace_flows_route_end.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -aie-insert-trace-flows %s 2>&1 | FileCheck %s --check-prefix=FULL
// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-insert-trace-flows{lateral-routing=true},aie-objectFifo-stateful-transform))' %s | FileCheck %s --check-prefix=LATERAL

// An aie.route DMA end and an objectFIFO both end at shim (0, 0), and their
// lowering, after this pass, takes both its S2MM channels. The trace counts
// the route end as it counts the objectFIFO, so it reports the full shim
// rather than taking a channel allocation then finds short, and with
// lateral routing it moves to spare column 1, which allocation leaves free.
// FULL:    error: no S2MM channels available on shim tile at column 0
// LATERAL: aie.packet_dest<%shim_noc_tile_1_0, DMA : 1>
// LATERAL: aie.shim_dma_allocation @out0__shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
// LATERAL: aie.shim_dma_allocation @b__shim_alloc(%shim_noc_tile_0_0, S2MM, 1)
module @route_end_at_shim {
  aie.device(npu1_2col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)
    aie.objectfifo @out0(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %b = aie.buffer(%tile02) {sym_name = "b"} : memref<16xi32>
    aie.route_endpoint @prog(%tile02) DMA
    aie.route_endpoint @host(%tile00) DMA {fifoName = "b"}
    aie.route from @prog to [@host]
    aie.mem(%tile02) {
      aie.dma_start(MM2S, @prog, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%b : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.core(%tile02) {
      %0 = aie.objectfifo.acquire @out0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out0(Produce, 1)
      aie.end
    }
    aie.trace @trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace
    }
  }
}
