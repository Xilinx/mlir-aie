//===- aiecc_auto_packetize.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Auto-packetization in aiecc, under --load-pdi-to-ctrl-pkt.
//
// Without the packet flows below, the design's two inputs hold both shim MM2S
// channels of column 0 as circuits. By default one becomes packet-switched and
// shares channel 0 with the control packets; with --ctrlpkt-auto-packetize=false
// the overlay has no channel.

// RUN: rm -rf %t && mkdir -p %t/circuit %t/circuit_off %t/retry %t/pinned
// RUN: sed -e '/aie.packet_flow/d' -e 's/ {packet}//' %s > %t/circuit.mlir
// RUN: cd %t/circuit && %aiecc --load-pdi-to-ctrl-pkt --get=input_physical.mlir %t/circuit.mlir
// RUN: FileCheck %s --check-prefix=CIRCUIT < %t/circuit/input_physical.mlir
// RUN: cd %t/circuit_off && not %aiecc --load-pdi-to-ctrl-pkt --ctrlpkt-auto-packetize=false --get=input_physical.mlir %t/circuit.mlir 2>&1 | FileCheck %s --check-prefix=OFF

// CIRCUIT-DAG: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%{{.+}}, MM2S, 0)
// CIRCUIT-DAG: aie.shim_dma_allocation @in1_shim_alloc(%{{.+}}, MM2S, 0, <pkt_type = 0, pkt_id = {{[0-9]+}}>)
// CIRCUIT-DAG: aie.shim_dma_allocation @in0_shim_alloc(%{{.+}}, MM2S, 1)

// OFF: error: {{.*}}every shim mm2s dma channel of column 0 moves data, so none is left for its control packets

// As written, both inputs are packet-switched and the packet flows below leave
// shim DMA 0 for eleven destinations. Control on channel 0, beside @in0 and
// those flows, does not route (the pinned run, where the pass is told channel 0
// and aiecc is told not to retry); aiecc retries column 0 on channel 1, which
// routes.

// RUN: cd %t/retry && %aiecc --load-pdi-to-ctrl-pkt --get=input_physical.mlir %s
// RUN: FileCheck %s --check-prefix=RETRIED < %t/retry/input_physical.mlir
// RUN: aie-opt --aie-auto-packetize-shim-input="control-channels=0:0" %s -o %t/pinned.mlir
// RUN: cd %t/pinned && not %aiecc --load-pdi-to-ctrl-pkt --ctrlpkt-auto-packetize=false --get=input_physical.mlir %t/pinned.mlir 2>&1 | FileCheck %s --check-prefix=PINNED

// RETRIED-DAG: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1(%{{.+}}, MM2S, 1)
// RETRIED-DAG: aie.shim_dma_allocation @in0_shim_alloc(%{{.+}}, MM2S, 1, <pkt_type = 0, pkt_id = {{[0-9]+}}>)
// RETRIED-DAG: aie.shim_dma_allocation @in1_shim_alloc(%{{.+}}, MM2S, 0, <pkt_type = 0, pkt_id = {{[0-9]+}}>)

// PINNED: error: Unable to find a legal routing

module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %comp = aie.tile(0, 2)
    %t3 = aie.tile(0, 3)
    %t4 = aie.tile(0, 4)
    %t5 = aie.tile(0, 5)
    aie.objectfifo @in0 (%shim, {%comp}, 1 : i32) {packet} : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @in1 (%shim, {%comp}, 1 : i32) {packet} : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @out (%comp, {%shim}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.core(%comp) {
      %e0 = aie.objectfifo.acquire @in0 (Consume, 1) : memref<4xi32>
      %e1 = aie.objectfifo.acquire @in1 (Consume, 1) : memref<4xi32>
      %eo = aie.objectfifo.acquire @out (Produce, 1) : memref<4xi32>
      aie.objectfifo.release @in0 (Consume, 1)
      aie.objectfifo.release @in1 (Consume, 1)
      aie.objectfifo.release @out (Produce, 1)
      aie.end
    }
    aie.packet_flow(2) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%mem, DMA : 1> }
    aie.packet_flow(4) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%mem, DMA : 2> }
    aie.packet_flow(8) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%mem, DMA : 3> }
    aie.packet_flow(16) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%mem, DMA : 4> }
    aie.packet_flow(5) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%mem, DMA : 5> }
    aie.packet_flow(12) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%t3, DMA : 0> }
    aie.packet_flow(20) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%t4, DMA : 0> }
    aie.packet_flow(21) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%t5, DMA : 0> }
    aie.packet_flow(22) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%t3, DMA : 1> }
    aie.packet_flow(23) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%t4, DMA : 1> }
    aie.packet_flow(24) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%t5, DMA : 1> }
    aie.runtime_sequence @sequence(%a0 : memref<4xi32>, %a1 : memref<4xi32>, %argout : memref<4xi32>) {
      %t_in0 = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%a0 : memref<4xi32> offset = 0 len = 4)
        aie.end
      }
      %t_in1 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%a1 : memref<4xi32> offset = 0 len = 4)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%argout : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in0)
      aiex.dma_start_task(%t_in1)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_in0)
      aiex.dma_free_task(%t_in1)
      aiex.dma_free_task(%t_out)
    }
  }
}
