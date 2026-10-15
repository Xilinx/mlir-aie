//===- controller_ids.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The column control overlay routes control packets on the tile controller IDs
// (15 for an npu2 shim). aiecc adds it after objectFIFO lowering, so the
// lowering keeps those IDs off the design's packet flows. With IDs 0-14 taken,
// @in0 gets 16, not 15. @in0 leaves from column 1, so each column keeps a shim
// channel free for its control packets.

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s --check-prefix=PLAIN
// RUN: aie-opt --aie-objectFifo-stateful-transform="reserve-controller-packet-ids=true" %s | FileCheck %s --check-prefix=RESERVED
// RUN: rm -rf %t && mkdir -p %t && cd %t && %aiecc --load-pdi-to-ctrl-pkt --get=input_with_symbols.mlir %s
// RUN: FileCheck %s --check-prefix=RESERVED < %t/input_with_symbols.mlir

// PLAIN:        aie.packet_flow(15) {
// PLAIN-NEXT:     aie.packet_source<%shim_noc_tile_1_0, DMA : 0>
// RESERVED-NOT:   aie.packet_flow(15) {
// RESERVED:       aie.packet_flow(16) {
// RESERVED-NEXT:    aie.packet_source<%shim_noc_tile_1_0, DMA : 0>

module {
  aie.device(npu2) @design {
    %shim = aie.tile(0, 0)
    %comp = aie.tile(0, 2)
    %shim1 = aie.tile(1, 0)
    %comp1 = aie.tile(1, 2)
    aie.objectfifo @in0 (%shim1, {%comp1}, 1 : i32) {packet} : !aie.objectfifo<memref<4xi32>>
    aie.objectfifo @out (%comp1, {%shim1}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
    aie.core(%comp1) {
      %e0 = aie.objectfifo.acquire @in0 (Consume, 1) : memref<4xi32>
      %eo = aie.objectfifo.acquire @out (Produce, 1) : memref<4xi32>
      aie.objectfifo.release @in0 (Consume, 1)
      aie.objectfifo.release @out (Produce, 1)
      aie.end
    }
    aie.packet_flow(0) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(1) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(3) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(4) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(5) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(6) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(7) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(8) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(9) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(10) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(11) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(12) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(13) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }
    aie.packet_flow(14) { aie.packet_source<%shim, DMA : 0> aie.packet_dest<%comp, DMA : 1> }

    aie.runtime_sequence @sequence(%a0 : memref<4xi32>, %argout : memref<4xi32>) {
      %t_in0 = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%a0 : memref<4xi32> offset = 0 len = 4)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%argout : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in0)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_in0)
      aiex.dma_free_task(%t_out)
    }
  }
}
