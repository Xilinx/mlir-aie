//===- column_control_overlay_packet_data_shareable.mlir -------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true emit-standalone-overlay=false" %s | FileCheck %s

// A packet-data shim input (aie.packet_flow + a data aie.shim_dma_allocation,
// no aie.flow) is NOT circuit-occupied: it must remain shareable with control
// ingress on the same channel 0, unlike the circuit case in
// column_control_overlay_circuit_cooccupied.mlir.

// CHECK-NOT: aie.shim_dma_allocation @ctrlpkt{{.*}}MM2S, 1
// CHECK: packet_source{{.*}}DMA : 0

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  aie.packet_flow(0) {
    aie.packet_source<%tile_0_0, DMA : 0>
    aie.packet_dest<%tile_0_1, DMA : 0>
  }
  aie.shim_dma_allocation @packet_in(%tile_0_0, MM2S, 0)
}
