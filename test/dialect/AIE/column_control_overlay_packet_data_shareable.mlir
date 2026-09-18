//===- column_control_overlay_packet_data_shareable.mlir -------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true emit-standalone-overlay=false" %s | FileCheck %s
// RUN: aie-opt --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true emit-standalone-overlay=false" %s | FileCheck %s --check-prefix=NOALLOC

// A packet-data shim input (aie.packet_flow + a data aie.shim_dma_allocation, no
// aie.flow) is NOT circuit-occupied: control ingress may SHARE its channel,
// unlike the circuit case in column_control_overlay_circuit_cooccupied.mlir.
//
// To make the shareability observable, BOTH shim MM2S channels (0 and 1) carry
// packet data, leaving no free channel. Control ingress must then ride the
// chan-0 data trunk (DMA : 0 -> TileControl) rather than fail or mint its own
// channel, and col0 must get NO private @ctrlpkt_col0_mm2s_chan* allocation --
// it shares @packet_in0. (With a free channel available control would prefer
// that free channel; forcing the contention is what exercises the shareable
// path. A regression treating packet data as unshareable would fail to route
// here or mint a private control channel.)

// Data allocs on both channels, then control rides the chan-0 (DMA : 0) trunk:
// CHECK: aie.shim_dma_allocation @packet_in1(%{{.*}}, MM2S, 1)
// CHECK: aie.packet_source<%{{.*}}, DMA : 0>
// CHECK-NEXT: aie.packet_dest<%{{.*}}, TileControl : 0>

// ...and no private control allocation is minted on col0 (it shares the data
// channel). Scoped whole-file via its own prefix so an out-of-window mint is
// still caught:
// NOALLOC-NOT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  aie.packet_flow(0) {
    aie.packet_source<%tile_0_0, DMA : 0>
    aie.packet_dest<%tile_0_1, DMA : 0>
  }
  aie.packet_flow(1) {
    aie.packet_source<%tile_0_0, DMA : 1>
    aie.packet_dest<%tile_0_1, DMA : 1>
  }
  aie.shim_dma_allocation @packet_in0(%tile_0_0, MM2S, 0)
  aie.shim_dma_allocation @packet_in1(%tile_0_0, MM2S, 1)
}
