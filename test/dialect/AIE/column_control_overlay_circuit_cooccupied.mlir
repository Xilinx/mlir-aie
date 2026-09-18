//===- column_control_overlay_circuit_cooccupied.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true emit-standalone-overlay=false" %s | FileCheck %s
// RUN: aie-opt --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true emit-standalone-overlay=false" %s | FileCheck %s --check-prefix=NOSHARE

// A circuit objectFifo shim input co-emits both a circuit-switched aie.flow
// AND a data aie.shim_dma_allocation on the SAME shim MM2S channel (0) -- the
// alloc must not mask the circuit reservation. Occupancy-aware selection must
// still treat channel 0 as circuit-occupied and relocate control ingress to
// the free channel 1, instead of sharing it (a circuit-mode slave port cannot
// carry a second SlvPktEn control stream).

// CHECK: aie.tile(0, 1){{.*}}ctrl_pkt_shim_chan = 1
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1({{.*}}, MM2S, 1, <pkt_type = 0, pkt_id = 15>)

// Control must NOT land on the circuit-occupied channel 0 (it relocated to
// chan 1). Whole-file via its own prefix -- the previous single-line
// `CHECK-NOT: packet_source ... DMA : 0 ... TileControl` was vacuous (FileCheck
// is line-oriented; a source and its dest are never on one line).
// NOSHARE-NOT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
  aie.shim_dma_allocation @circuit_in(%tile_0_0, MM2S, 0)
}
