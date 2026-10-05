//===- column_control_overlay_requests_routed.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// With route-shim-to-tile-ctrl=true, a routed response alone does not complete
// the overlay: the shim DMA to TileControl requests are still generated. Once
// the requests are routed as well, a further run adds nothing.

// RUN: aie-opt %s -aie-generate-column-control-overlay -aie-create-pathfinder-flows | aie-opt -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" | FileCheck %s --check-prefix=RESPONSE
// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" -aie-create-pathfinder-flows | aie-opt -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" | FileCheck %s --check-prefix=ROUTED

// RESPONSE-DAG: %[[MEM:.*]] = aie.tile(0, 1)
// RESPONSE-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
// RESPONSE-DAG: %[[COMPUTE:.*]] = aie.tile(0, 2)
// RESPONSE-NOT: aie.packet_source<%{{.*}}, TileControl : 0>
// RESPONSE:      aie.packet_source<%[[SHIM]], DMA : 0>
// RESPONSE-NEXT: aie.packet_dest<%[[SHIM]], TileControl : 0>
// RESPONSE:      aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// RESPONSE:      aie.packet_source<%[[SHIM]], DMA : 0>
// RESPONSE-NEXT: aie.packet_dest<%[[MEM]], TileControl : 0>
// RESPONSE:      aie.packet_source<%[[SHIM]], DMA : 0>
// RESPONSE-NEXT: aie.packet_dest<%[[COMPUTE]], TileControl : 0>
// RESPONSE-NOT: aie.packet_source<%{{.*}}, TileControl : 0>

// ROUTED:     aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// ROUTED-NOT: aie.shim_dma_allocation
// ROUTED-NOT: aie.packet_flow

aie.device(npu1_1col) {
  %shim = aie.tile(0, 0)
  %compute = aie.tile(0, 2)
}
