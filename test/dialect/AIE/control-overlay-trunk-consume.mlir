//===- control-overlay-trunk-consume.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The column's row-0 shim tile carries a Stage-1 stamp
// `ctrl_pkt_trunk_chan = 1`. Stage-1 (AIEAutoPacketizeControlIngress) is
// authoritative: it unions the data-pin / shim-mux reservations across ALL
// configs and conforms each config to pin control's leg to K, and its
// "shareable channel" criterion co-tenants control onto a packet leg's channel.
// So the overlay must CONSUME the stamp and route control ingress on channel 1,
// even though its own chooseCtrlShimChan would independently pick channel 0
// (the lowest fully-free channel on this bare column). Consuming the stamp --
// rather than recompute-and-assert-agreement -- is what lets a single packet
// ingress leg (which Stage-1 co-tenants control onto) build instead of
// spuriously failing loud on a benign criterion divergence. Absent the stamp
// the overlay falls back to its own choice (see control-overlay-single-trunk.mlir).

// RUN: aie-opt -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" %s 2>&1 | FileCheck %s
// The consumed stamp lands on the tile attr and every control ingress flow.
// CHECK: ctrl_pkt_shim_chan = 1 : i32
// CHECK: aie.packet_source<%shim_noc_tile_0_0, DMA : 1>
// CHECK-NOT: aie.packet_source<%shim_noc_tile_0_0, DMA : 0>

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0) {ctrl_pkt_trunk_chan = 1 : i32}
  %tile_0_2 = aie.tile(0, 2)
}
