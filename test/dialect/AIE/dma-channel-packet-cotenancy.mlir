//===- dma-channel-packet-cotenancy.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DMAChannelAnalysis keys `usedChannels` on occupant KIND (Circuit vs.
// Packet) instead of a plain used/free bit, so a physical channel can
// co-tenant several packet flows while staying exclusive for circuit flows.
// `--aie-objectfifo-allocate`'s `assignChannels` now threads the real
// packet-ness of each endpoint (read off the owning `aie.route`'s `$packet`
// UnitAttr) into both `reservePinnedChannel` and `getDMAChannelIndex`, so
// packet-onto-packet co-tenancy is reachable; a circuit-occupied slot still
// rejects everything, packet or circuit.
//
// Splits, in source order:
//  1. auto_assign_two_distinct: an ordinary unpinned two-shim-fifo case still
//     lands on two distinct channels (0 and 1) -- the kinded map does not
//     regress plain allocation.
//  2. pinned_packet_circuit_collision: a circuit-pinned source and a
//     packet-routed source pinned to the SAME shim MM2S channel still
//     collide -- a circuit slot rejects a packet claim (and, by the same
//     occupant check, any other circuit claim), so circuit exclusivity is
//     preserved even though the allocator now knows about packet-ness.
//  3. packet_cotenancy_reachable: two endpoints on `{packet}`-routed flows
//     pinned to the same channel now CO-TENANT successfully: both
//     `aie.packet_flow` ops source from the identical shim channel 0, no
//     "already in use" error.

// RUN: not aie-opt -split-input-file --aie-objectfifo-allocate %s 2>&1 | FileCheck %s

// Diagnostics (stderr) for the one remaining failing split are flushed
// before the successful splits' module dumps (stdout), so under `2>&1` the
// error appears first regardless of split order.

// CHECK: pinned MM2S DMA channel 0 is out of range or already in use on this tile
// CHECK: fifoName = "pkt_b"
// CHECK: aie.shim_dma_allocation @fifo_a_shim_alloc({{.*}}, MM2S, 0)
// CHECK: aie.shim_dma_allocation @fifo_b_shim_alloc({{.*}}, MM2S, 1)
// CHECK: aie.packet_source<[[SHIM:%[A-Za-z0-9_]+]], DMA : 0>
// CHECK: aie.packet_source<[[SHIM]], DMA : 0>

module @auto_assign_two_distinct {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %t1 = aie.tile(0, 2)
    %t2 = aie.tile(0, 3)

    aie.route_endpoint @src_a(%shim) DMA {fifoName = "fifo_a"}
    aie.route_endpoint @dst_a(%t1) Core {channelIndex = 0 : i32}
    aie.route from @src_a to [@dst_a]

    aie.route_endpoint @src_b(%shim) DMA {fifoName = "fifo_b"}
    aie.route_endpoint @dst_b(%t2) Core {channelIndex = 0 : i32}
    aie.route from @src_b to [@dst_b]
  }
}

// -----

// A circuit source pinned to shim MM2S channel 0 claims it as a Circuit
// occupant; a packet-routed source pinned to the SAME channel must still
// collide, because `reservePinnedChannel` only admits a co-tenant when the
// incoming claim AND the existing occupant are both Packet. This proves a
// circuit slot rejects everything -- packet or circuit -- even now that
// packet-ness is threaded through.
module @pinned_packet_circuit_collision {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %t1 = aie.tile(0, 2)
    %t2 = aie.tile(0, 3)

    aie.route_endpoint @src_a(%shim) DMA {channelIndex = 0 : i32, fifoName = "circuit_a"}
    aie.route_endpoint @dst_a(%t1) Core {channelIndex = 0 : i32}
    aie.route from @src_a to [@dst_a]

    aie.route_endpoint @src_b(%shim) DMA {channelIndex = 0 : i32, fifoName = "pkt_b"}
    aie.route_endpoint @dst_b(%t2) Core {channelIndex = 1 : i32}
    aie.route from @src_b to [@dst_b] {packet}
  }
}

// -----

// Two endpoints on `{packet}`-routed flows pinned to the same shim MM2S
// channel now co-tenant: `assignChannels` reads both routes' `$packet` and
// passes `isPacket=true` into `reservePinnedChannel` for each, so the second
// claim is admitted onto the first's Packet occupant instead of colliding.
module @packet_cotenancy_reachable {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %t1 = aie.tile(0, 2)
    %t2 = aie.tile(0, 3)

    aie.route_endpoint @src_a(%shim) DMA {channelIndex = 0 : i32, fifoName = "pkt_a"}
    aie.route_endpoint @dst_a(%t1) Core {channelIndex = 0 : i32}
    aie.route from @src_a to [@dst_a] {packet}

    aie.route_endpoint @src_b(%shim) DMA {channelIndex = 0 : i32, fifoName = "pkt_b"}
    aie.route_endpoint @dst_b(%t2) Core {channelIndex = 1 : i32}
    aie.route from @src_b to [@dst_b] {packet}
  }
}
